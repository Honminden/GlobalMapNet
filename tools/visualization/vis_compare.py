import copy
import datetime
import json
import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from shapely import ops, affinity
from shapely.geometry import box, LineString, MultiLineString, Polygon, MultiPolygon, Point, CAP_STYLE, JOIN_STYLE
from shapely.validation import explain_validity, make_valid
from pyquaternion import Quaternion
from PIL import Image
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from plugin.models.globalmapnet.map_utils.functional.ego import generate_patch_box


PATCH_SIZE = (60, 30)
PATCH_SIZE_LARGE = (100, 50)
NUSC_LOCATIONS = ['singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown', 'boston-seaport']

def get_prefix(maps):
    prefix = list(maps.keys())[0]
    return prefix[:prefix.rfind('_')]

def print_map(ax, map_elements, poses=None, rotate=True, delta_angle=np.pi / 2, score_thr=0.4, single_frame_idx=None, patch_size=(60, 30), linewidth=3.5, hide_traced_region=False):
    ax.grid(False)
    ax.axis('equal')
    ax.axis('off')

    if poses != None:
        patch_boxes = [generate_patch_box(patch_size, pose) for pose in poses]
        patch_trace = ops.unary_union(patch_boxes)
        if rotate:
            angle = Quaternion(poses[0][3:]).yaw_pitch_roll[0]
            rot_angle = -angle + delta_angle
            center = patch_trace.centroid.coords[0]

    for map_element in map_elements:
        if 'details' in map_element and map_element['details']['score'] < score_thr:
            continue
        coords = map_element['coords']
        if poses != None and rotate:
            line = LineString(coords)
            line = affinity.rotate(line, rot_angle, origin=center, use_radians=True)
            coords = list(line.coords)
        
        if map_element['category'] == 'road':
            ax.plot([coord[0] for coord in coords], [coord[1] for coord in coords], 'g-', linewidth=linewidth)
        elif map_element['category'] == 'lane':
            ax.plot([coord[0] for coord in coords], [coord[1] for coord in coords], 'r-', linewidth=linewidth)
        elif map_element['category'] == 'ped':
            ax.plot([coord[0] for coord in coords], [coord[1] for coord in coords], 'b-', linewidth=linewidth)

    if single_frame_idx == None:
        if poses != None and rotate:
            patch_trace = affinity.rotate(patch_trace, rot_angle, origin=center, use_radians=True)
        ax.plot(patch_trace.exterior.xy[0], patch_trace.exterior.xy[1], '-', color='gray', linewidth=linewidth)
    else:
        if not hide_traced_region:
            patch_box = generate_patch_box(patch_size, poses[single_frame_idx])
            patch_box = affinity.rotate(patch_box, rot_angle, origin=(center[0], center[1]), use_radians=True)
            ax.plot(patch_box.exterior.xy[0], patch_box.exterior.xy[1], '--', color='gray' , linewidth=linewidth, alpha=1)

def print_all_maps(scene_names, patch_size,
                    globalmapnet_maps, streammapnet_maps, gt_maps,
                    globalmapnet_prefix, streammapnet_prefix,
                    locations=None, is_nusc=True, save_path=None,
                    fig_width=30, fig_height=30, **kwargs):
    scene_num = len(scene_names)
    fig, axs = plt.subplots(scene_num, 3, figsize=(fig_width, fig_height * scene_num))
    for i in range(scene_num):
        scene_name = scene_names[i]
        if is_nusc:
            location = locations[i]

        # globalmapnet
        ax = axs[i][1] if scene_num > 1 else axs[1]
        globalmapnet_map = globalmapnet_maps[f'{globalmapnet_prefix}_{scene_name}']
        print_map(ax, globalmapnet_map['map_elements'], globalmapnet_map['poses'], patch_size=patch_size, **kwargs)
        ax.set_title(f'globalmapnet {scene_name}')

        # streammapnet
        ax = axs[i][0] if scene_num > 1 else axs[0]
        streammapnet_map = streammapnet_maps[f'{streammapnet_prefix}_{scene_name}']
        print_map(ax, streammapnet_map['map_elements'], streammapnet_map['poses'], patch_size=patch_size, **kwargs)
        ax.set_title(f'streammapnet {scene_name}')

        # gt
        ax = axs[i][2] if scene_num > 1 else axs[2]

        patch_boxes = [generate_patch_box(patch_size, pose) for pose in streammapnet_map['poses']]
        patch_trace = ops.unary_union(patch_boxes)
        if is_nusc:
            gt_map = gt_maps[location]
        else:
            gt_map = gt_maps[scene_name]
        gt_map_elements = list()
        for map_element in gt_map['map_elements']:
            line = LineString(map_element['coords'])
            new_line = line.intersection(patch_trace)
            if not new_line.is_empty:
                if new_line.geom_type == 'MultiLineString':
                    for new_polyline in new_line.geoms:
                        gt_map_elements.append({'category': map_element['category'], 'coords': list(new_polyline.coords)})
                else:
                    gt_map_elements.append({'category': map_element['category'], 'coords': list(new_line.coords)})

        print_map(ax, gt_map_elements, streammapnet_map['poses'], patch_size=patch_size, **kwargs)

    plt.tight_layout()
    if save_path.endswith('.eps'):
        plt.savefig(save_path, format='eps')
    elif save_path.endswith('.tiff'):
        plt.savefig(save_path, format='tiff')
    else:
        plt.savefig(save_path)

def vis_single_scene(globalmapnet_maps_path, streammapnet_maps_path, gt_maps_path, save_dir, patch_size, scene_num_print=1, save_format='png', is_nusc=True):
    with open(globalmapnet_maps_path, 'r') as f:
        globalmapnet_maps = json.load(f)
    with open(streammapnet_maps_path, 'r') as f:
        streammapnet_maps = json.load(f)
    gt_maps = dict()
    if is_nusc:
        for location in NUSC_LOCATIONS:
            with open(os.path.join(gt_maps_path, rf'{location}_gt.json'), 'r') as f:
                gt_maps[location] = json.load(f)
    else:
        with open(gt_maps_path, 'r') as f:
            gt_maps = json.load(f)

    globalmapnet_prefix = get_prefix(globalmapnet_maps)
    streammapnet_prefix = get_prefix(streammapnet_maps)

    scene_names = list(globalmapnet_maps.keys())
    scene_names = [scene_name.split('_')[-1] for scene_name in scene_names]
    if is_nusc:
        locations = [globalmapnet_maps[f'{globalmapnet_prefix}_{scene_name}']['meta_info']['location'] for scene_name in scene_names]

        for i in range((len(scene_names) + 1) // scene_num_print):
            scene_names_print = scene_names[i * scene_num_print: (i + 1) * scene_num_print]
            if len(scene_names_print) < 1:
                continue
            locations_print = locations[i * scene_num_print: (i + 1) * scene_num_print]
            print_all_maps(scene_names_print, patch_size, globalmapnet_maps, streammapnet_maps, gt_maps, globalmapnet_prefix, streammapnet_prefix, locations=locations_print,
                save_path=os.path.join(save_dir, f'{scene_names_print[0]}_compare.{save_format}'))
    else:
        for i in range((len(scene_names) + 1) // scene_num_print):
            scene_names_print = scene_names[i * scene_num_print: (i + 1) * scene_num_print]
            if len(scene_names_print) < 1:
                continue
            print_all_maps(scene_names_print, patch_size, globalmapnet_maps, streammapnet_maps, gt_maps, globalmapnet_prefix, streammapnet_prefix, is_nusc=False,
                save_path=os.path.join(save_dir, f'{scene_names_print[0]}_compare.{save_format}'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                    description='visualize vectorized global map')
    parser.add_argument(
        '--globalmapnet-path',
        default='./cache/global_maps/globalmapnet_maps.json',
        help='path of globalmapnet maps')
    parser.add_argument(
        '--streammapnet-path',
        default='./cache/global_maps/streammapnet_maps.json',
        help='path of streammapnet maps')
    parser.add_argument(
        '--gt-path',
        default='./cache/global_maps',
        help='path of gt global map')
    parser.add_argument(
        '--save-dir',
        default='./cache/vis',
        help='save dir of visualization')
    parser.add_argument('--format', type=str, help='save format of figure', choices=[
        'png',
        'tiff',
        'eps'],
        default='png',)
    parser.add_argument(
        '--is-nusc',
        action='store_true',
        help='whether the dataset is nuscenes')
    parser.add_argument(
        '--large',
        action='store_true',
        help='use large patch size (100x50)')

    args = parser.parse_args()
    vis_single_scene(args.globalmapnet_path, args.streammapnet_path, args.gt_path, args.save_dir, PATCH_SIZE_LARGE if args.large else PATCH_SIZE,
        is_nusc=args.is_nusc, save_format=args.format)