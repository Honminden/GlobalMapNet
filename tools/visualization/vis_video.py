import argparse
import cv2
import mmcv
import json
import os
import os.path as osp
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from PIL import Image
from shapely import ops, affinity
from shapely.geometry import box, LineString, MultiLineString, Polygon, MultiPolygon, Point, CAP_STYLE, JOIN_STYLE
from pyquaternion import Quaternion
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from IPython import embed
from plugin.datasets.builder import build_dataloader
from plugin.models.globalmapnet.map_utils.functional.ego import generate_patch_box
from tools.visualization.vis_compare import print_map


DATASET_NAMES = {'nusc': 'nuScenes', 'av2': 'Argoverse2'}

CAMS_NUSC = [
    ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
    ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
]

CAMS_AV2 = [
    ['ring_front_left', 'ring_front_center', 'ring_front_right'],
    ['ring_side_left', None, 'ring_side_right'],
    ['ring_rear_left', None, 'ring_rear_right'],
]

COLOR_MAPS_PLT = {
    'divider': 'r',
    'boundary': 'g',
    'ped_crossing': 'b',
    'centerline': 'orange',
    'drivable_area': 'y',
}

PATCH_SIZE = (60, 30)
PATCH_SIZE_LARGE = (100, 50)

def import_plugin(cfg):
    # import modules from plguin/xx, registry will be updated
    import sys
    sys.path.append(os.path.abspath('.'))  
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                def import_path(plugin_dir):
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)

                plugin_dirs = cfg.plugin_dir
                if not isinstance(plugin_dirs,list):
                    plugin_dirs = [plugin_dirs,]
                for plugin_dir in plugin_dirs:
                    import_path(plugin_dir)
                
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

def render_img(ax, img_fpath):
    img = Image.open(img_fpath)
    ax.imshow(img)
    ax.axis('off')

def render_local_map(ax, vectors, id2cat, patch_size, car_img=None, linewidth=1.5):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if car_img is not None:
            imagebox = OffsetImage(car_img, zoom = 0.5)
            ab = AnnotationBbox(imagebox, (0.5, 0.5), frameon = False)
            ax.add_artist(ab)

        for label, vector_list in vectors.items():
            cat = id2cat[label]
            color = COLOR_MAPS_PLT[cat]
            for vector in vector_list:
                if isinstance(vector, list):
                    vector = np.array(vector)
                    vector = np.array(LineString(vector).simplify(0.2).coords)
                pts = vector[:, :2]
                x = np.array([pt[0] for pt in pts])
                y = np.array([pt[1] for pt in pts])
                ax.plot(x, y, '-', color=color, linewidth=linewidth)

def render_global_map(ax, global_map, score_thr, patch_size, car_img=None, linewidth=1.5, render_patch_box=True):
    map_elements, poses = global_map['map_elements'], global_map['poses']
    ax.grid(False)
    ax.axis('equal')
    ax.axis('off')

    if poses != None:
        patch_boxes = [generate_patch_box(patch_size, pose) for pose in poses]
        patch_trace = ops.unary_union(patch_boxes)

    for map_element in map_elements:
        if 'details' in map_element and map_element['details']['score'] < score_thr:
            continue
        coords = map_element['coords']
        
        if map_element['category'] == 'road':
            ax.plot([coord[0] for coord in coords], [coord[1] for coord in coords], 'g-', linewidth=linewidth)
        elif map_element['category'] == 'lane':
            ax.plot([coord[0] for coord in coords], [coord[1] for coord in coords], 'r-', linewidth=linewidth)
        elif map_element['category'] == 'ped':
            ax.plot([coord[0] for coord in coords], [coord[1] for coord in coords], 'b-', linewidth=linewidth)

    if patch_trace.geom_type == 'MultiPolygon':
        for poly in patch_trace:
            ax.plot(poly.exterior.xy[0], poly.exterior.xy[1], '-', color='gray', linewidth=linewidth)
    else:
        ax.plot(patch_trace.exterior.xy[0], patch_trace.exterior.xy[1], '-', color='gray', linewidth=linewidth)

    if render_patch_box:
        patch_box = generate_patch_box(patch_size, poses[-1])
        ax.plot(patch_box.exterior.xy[0], patch_box.exterior.xy[1], '--', color='gray', linewidth=linewidth / 2, alpha=1)

    if car_img is not None:
        minx, miny, maxx, maxy = patch_trace.bounds
        scale = max((maxx - minx) / (patch_size[0] + 1e-9), (maxy - miny) / (patch_size[0] + 1e-9))
        # rotate the car image
        angle = Quaternion(poses[-1][3:]).yaw_pitch_roll[0]
        car_img = car_img.rotate(angle * 180 / np.pi)
        width, height = car_img.size
        car_img = car_img.resize((int(width / scale), int(height / scale)))
        imagebox = OffsetImage(car_img, zoom = 0.5)
        ab = AnnotationBbox(imagebox, (poses[-1][0], poses[-1][1]), frameon = False)
        ax.add_artist(ab)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                    description='visualize vectorized global map to video')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file')
    parser.add_argument('idx', type=int,
        help='which scene to visualize')
    parser.add_argument(
        '--save-dir',
        default='./cache/vis',
        help='save dir of visualization')
    parser.add_argument(
        '--is-nusc',
        action='store_true',
        help='whether the dataset is nuscenes')
    parser.add_argument(
        '--large',
        action='store_true',
        help='use large patch size (100x50)')


    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    dataset = build_dataset(cfg.data.test)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda()
    model.eval()

    # prepare data metas
    scenes = dict()
    for sample in dataset.samples:
        scene_name = sample['scene_name']
        if scene_name not in scenes:
            scenes[scene_name] = list()
        scenes[scene_name].append(sample)

    car_img = Image.open('resources/car.png')

    cat2id = cfg.cat2id
    id2cat = {v: k for k, v in cat2id.items()}

    is_nusc = args.is_nusc
    dataset_name = DATASET_NAMES['nusc'] if is_nusc else DATASET_NAMES['av2']
    scene_name = list(scenes.keys())[args.idx]
    print(len(list(scenes.keys())))
    samples = scenes[scene_name]

    # start video output stream
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    real_frame_per_sec = 4 # 2x speed
    frame_repeat_time = 5
    frame_per_sec = real_frame_per_sec * frame_repeat_time # 20 fps
    out = cv2.VideoWriter(os.path.join(args.save_dir, rf'{scene_name}.mp4'), fourcc, frame_per_sec, (1920, 1080))

    model.init_global_args()
    if not is_nusc:
        init_time = None
    for sample in samples:
        with torch.no_grad():
            idx = sample['sample_idx']
            if is_nusc:
                input_dict = dataset.get_sample(idx)
            else:
                input_dict = dataset.get_sample(idx // 5)
            data = dataset.pipeline(input_dict)
            result = model(return_loss=False, rescale=True, img_metas=[data['img_metas'].data], img=data['img'].data.unsqueeze(0).cuda())[0]

            score_thr = 0.4
            vectors = {label: [] for label in cat2id.values()}
            for i in range(len(result['labels'])):
                score = result['scores'][i]
                label = result['labels'][i]
                v = result['vectors'][i]

                if score > score_thr:
                    vectors[label].append(v)

            fig = plt.figure(dpi=100)
            fig.set_size_inches(19.20, 10.80)
            scene_name_title = scene_name if is_nusc else f'scene-{scene_name[:6]}'
            fig.suptitle(f'Single-scene Evaluation on {dataset_name} ({scene_name_title})', fontsize=32)
            if is_nusc:
                ts = datetime.fromtimestamp(sample['timestamp'] / 1e6).strftime('%Y-%m-%d %H:%M:%S.%f')
            else:
                if init_time is None:
                    init_time = int(sample['token']) / 1e9
                delta_time = int(sample['token']) / 1e9 - init_time
                ts = f'+{delta_time:.2f}s'
            plt.figtext(0.5, 0.92, f'time: {ts}', fontsize=16, ha='center')
            plt.figtext(0.5, 0.01, 'GlobalMapNet: An Online Framework for Vectorized Global HD Map Construction, Shi et al.', fontsize=16, ha='center')

            gs = gridspec.GridSpec(12, 6, figure=fig, wspace=0, hspace=0)
            # cams
            if is_nusc:
                for i in range(2):
                    for j in range(3):
                        ax = fig.add_subplot(gs[i*3:(i+1)*3, j])
                        img_fpath = sample['cams'][CAMS_NUSC[i][j]]['img_fpath']
                        render_img(ax, img_fpath)
                        ax.set_title(CAMS[i][j])
            else:
                for i in range(3):
                    for j in range(3):
                        cam_name = CAMS_AV2[i][j]
                        if cam_name == None:
                            continue
                        elif cam_name == 'ring_front_center':
                            ax = fig.add_subplot(gs[:6, j])
                        else:
                            ax = fig.add_subplot(gs[i*2:(i+1)*2, j])
                        img_fpath = sample['cams'][cam_name]['img_fpath']
                        render_img(ax, img_fpath)
                        ax.set_title(cam_name, ha='center', y = 0.01, color='white')

            # local map
            ax = fig.add_subplot(gs[6:, :3])
            render_local_map(ax, vectors, id2cat, PATCH_SIZE, car_img)
            ax.set_title('Local Map', fontsize=16)

            # global map
            ax = fig.add_subplot(gs[:, 3:])
            global_map = model.map_builder.global_maps[f'{cfg.global_map_config.map_name}_{scene_name}']
            render_global_map(ax, global_map, score_thr, PATCH_SIZE, car_img)
            ax.set_title('Global Map', fontsize=24)

            fig.tight_layout()
            fig.canvas.draw()
            canvas = np.array(fig.canvas.renderer._renderer)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGR)
            for _ in range(frame_repeat_time):
                out.write(canvas)

    out.release()
    print(f"Video saved to {os.path.join(args.save_dir, rf'{scene_name}.mp4')}.")