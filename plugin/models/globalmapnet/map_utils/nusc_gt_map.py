import json
import os
from shapely import ops
from shapely.geometry import LineString, MultiLineString, MultiPolygon
from shapely.validation import make_valid
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.map_expansion.map_api import NuScenesMap
from plugin.models.globalmapnet.map_utils.postprocess.nusc_stream_postprocess import road_postprocess, lane_postprocess, ped_postprocess
from plugin.models.globalmapnet.map_utils.indexing.index import index_map_elements, NUSC


def _polygon_postprocess(polygon_geoms, local_patch=None):
    exteriors = list()
    interiors = list()
    un_geoms = ops.unary_union(polygon_geoms)
    if un_geoms.geom_type != 'MultiPolygon':
        un_geoms = MultiPolygon([un_geoms])
    for poly in un_geoms.geoms:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)

    results = list()
    for ext in exteriors:
        if ext.is_ccw:
            ext.coords = list(ext.coords)[::-1]
        if local_patch != None:
            ext = ext.intersection(local_patch)
        if ext.geom_type == 'LinearRing':
            ext = LineString(ext)
            results.append(ext)
        elif ext.geom_type == 'MultiLineString':
            ext = ops.linemerge(ext)
            if ext.geom_type == 'MultiLineString':
                results += ext.geoms
            else:
                results.append(ext)
        else:
            results.append(ext)

    for inter in interiors:
        if not inter.is_ccw:
            inter.coords = list(inter.coords)[::-1]
        if local_patch != None:
            inter = inter.intersection(local_patch)
        if inter.geom_type == 'LinearRing':
            inter = LineString(inter)
            results.append(inter)
        elif inter.geom_type == 'MultiLineString':
            inter = ops.linemerge(inter)
            if inter.geom_type == 'MultiLineString':
                results += inter.geoms
            else:
                results.append(inter)
        else:
            results.append(inter)
    return results


def _polyline_postprocess(geoms, local_patch=None):
    polyline_list = list()
    for geom in geoms:
        if geom.geom_type == 'MultiPolygon':
            geom = MultiLineString([polygon.exterior for polygon in geom.geoms])
        elif geom.geom_type == 'Polygon':
            geom = geom.exterior

        if local_patch != None:
            geom = local_patch.intersection(geom)
        if geom.geom_type == 'MultiLineString':
            for single_polyline in geom.geoms:
                polyline_list.append(single_polyline)
        else:
            polyline_list.append(geom)

    return polyline_list


def gen_nusc_gt_map(dataroot, location, postprocess='naive'):
    nusc_map = NuScenesMap(dataroot=dataroot, map_name=location)

    global_layer_geoms = dict()
    for layer_name in ['road_divider', 'lane_divider', 'road_segment', 'lane', 'ped_crossing']:
        global_layer_geoms[layer_name] = list()
        records = getattr(nusc_map, layer_name)
        for record in records:
            if layer_name in ['road_divider', 'lane_divider']:
                polyline = nusc_map.extract_line(record['line_token'])
                global_layer_geoms[layer_name].append(polyline)
            elif layer_name in ['road_segment', 'lane', 'ped_crossing']:
                polygon = nusc_map.extract_polygon(record['polygon_token'])
                if not polygon.is_valid:
                    polygon = make_valid(polygon)
                    if polygon.geom_type == 'MultiPolygon':
                        for poly in polygon.geoms:
                            global_layer_geoms[layer_name].append(poly)
                        continue
                global_layer_geoms[layer_name].append(polygon)
    
    if postprocess == 'naive':
        global_road_lines = _polygon_postprocess(global_layer_geoms['road_segment'] + global_layer_geoms['lane'])
        global_lane_lines = _polyline_postprocess(global_layer_geoms['road_divider'] + global_layer_geoms['lane_divider'])
        global_ped_lines = _polygon_postprocess(global_layer_geoms['ped_crossing'])
    elif postprocess == 'stream':
        global_road_lines = road_postprocess(global_layer_geoms['road_segment'], global_layer_geoms['lane'])
        global_lane_lines = lane_postprocess(global_layer_geoms['road_divider'] + global_layer_geoms['lane_divider'])
        global_ped_lines = ped_postprocess(global_layer_geoms['ped_crossing'])

    global_map = dict()
    global_map['meta_info'] = {'map_name': location}
    global_map['poses'] = list()
    global_map['map_elements'] = list()

    road_lines_indexed, lane_lines_indexed, ped_lines_indexed = index_map_elements(global_road_lines, global_lane_lines, global_ped_lines, location, dataset=NUSC)

    for global_id, line in road_lines_indexed.items():
        global_map['map_elements'].append({'category': 'road', 'coords': list(line.coords), 'details': {'score': 1.0, 'global_id': global_id}})

    for global_id, line in lane_lines_indexed.items():
        global_map['map_elements'].append({'category': 'lane', 'coords': list(line.coords), 'details': {'score': 1.0, 'global_id': global_id}})

    for global_id, line in ped_lines_indexed.items():
        global_map['map_elements'].append({'category': 'ped', 'coords': list(line.coords), 'details': {'score': 1.0, 'global_id': global_id}})

    return global_map


def _gen_test_ego_poses(version, dataroot):
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    scenes_train = create_splits_scenes()['train']
    samples = [samp for samp in nusc.sample 
        if nusc.get('scene', samp['scene_token'])['name'] in scenes_train]
    samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

    seq_ego_poses = [nusc.get('ego_pose', nusc.get('sample_data', samples[i]['data']['LIDAR_TOP'])['ego_pose_token']) for i in range(8, 16)]
    seq_ego_poses = [seq_ego_pose['translation'] + seq_ego_pose['rotation'] for seq_ego_pose in seq_ego_poses]
    return seq_ego_poses


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                    description='generate vectorized global map from nuscenes')
    parser.add_argument(
        '--dataroot',
        default='./data/nuscenes/',
        help='root dir of dataset')
    parser.add_argument(
        '--root-dir',
        default='./cache/global_maps',
        help='root dir of global map storage')
    parser.add_argument('--location', type=str, help='location of the map', choices=[
        'boston-seaport',
        'singapore-hollandvillage',
        'singapore-onenorth',
        'singapore-queenstown',
        'all'])
    parser.add_argument('--postprocess', type=str, help='postprocessor of map elements', choices=[
        'naive',
        'stream'],
        default='stream',)
    parser.add_argument(
        '--gen-pose',
        action='store_true',
        help='whether to generate poses for test (may be very slow)')
    
    args = parser.parse_args()

    if args.location == 'all':
        locations = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
    else:
        locations = [args.location]

    for location in locations:
            global_map = gen_nusc_gt_map(dataroot=args.dataroot, location=location, postprocess=args.postprocess)

            with open(os.path.join(args.root_dir, f'{location}_gt.json'), 'w') as f:
                json.dump(global_map, f)
                print(f'exported gt map with location: {location}')

    if args.gen_pose:
        # generate poses for test, may be very slow
        seq_ego_poses = _gen_test_ego_poses(version='v1.0-trainval', dataroot=arg.dataroot)
        with open(os.path.join(args.root_dir, 'seq_ego_poses.json'), 'w') as f:
            json.dump(seq_ego_poses, f)