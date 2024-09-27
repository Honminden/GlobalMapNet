import json
import os
from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path
from shapely.geometry import LineString, box, Polygon, LinearRing
from shapely import ops
import numpy as np
from scipy.spatial import distance
import mmcv
import matplotlib.pyplot as plt


def get_ped_crossing_contour(polygon):
    ''' Extract ped crossing contours to get a closed polyline.
    Different from `get_drivable_area_contour`, this function ensures a closed polyline.

    Args:
        polygon (Polygon): ped crossing polygon to be extracted.
    
    Returns:
        line (LineString): a closed line
    '''

    ext = polygon.exterior
    if not ext.is_ccw:
        ext = LinearRing(list(ext.coords)[::-1])
    lines = LineString(ext)
    if lines.type != 'LineString':
        # remove points in intersection results
        lines = [l for l in lines.geoms if l.geom_type != 'Point']
        lines = ops.linemerge(lines)
        
        # same instance but not connected.
        if lines.type != 'LineString':
            ls = []
            for l in lines.geoms:
                ls.append(np.array(l.coords))
            
            lines = np.concatenate(ls, axis=0)
            lines = LineString(lines)

        start = list(lines.coords[0])
        end = list(lines.coords[-1])
        if not np.allclose(start, end, atol=1e-3):
            new_line = list(lines.coords)
            new_line.append(start)
            lines = LineString(new_line) # make ped cross closed

    if not lines.is_empty:
        return lines
    
    return None


def get_drivable_area_contour(drivable_areas):
    ''' Extract drivable area contours to get list of boundaries.

    Args:
        drivable_areas (list): list of drivable areas.
    
    Returns:
        boundaries (List): list of boundaries.
    '''
    exteriors = []
    interiors = []
    
    for poly in drivable_areas:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)
    
    results = []
    for ext in exteriors:
        # NOTE: we make sure all exteriors are clock-wise
        # such that each boundary's right-hand-side is drivable area
        # and left-hand-side is walk way
        
        if ext.is_ccw:
            ext = LinearRing(list(ext.coords)[::-1])
        lines = LineString(ext)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        assert lines.geom_type in ['MultiLineString', 'LineString']
        
        results.extend(split_collections(lines))

    for inter in interiors:
        # NOTE: we make sure all interiors are counter-clock-wise
        if not inter.is_ccw:
            inter = LinearRing(list(inter.coords)[::-1])
        lines = LineString(inter)
        if lines.geom_type == 'MultiLineString':
            lines = ops.linemerge(lines)
        assert lines.geom_type in ['MultiLineString', 'LineString']
        
        results.extend(split_collections(lines))

    return results


def split_collections(geom):
    ''' Split Multi-geoms to list and check is valid or is empty.
        
    Args:
        geom (BaseGeometry): geoms to be split or validate.
    
    Returns:
        geometries (List): list of geometries.
    '''
    assert geom.geom_type in ['MultiLineString', 'LineString', 'MultiPolygon', 
        'Polygon', 'GeometryCollection'], f"got geom type {geom.geom_type}"
    if 'Multi' in geom.geom_type:
        outs = []
        for g in geom.geoms:
            if g.is_valid and not g.is_empty:
                outs.append(g)
        return outs
    else:
        if geom.is_valid and not geom.is_empty:
            return [geom,]
        else:
            return []


def remove_repeated_lines(lines):
    ''' Remove repeated dividers since each divider in argoverse2 is mentioned twice
    by both left lane and right lane.

    Args:
        lines (List): list of dividers

    Returns:
        lines (List): list of left dividers
    '''

    new_lines = []
    for line in lines:
        repeated = False
        for l in new_lines:
            length = min(line.length, l.length)
            
            # hand-crafted rule to check overlap
            if line.buffer(0.3).intersection(l.buffer(0.3)).area \
                    > 0.2 * length:
                repeated = True
                break
        
        if not repeated:
            new_lines.append(line)
    
    return new_lines


def remove_boundary_dividers(dividers, boundaries):
    ''' Some dividers overlaps with boundaries in argoverse2 dataset so
    we need to remove these dividers.

    Args:
        dividers (list): list of dividers
        boundaries (list): list of boundaries

    Returns:
        left_dividers (list): list of left dividers
    '''

    for idx in range(len(dividers))[::-1]:
        divider = dividers[idx]
        
        for bound in boundaries:
            length = min(divider.length, bound.length)

            # hand-crafted rule to check overlap
            if divider.buffer(0.3).intersection(bound.buffer(0.3)).area \
                    > 0.2 * length:
                # the divider overlaps boundary
                dividers.pop(idx)
                break

    return dividers


def connect_lines(lines):
    ''' Some dividers are split into multiple small parts
    so we need to connect these lines.

    Args:
        dividers (list): list of dividers
        boundaries (list): list of boundaries

    Returns:
        left_dividers (list): list of left dividers
    '''

    new_lines = []
    eps = 0.1 # threshold to identify continuous lines
    while len(lines) > 1:
        line1 = lines[0]
        merged_flag = False
        for i, line2 in enumerate(lines[1:]):
            # hand-crafted rule
            begin1 = list(line1.coords)[0]
            end1 = list(line1.coords)[-1]
            begin2 = list(line2.coords)[0]
            end2 = list(line2.coords)[-1]

            dist_matrix = distance.cdist([begin1, end1], [begin2, end2])
            if dist_matrix[0, 0] < eps:
                coords = list(line2.coords)[::-1] + list(line1.coords)
            elif dist_matrix[0, 1] < eps:
                coords = list(line2.coords) + list(line1.coords)
            elif dist_matrix[1, 0] < eps:
                coords = list(line1.coords) + list(line2.coords)
            elif dist_matrix[1, 1] < eps:
                coords = list(line1.coords) + list(line2.coords)[::-1]
            else: continue

            new_line = LineString(coords)
            lines.pop(i + 1)
            lines[0] = new_line
            merged_flag = True
            break
        
        if merged_flag: continue

        new_lines.append(line1)
        lines.pop(0)

    if len(lines) == 1:
        new_lines.append(lines[0])

    return new_lines


def gen_argo_gt_map(ann_root, polygon_ped=True):
    ann = mmcv.load(ann_root)
    id2map = dict()
    for log_id, path in ann['id2map'].items():
        id2map[log_id] = ArgoverseStaticMap.from_json(Path(path))

    global_maps = dict()
    for log_id, avm in id2map.items():
        all_dividers = []
        # for every lane segment, extract its right/left boundaries as road dividers
        for _, ls in avm.vector_lane_segments.items():
            # right divider
            right_xyz = ls.right_lane_boundary.xyz
            right_mark_type = ls.right_mark_type

            right_line = LineString(right_xyz)

            if not right_line.is_empty and not right_mark_type in ['NONE', 'UNKNOWN']:
                all_dividers += split_collections(right_line)
                
            # left divider
            left_xyz = ls.left_lane_boundary.xyz
            left_mark_type = ls.left_mark_type

            left_line = LineString(left_xyz)

            if not left_line.is_empty and not left_mark_type in ['NONE', 'UNKNOWN']:
                all_dividers += split_collections(left_line)

        # remove repeated dividers since each divider in argoverse2 is mentioned twice
        # by both left lane and right lane
        all_dividers = remove_repeated_lines(all_dividers)

        ped_crossings = [] 
        for _, pc in avm.vector_pedestrian_crossings.items():
            edge1_xyz = pc.edge1.xyz
            edge2_xyz = pc.edge2.xyz

            # if True, organize each ped crossing as closed polylines. 
            if polygon_ped:
                vertices = np.concatenate([edge1_xyz, edge2_xyz[::-1, :]])
                p = Polygon(vertices)
                line = get_ped_crossing_contour(p)
                if line is not None:
                    ped_crossings.append(line)

            # Otherwise organize each ped crossing as two parallel polylines.
            else:
                line1 = LineString(edge1_xyz)
                line2 = LineString(edge2_xyz)

                # take the whole ped cross if all two edges are in roi range
                if not line1.is_empty and not line2.is_empty:
                    ped_crossings.append(line1)
                    ped_crossings.append(line2)

        drivable_areas = []
        for _, da in avm.vector_drivable_areas.items():
            polygon_xyz = da.xyz
            polygon = Polygon(polygon_xyz)

            drivable_areas.append(polygon)

        # union all drivable areas polygon
        drivable_areas = ops.unary_union(drivable_areas)
        drivable_areas = split_collections(drivable_areas)

        # boundaries are defined as the contour of drivable areas
        boundaries = get_drivable_area_contour(drivable_areas)

        # some dividers overlaps with boundaries in argoverse2 dataset
        # we need to remove these dividers
        all_dividers = remove_boundary_dividers(all_dividers, boundaries)

        # some dividers are split into multiple small parts
        # we connect these lines
        all_dividers = connect_lines(all_dividers)

        # out_dict = dict(
        #     divider=all_dividers, # List[LineString]
        #     ped_crossing=ped_crossings, # List[LineString]
        #     boundary=boundaries, # List[LineString]
        #     drivable_area=drivable_areas, # List[Polygon],
        # )

        global_map = dict()
        global_map['meta_info'] = {'map_name': log_id}
        global_map['poses'] = list()
        global_map['map_elements'] = list()


        for line in boundaries:
            _coords = [pts[:2] for pts in list(line.coords)]
            global_map['map_elements'].append({'category': 'road', 'coords': _coords, 'details': {'score': 1.0, 'global_id': -1}})

        for line in all_dividers:
            _coords = [pts[:2] for pts in list(line.coords)]
            global_map['map_elements'].append({'category': 'lane', 'coords': _coords, 'details': {'score': 1.0, 'global_id': -1}})

        for line in ped_crossings:
            _coords = [pts[:2] for pts in list(line.coords)]
            global_map['map_elements'].append({'category': 'ped', 'coords': _coords, 'details': {'score': 1.0, 'global_id': -1}})

        global_maps[log_id] = global_map

    return global_maps


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                    description='generate vectorized global map from argoverse2')
    parser.add_argument(
        '--ann_dir',
        default='./cache/annotations',
        help='root dir of annotations')
    parser.add_argument(
        '--root-dir',
        default='./cache/global_maps',
        help='root dir of global map storage')
    parser.add_argument('--split', type=str, help='split name (locating annotation file)', choices=[
        'train_newsplit',
        'val_newsplit',
        'train',
        'val',
        'test',
        'all'])
    
    args = parser.parse_args()

    if args.split == 'all':
        # all in one
        splits = ['train_newsplit', 'val_newsplit', 'train', 'val', 'test']
    else:
        splits = [args.split]

    global_maps_total = dict()
    for split in splits:
            ann_root = os.path.join(args.ann_dir, f'av2_map_infos_{split}.pkl')
            global_maps = gen_argo_gt_map(ann_root)
            global_maps_total = {**global_maps_total, **global_maps}
            
    with open(os.path.join(args.root_dir, f'av2_{args.split}_gt.json'), 'w') as f:
        json.dump(global_maps_total, f)
        print(f'exported gt map with splits: {splits}')