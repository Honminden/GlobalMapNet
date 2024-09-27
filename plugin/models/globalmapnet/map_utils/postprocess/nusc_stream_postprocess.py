# adapted from https://github.com/yuantianyuan01/StreamMapNet/blob/main/plugin/datasets/map_utils/nuscmap_extractor.py

from shapely.geometry import Polygon
from shapely import ops, strtree

import numpy as np
from .utils import split_collections, get_drivable_area_contour, \
        get_ped_crossing_contour
from typing import Dict, List, Union


def _union_ped(ped_geoms: List[Polygon]) -> List[Polygon]:
    ''' merge close ped crossings.
    
    Args:
        ped_geoms (list): list of Polygon
    
    Returns:
        union_ped_geoms (Dict): merged ped crossings 
    '''

    def get_rec_direction(geom):
        rect = geom.minimum_rotated_rectangle
        rect_v_p = np.array(rect.exterior.coords)[:3]
        rect_v = rect_v_p[1:]-rect_v_p[:-1]
        v_len = np.linalg.norm(rect_v, axis=-1)
        longest_v_i = v_len.argmax()

        return rect_v[longest_v_i], v_len[longest_v_i]

    tree = strtree.STRtree(ped_geoms)
    index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

    final_pgeom = []
    remain_idx = [i for i in range(len(ped_geoms))]
    for i, pgeom in enumerate(ped_geoms):

        if i not in remain_idx:
            continue
        # update
        remain_idx.pop(remain_idx.index(i))
        pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
        final_pgeom.append(pgeom)

        for o in tree.query(pgeom):
            o_idx = index_by_id[id(o)]
            if o_idx not in remain_idx:
                continue

            o_v, o_v_norm = get_rec_direction(o)
            cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)
            if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees.
                final_pgeom[-1] =\
                    final_pgeom[-1].union(o)
                # update
                remain_idx.pop(remain_idx.index(o_idx))

    results = []
    for p in final_pgeom:
        results.extend(split_collections(p))
    return results

def road_postprocess(road_segments_geoms, lane_geoms):
    union_roads = ops.unary_union(road_segments_geoms)
    union_lanes = ops.unary_union(lane_geoms)
    drivable_areas = ops.unary_union([union_roads, union_lanes])
        
    drivable_areas = split_collections(drivable_areas)
        
    # boundaries are defined as the contour of drivable areas
    boundaries = get_drivable_area_contour(drivable_areas)
    return boundaries

def lane_postprocess(lane_geoms):
    all_dividers = []
    for line in lane_geoms:
        all_dividers += split_collections(line)
    return all_dividers

def ped_postprocess(ped_geoms):
    ped_crossings = []
    for p in ped_geoms:
        ped_crossings += split_collections(p)
    # some ped crossings are split into several small parts
    # we need to merge them
    ped_crossings = _union_ped(ped_crossings)
    
    ped_crossing_lines = []
    for p in ped_crossings:
        # extract exteriors to get a closed polyline
        line = get_ped_crossing_contour(p)
        if line is not None:
                ped_crossing_lines.append(line)
    
    return ped_crossing_lines