import copy
from shapely.geometry import LineString, Point
from .simple_replace_mode import _find_intersection
from .matching import chamfer_distance_match
from .misc import unified_timestamp


def match_replace(global_map, local_map, patch_box, threshold=[0.05, 0.5, 0.5], sample_num=100, simplify=True):
    if isinstance(threshold, list):
        ignorance_threshold = threshold[0]
        cd_threshold = threshold[1]
        simplify_threshold = threshold[2]
    else:
        ignorance_threshold = threshold
        cd_threshold = threshold
        simplify_threshold = threshold

    if not simplify:
        simplify_threshold = None

    global_map_replaced = global_map[:]
    global_map_intersect, global_map_kept = _find_intersection(global_map_replaced, patch_box, keep_in_or_out='in', keep_index=True, ignorance_threshold=ignorance_threshold)
    global_map_replaced = [map_element for map_element in global_map_replaced if map_element not in global_map_intersect]
    _, local_map_kept = _find_intersection(local_map, patch_box, keep_in_or_out='in', ignorance_threshold=ignorance_threshold)

    global_map_by_category, local_map_by_category = _collect_by_category(global_map_kept, local_map_kept)
    matched_pairs_by_category = _match_pairs(global_map_by_category, local_map_by_category, cd_threshold=cd_threshold, sample_num=sample_num)
    global_map_updated, local_map_rest_by_category = _part_update(global_map_intersect, global_map_by_category, local_map_by_category, matched_pairs_by_category, 
                                                                  simplify_threshold=simplify_threshold)

    for global_map_element in global_map_updated:
        global_map_replaced.append(global_map_element)
    for category in local_map_rest_by_category.keys():
        global_map_replaced += local_map_rest_by_category[category]
    
    return global_map_replaced


def _collect_by_category(global_map_kept, local_map_kept):
    global_map_by_category = dict()
    for map_element, index in global_map_kept:
        if map_element['category'] not in global_map_by_category:
            global_map_by_category[map_element['category']] = {'map_elements': list(), 'indices': list()}
        global_map_by_category[map_element['category']]['map_elements'].append(map_element)
        global_map_by_category[map_element['category']]['indices'].append(index)

    local_map_by_category = dict()
    for map_element in local_map_kept:
        if map_element['category'] not in local_map_by_category:
            local_map_by_category[map_element['category']] = list()
        local_map_by_category[map_element['category']].append(map_element)
    
    return global_map_by_category, local_map_by_category


def _match_pairs(global_map_by_category, local_map_by_category, cd_threshold=0.5, sample_num=100):
    categories = set(global_map_by_category.keys()) | set(local_map_by_category.keys())
    matched_pairs_by_category = dict()
    for category in categories:
        if category not in global_map_by_category.keys():
            global_map_elements = list()
        else:
            global_map_elements = [map_element['coords'] for map_element in global_map_by_category[category]['map_elements']]
        
        if category not in local_map_by_category.keys():
            local_map_elements = list()
        else:
            local_map_elements = [map_element['coords'] for map_element in local_map_by_category[category]]

        cd_threshold_i = cd_threshold
        if isinstance(cd_threshold, dict):
            cd_threshold_i = cd_threshold[category]

        matched_pairs = chamfer_distance_match(src_lines=global_map_elements, dst_lines=local_map_elements, cd_threshold=cd_threshold_i, sample_num=sample_num, one_way=False)
        matched_pairs_by_category[category] = matched_pairs
    
    return matched_pairs_by_category


def _part_update(global_map_intersect, global_map_by_category, local_map_by_category, matched_pairs_by_category, 
                 simplify_threshold=0.5):
    global_map_updated = copy.deepcopy(global_map_intersect)
    local_map_rest_by_category = dict()
    for category in local_map_by_category.keys():
        local_map_rest_by_category[category] = local_map_by_category[category][:] # copy

    for category in matched_pairs_by_category.keys():
        for matched_pair in matched_pairs_by_category[category]:
            # find original global element
            global_map_element_index = global_map_by_category[category]['indices'][matched_pair[0]]
            global_map_element = global_map_updated[global_map_element_index]
            local_map_element = local_map_by_category[category][matched_pair[1]]
            global_line = LineString(global_map_element['coords'])
            local_coords = local_map_element['coords'][:]
            
            # perform Douglas-Peucker algorithm before update
            if simplify_threshold != None:
                local_line = LineString(local_coords)
                local_line = local_line.simplify(simplify_threshold, preserve_topology=False)
                local_coords = list(local_line.coords)

                global_line = global_line.simplify(simplify_threshold, preserve_topology=False)
                global_coords = list(global_line.coords)

            # project head point and tail point local element onto global line
            head_projection_dist = global_line.project(Point(local_coords[0]))
            tail_projection_dist = global_line.project(Point(local_coords[-1]))

            if head_projection_dist > tail_projection_dist:
                # if different direction with global element, reverse coords of local element
                local_coords = local_coords[::-1]

            # find min and max projection dist of local element points
            min_projection_dist = head_projection_dist
            max_projection_dist = tail_projection_dist
            for local_point_coords in local_coords:
                local_point_projection_dist = global_line.project(Point(local_point_coords))
                if local_point_projection_dist < min_projection_dist:
                    min_projection_dist = local_point_projection_dist
                if local_point_projection_dist > max_projection_dist:
                    max_projection_dist = local_point_projection_dist

            # if global element is closed, the tail point is left to append after for loop
            global_element_closed = global_line.project(Point(global_map_element['coords'][-1])) == 0.0
            global_coords_to_append = global_coords
            if global_element_closed:
                global_coords_to_append = global_coords[:-1]
            
            new_coords = list()
            inserted = False
            for global_point_coords in global_coords_to_append:
                point_projection_dist = global_line.project(Point(global_point_coords))
                if point_projection_dist < min_projection_dist or point_projection_dist > max_projection_dist:
                    # part not to update, keep the original global element point
                    new_coords.append(global_point_coords)
                elif not inserted:
                    # insert entire local element
                    new_coords += local_coords
                    inserted = True
                # else skip
            
            if global_element_closed:
                new_coords.append(global_map_element['coords'][-1])
            
            # perform Douglas-Peucker algorithm after update
            if simplify_threshold != None:
                new_line = LineString(new_coords)
                new_line = new_line.simplify(simplify_threshold, preserve_topology=False)
                new_coords = list(new_line.coords)

            # update global map element and remove appended local map element
            
            if 'details' in global_map_element.keys() and 'details' in local_map_element.keys():
                details = copy.deepcopy(local_map_element['details'])
                if 'score' in global_map_element['details'].keys() and 'score' in local_map_element['details'].keys():
                    score_global, score_local = global_map_element['details']['score'], local_map_element['details']['score']
                    if len(local_coords) > 1:
                        score_local_weight = max(LineString(local_coords).length, 1e-9)
                    else:
                        score_local_weight = 1e-9
                    if len(new_coords) > 1:
                        score_sum_weight = max(LineString(new_coords).length, score_local_weight)
                    else:
                        score_sum_weight = max(score_local_weight, 1e-9)
                    new_score = (score_global * (score_sum_weight - score_local_weight) + score_local * score_local_weight) / score_sum_weight
                    details['score'] = new_score
                if 'timestamp' in global_map_element['details'].keys() and 'timestamp' in local_map_element['details'].keys():
                    timestamp = unified_timestamp()
                    details['timestamp'] = timestamp
                global_map_element['details'] = details

            global_map_element['coords'] = new_coords
            local_map_rest_by_category[category].remove(local_map_element)
    
    return global_map_updated, local_map_rest_by_category