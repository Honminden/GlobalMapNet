import copy
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
from .misc import unified_timestamp, init_map_element_from


def simple_replace(global_map, local_map, patch_box, threshold=0.05):
    if isinstance(threshold, list):
        ignorance_threshold = threshold[0]
        buffer_threshold = threshold[1]
        distance_threshold = threshold[2]
    else:
        ignorance_threshold = threshold
        buffer_threshold = threshold
        distance_threshold = threshold

    global_map_replaced = global_map[:]
    global_map_intersect, global_map_kept = _find_intersection(global_map_replaced, patch_box, keep_in_or_out='out', ignorance_threshold=ignorance_threshold)
    global_map_replaced = [map_element for map_element in global_map_replaced if map_element not in global_map_intersect]
    _, local_map_kept = _find_intersection(local_map, patch_box, keep_in_or_out='in', ignorance_threshold=ignorance_threshold)

    patch_buff = patch_box.exterior.buffer(buffer_threshold)
    map_elements_contained, map_elements_no_contained, indices_set = _filter_elements_by_head_tail(global_map_kept + local_map_kept, patch_buff)

    map_elements_connected = _match_and_connect(map_elements_contained, indices_set, distance_threshold=distance_threshold)

    for category in map_elements_no_contained.keys():
        global_map_replaced += map_elements_no_contained[category]
    for category in map_elements_connected.keys():
        global_map_replaced += map_elements_connected[category]
    
    return global_map_replaced


def _find_intersection(map, patch_box, keep_in_or_out='out', keep_index=False, ignorance_threshold=0.05):
    def _add_new_element(map_elements_kept, map_element, index, line):
        new_map_element = init_map_element_from(map_element, coords=list(line.coords))
        if keep_index:
            # keep the index of the original map element
            # if the intersection part contains multiple lines, their indices point to the same map element
            map_elements_kept.append((new_map_element, index))
        else:
            map_elements_kept.append(new_map_element)

    assert keep_in_or_out in ['in', 'out'], 'keep_in_or_out should be either "in" or "out".'
    map_elements_intersect = list()
    map_elements_kept = list()
    for map_element in map:
        line = LineString(map_element['coords'])
        if line.intersects(patch_box):
            map_elements_intersect.append(map_element)
            index = len(map_elements_intersect) - 1
            if keep_in_or_out == 'in':
                # keep the part inside patch_box
                line = line.intersection(patch_box)
            else:
                # keep the part outside patch_box
                line = line.difference(patch_box)
            if not line.is_empty:
                if 'Multi' in line.geom_type or 'Collection' in line.geom_type:
                    geoms = [geom for geom in line.geoms if geom.geom_type != 'Point'] # remove points
                    line = linemerge(geoms)
                if not line.is_empty and line.geom_type == 'MultiLineString':
                    if line.geom_type == 'MultiLineString':
                        for single_line in line.geoms:
                            if single_line.length > ignorance_threshold:
                                _add_new_element(map_elements_kept, map_element, index, single_line)
                    else:
                        _add_new_element(map_elements_kept, map_element, index, line)
                else:
                    _add_new_element(map_elements_kept, map_element, index, line)
    
    return map_elements_intersect, map_elements_kept


def _filter_elements_by_head_tail(map_elements, patch_buff):
    """
    Filters map elements based on whether their head or tail point is contained within a given patch buffer.

    Args:
        map_elements (dict): A dictionary of map elements grouped by category.
        patch_buff (shapely.geometry.Polygon): A polygon representing the patch buffer.

    Returns:
        map_elements_contained contains the filtered map elements, grouped by category.
        map_elements_no_contained contains the unfiltered map elements, grouped by category.
        indices_set contains filtered indices mapped by points, grouped by category.
        When the head point is contained in the patch buffer, the index is non-negative. When the tail point is contained, the index is -1 - origin_index.
    """
    map_elements_contained = dict()
    map_elements_no_contained = dict()
    indices_set = dict()
    for map_element in map_elements:
        head, tail = Point(map_element['coords'][0]), Point(map_element['coords'][-1])
        head_selected, tail_selected = patch_buff.contains(head), patch_buff.contains(tail)
        if head_selected or tail_selected:
            if map_element['category'] not in map_elements_contained:
                map_elements_contained[map_element['category']] = list()
            map_elements_contained[map_element['category']].append(map_element)
            index = len(map_elements_contained[map_element['category']]) - 1
            if map_element['category'] not in indices_set:
                indices_set[map_element['category']] = list()
            if head_selected:
                indices_set[map_element['category']].append(index)
            if tail_selected:
                indices_set[map_element['category']].append(-1 - index)
        else:
            if map_element['category'] not in map_elements_no_contained:
                map_elements_no_contained[map_element['category']] = list()
            map_elements_no_contained[map_element['category']].append(map_element)
        
    return map_elements_contained, map_elements_no_contained, indices_set


def _match_and_connect(map_elements, indices_set, distance_threshold=0.05):
    """
    Matches and connects map elements based on their nearness of head or tail points and returns the connected map elements.

    Args:
    - map_elements (dict): A dictionary containing map elements categorized by their type.
    - indices_set (dict): A dictionary containing the indices of head or tail points of map elements categorized by their type.
    - buffer_threshold (float): The maximum distance between two points for them to be considered as a match.

    Returns:
    - map_elements_connected (dict): A dictionary containing the connected map elements categorized by their type.
    """
    map_elements_connected = dict()
    for category in map_elements.keys():
        indices = copy.deepcopy(indices_set[category])
        matched_pairs = list()
        while len(indices) > 0:
            index1 = indices[0]
            indices.remove(index1)
            if index1 >= 0:
                target_index1 = index1
                point1_coords = map_elements[category][target_index1]['coords'][0]
            else:
                target_index1 = -1 - index1
                point1_coords = map_elements[category][target_index1]['coords'][-1]
            point1 = Point(point1_coords)

            for index2 in indices:
                if index2 >= 0:
                    target_index2 = index2
                    point2_coords = map_elements[category][target_index2]['coords'][0]
                else:
                    target_index2 = -1 - index2
                    point2_coords = map_elements[category][target_index2]['coords'][-1]
                point2 = Point(point2_coords)
                
                if target_index1 != target_index2 and point1.distance(point2) <= distance_threshold:
                    new_point_coords = ((point1_coords[0] + point2_coords[0]) / 2.0,
                                (point1_coords[1] + point2_coords[1]) / 2.0)
                    matched_pairs.append((index1, index2, new_point_coords))
                    indices.remove(index2)
                    break
        
        history_map_elements = copy.deepcopy(map_elements[category])
        active_map_elements = history_map_elements[:]
        while len(matched_pairs) > 0:
            index1, index2, new_point_coords = matched_pairs.pop(0)
            # connect: [coords1] -> new_point_coords -> [coords2]
            if index1 >= 0:
                target_index1 = index1
                # reverse and remove the last point
                map_element1 = history_map_elements[target_index1]
                coords1 = map_element1['coords'][::-1][:-1]
            else:
                # remove the last point
                target_index1 = -1 - index1
                map_element1 = history_map_elements[target_index1]
                coords1 = map_element1['coords'][:-1]

            if index2 >= 0:
                target_index2 = index2
                # remove the first point
                map_element2 = history_map_elements[target_index2]
                coords2 = map_element2['coords'][1:]
            else:
                # reverse and remove the first point
                target_index2 = -1 - index2
                map_element2 = history_map_elements[target_index2]
                coords2 = map_element2['coords'][::-1][1:]

            new_coords = coords1 + [new_point_coords] + coords2
            if 'details' in map_element1.keys() and 'details' in map_element2.keys():
                details = dict()
                if 'timestamp' in map_element1['details'].keys() and 'timestamp' in map_element2['details'].keys():
                    if map_element1['details']['timestamp'] > map_element2['details']['timestamp']:
                        details = copy.deepcopy(map_element1['details'])
                    else:
                        details = copy.deepcopy(map_element2['details'])
                if 'score' in map_element1['details'].keys() and 'score' in map_element2['details'].keys():
                    score1, score2 = map_element1['details']['score'], map_element2['details']['score']
                    if len(coords1) > 1:
                        score1_weight = LineString(coords1).length
                    else:
                        score1_weight = 1e-9
                    if len(coords2) > 1:
                        score2_weight = LineString(coords2).length
                    else:
                        score2_weight = 1e-9
                    new_score = (score1 * score1_weight + score2 * score2_weight) / (score1_weight + score2_weight)
                    details['score'] = new_score
                if 'timestamp' in map_element1['details'].keys() and 'timestamp' in map_element2['details'].keys():
                    timestamp = unified_timestamp()
                    details['timestamp'] = timestamp
                new_map_element = {'category': category, 'coords': new_coords, 'details': details}
            else:
                new_map_element = {'category': category, 'coords': new_coords}

            history_map_elements.append(new_map_element)
            active_map_elements.append(new_map_element)
            active_map_elements.remove(history_map_elements[target_index1])
            active_map_elements.remove(history_map_elements[target_index2])

            # update indices in rest pairs
            new_target_index = len(history_map_elements) - 1
            remove_list = list()
            for idx, (rest_index1, rest_index2, rest_new_point_coords) in enumerate(matched_pairs):
                if (rest_index1 == -1 - index1 and rest_index2 == -1 - index2) or (rest_index1 == -1 - index2 and rest_index2 == -1 - index1):
                    # the element is closed
                    remove_list.append(idx)
                elif rest_index1 == -1 - index1:
                    # become new head
                    matched_pairs[idx] = (new_target_index, rest_index2, rest_new_point_coords)
                elif rest_index2 == -1 - index1:
                    # become new head
                    matched_pairs[idx] = (rest_index1, new_target_index, rest_new_point_coords)
                elif rest_index1 == -1 - index2:
                    # become new tail
                    matched_pairs[idx] = (-1 - new_target_index, rest_index2, rest_new_point_coords)
                elif rest_index2 == -1 - index2:
                    # become new tail
                    matched_pairs[idx] = (rest_index1, -1 - new_target_index, rest_new_point_coords)
            
            for idx in remove_list[::-1]:
                matched_pairs.remove(idx)
        
        map_elements_connected[category] = active_map_elements
    return map_elements_connected