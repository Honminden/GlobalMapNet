from enum import Enum
from shapely.geometry import LineString, CAP_STYLE, JOIN_STYLE


class MapNMSPurgeMode(Enum):
    NONE_NMS = 0
    PURGE_BUFFER_ZONE = 1
    PURGE_BY_BIOU = 2


class MapNMSScoreMode(Enum):
    LENGTH = 0
    CONFIDENCE = 1


def map_nms_purge(global_map, buffer_distance=0.5, threshold=0.2, purge_mode=MapNMSPurgeMode.PURGE_BUFFER_ZONE, nms_mode=MapNMSScoreMode.LENGTH):
    global_map_purged = global_map[:]
    if purge_mode == MapNMSPurgeMode.NONE_NMS:
        return global_map_purged

    global_map_purged = global_map[:]

    map_elements_by_rank = dict()

    for map_element in global_map:
        category = map_element['category']
        if category not in map_elements_by_rank.keys():
            map_elements_by_rank[category] = list()
        map_elements_by_rank[category].append(map_element)

    for category in map_elements_by_rank.keys():
        if nms_mode == MapNMSScoreMode.LENGTH:
            map_elements_by_rank[category].sort(key=lambda x: LineString(x['coords']).length, reverse=True)
        elif nms_mode == MapNMSScoreMode.CONFIDENCE:
            map_elements_by_rank[category].sort(key=lambda x: x['details']['score'], reverse=True)
        else:
            raise NotImplementedError(f"Unknown nms_mode {nms_mode}")

        buffer_distance_i = buffer_distance
        if isinstance(buffer_distance, dict):
            buffer_distance_i = buffer_distance[category]
        threshold_i = threshold
        if isinstance(threshold, dict):
            threshold_i = threshold[category]

        while len(map_elements_by_rank[category]) > 0:
            map_element = map_elements_by_rank[category].pop(0)
            buffer_zone = LineString(map_element['coords']).buffer(buffer_distance_i, cap_style=CAP_STYLE.round, join_style=JOIN_STYLE.round)
            purge_check_list = map_elements_by_rank[category][:]
            for map_element_to_check in purge_check_list:
                if purge_mode == MapNMSPurgeMode.PURGE_BUFFER_ZONE:
                    should_purge = buffer_zone.intersects(LineString(map_element_to_check['coords']))
                elif purge_mode == MapNMSPurgeMode.PURGE_BY_BIOU:
                    buffer_zone_to_check = LineString(map_element_to_check['coords']).buffer(buffer_distance_i, cap_style=CAP_STYLE.round, join_style=JOIN_STYLE.round)
                    area_intersection = buffer_zone.intersection(buffer_zone_to_check).area
                    area_union = buffer_zone.union(buffer_zone_to_check).area
                    biou = area_intersection / (area_union + 1e-9)
                    should_purge = biou > threshold_i
                else:
                    raise NotImplementedError(f"Unknown purge_mode {purge_mode}")
                if should_purge:
                    map_elements_by_rank[category].remove(map_element_to_check)
                    global_map_purged.remove(map_element_to_check)
    
    return global_map_purged