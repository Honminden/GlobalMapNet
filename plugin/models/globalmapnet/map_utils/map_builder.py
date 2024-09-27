import copy
import json
import os
import numpy as np
from enum import Enum
from shapely import affinity, wkt
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import linemerge, unary_union

from .functional.ego import formalize_pose, get_trans_and_angle_2d, generate_patch_box
from .functional import (simple_replace, match_replace, map_nms_purge, MapNMSPurgeMode, MapNMSScoreMode, init_map_element_from)
from .postprocess.utils import split_collections, get_drivable_area_contour


# global_map
# |-meta_info, dict of meta information
# |-poses, list of translation and rotation
# |-map_elements, list of map elements
# |---map_element, dict
# |-----category, str
# |-----coords, list of coords
# |-----details, dict [optional]
# |-------score, float [optional]
# |-------timestamp, int [optional], in milliseconds, please refer to .functional.misc.unified_timestamp
# |-------...


class MapReplaceMode(Enum):
    SIMPLE_REPLACE = 0
    MATCH_REPLACE = 1
    MULTI_FRAME_VOTE = 2


class MapBuilder:
    def __init__(self, patch_size=(60, 30), root_dir=r'./cache/global_maps', threshold=0.05, cache_patch_traces=True, cross_scene_eval=False):
        self.patch_size = patch_size # (W, H), W is parallel to ego vehicle's forward direction, H is parallel to ego vehicle's left direction
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        self.threshold = threshold

        self.global_maps = dict() # map_name -> global_map
        self.cache_patch_traces = cache_patch_traces
        if cache_patch_traces:
            self.patch_traces = dict() # map_name -> patch_trace
        self.cross_scene_eval = cross_scene_eval

    def init_global_map(self, map_name, meta_info=None, load_path=None):
        assert map_name not in self.global_maps, f"Map {map_name} already exists"
        try:
            json.dumps(meta_info)
        except TypeError:
            print("Unable to serialize meta_info")

        self.global_maps[map_name] = dict()
        if load_path is not None:
            with open(load_path, 'r') as f:
                global_map_loaded = json.load(f)

                if global_map_loaded['meta_info'] is None:
                    self.global_maps[map_name]['meta_info'] = dict()
                else:
                    self.global_maps[map_name]['meta_info'] = global_map_loaded['meta_info']
                
                if global_map_loaded['poses'] is None:
                    self.global_maps[map_name]['poses'] = list()
                else:
                    self.global_maps[map_name]['poses'] = global_map_loaded['poses']

                if global_map_loaded['map_elements'] is None:
                    self.global_maps[map_name]['map_elements'] = list()
                else:
                    self.global_maps[map_name]['map_elements'] = global_map_loaded['map_elements']
        else:
            if meta_info is None:
                meta_info = dict()
            self.global_maps[map_name]['meta_info'] = meta_info
            self.global_maps[map_name]['poses'] = list()
            self.global_maps[map_name]['map_elements'] = list()
        
        if self.cache_patch_traces:
            self.patch_traces[map_name] = self._init_patch_trace(map_name)
        
        if self.cross_scene_eval:
            self._inherit_map(map_name)

    @property
    def global_map_names(self):
        return list(self.global_maps.keys())
    
    def map_exists(self, map_name):
        return map_name in self.global_map_names

    def assert_map_exists(self, map_name):
        assert self.map_exists(map_name), f"Map {map_name} not found"

    def get_global_map(self, map_name):
        self.assert_map_exists(map_name)
        return self.global_maps[map_name]

    def save_map(self, map_name):
        self.assert_map_exists(map_name)
        with open(os.path.join(self.root_dir, f'{map_name}.json'), 'w') as f:
            json.dump(self.global_maps[map_name], f)
    
    def _init_patch_trace(self, map_name):
        self.assert_map_exists(map_name)
        poses = self.global_maps[map_name]['poses']
        patch_boxes = [generate_patch_box(self.patch_size, pose) for pose in poses]
        patch_trace = unary_union(patch_boxes).wkt
        return patch_trace

    def _update_patch_trace(self, map_name, pose, patch_size=None):
        self.assert_map_exists(map_name)
        if patch_size is None:
            patch_size = self.patch_size
        
        patch_box = generate_patch_box(patch_size, pose)
        patch_trace = self.patch_traces[map_name]
        patch_trace_shape = wkt.loads(patch_trace)
        patch_trace = unary_union([patch_trace_shape, patch_box]).wkt
        self.patch_traces[map_name] = patch_trace

    def _get_patch_trace_shape(self, map_name):
        if not self.map_exists(map_name):
            return Polygon() # empty area
        if self.cache_patch_traces:
            return wkt.loads(self.patch_traces[map_name])
        else:
            return wkt.loads(self._init_patch_trace(map_name))

    def get_traced_region_boundaries(self, map_name, pose, bev_shape, patch_size=None):
        if patch_size is None:
            patch_size = self.patch_size

        bev_H, bev_W = bev_shape

        patch_trace_shape = self._get_patch_trace_shape(map_name)
        patch_box, translation, patch_angle = generate_patch_box(patch_size, pose, return_trans_and_angle=True)
        traced_region = patch_trace_shape.intersection(patch_box)
        if traced_region.is_empty:
            return []
        traced_region = affinity.rotate(traced_region, -patch_angle, origin=(translation[0], translation[1]), use_radians=False)
        traced_region = affinity.affine_transform(traced_region, [1.0, 0.0, 0.0, 1.0, -translation[0], -translation[1]])
        traced_region = affinity.affine_transform(traced_region, [bev_W / patch_size[0], 0, 0, bev_H / patch_size[1], bev_W / 2.0, bev_H / 2.0])
        traced_region_parts = split_collections(traced_region)
        traced_region_boundaries = get_drivable_area_contour(traced_region_parts)
        return traced_region_boundaries

    def _inherit_map(self, map_name):
        if not self.cross_scene_eval or 'init_pose' not in self.global_maps[map_name]['meta_info']:
            return
        pose = self.global_maps[map_name]['meta_info']['init_pose']
        ego_pose_point = Point(pose[:2])
        # find the latest map that contains the ego pose point
        for history_map_name, patch_trace in reversed(self.patch_traces.items()):
            patch_trace_shape = wkt.loads(patch_trace)
            if patch_trace_shape.contains(ego_pose_point):
                # init with the history map
                history_map = self.global_maps[history_map_name]
                self.global_maps[map_name]['map_elements'] = copy.deepcopy(history_map['map_elements'])
                self.global_maps[map_name]['poses'] = copy.deepcopy(history_map['poses'])
                self.patch_traces[map_name] = patch_trace
                self.global_maps[map_name]['meta_info']['parent'] = history_map_name
                print(f"!DEBUG: map inheritence: {history_map_name} -> {map_name}")
                break

    def get_local_maps(self, map_name, poses, patch_size=None, category_filter=None, to_ego_coords=False, use_3d_patch=False):
        self.assert_map_exists(map_name)
        assert poses is not None and isinstance(poses, list), "Poses should be a list of translation and rotation"

        if patch_size is None:
            patch_size = self.patch_size
        if use_3d_patch:
            box_common = box(-patch_size[0] / 2, -patch_size[1] / 2, patch_size[0] / 2, patch_size[1] / 2)
            boxes_with_trans_and_angle = [(box_common,) + get_trans_and_angle_2d(pose, return_degree=True) for pose in poses]
        else:
            boxes_with_trans_and_angle = [generate_patch_box(patch_size, pose, return_trans_and_angle=True) for pose in poses]

        local_maps = [list() for _ in boxes_with_trans_and_angle] # local_maps-map_elements-map_element-category, coords
        for map_element in self.global_maps[map_name]['map_elements']:
            if category_filter is not None and map_element['category'] not in category_filter:
                continue
            line = LineString(map_element['coords'])
            
            if use_3d_patch:
                # av2 gt local map extraction using 3d vectors, not used currently!
                assert line.has_z, "expect av2 geoms to be 3d but receive 2d"
                for i, (pose, (box_, translation, patch_angle)) in enumerate(zip(poses, boxes_with_trans_and_angle)):
                    e2g_translation = np.array(pose[:3])
                    e2g_rotation = Quaternion(pose[3:]).rotation_matrix
                    
                    g2e_translation = e2g_rotation.T.dot(-e2g_translation)
                    g2e_rotation = e2g_rotation.T
                    new_line = np.array(line.coords) @ g2e_rotation.T + g2e_translation
                    new_line = LineString(new_line)

                    new_line = new_line.intersection(box_)
                    if not new_line.is_empty:
                        if new_line.geom_type == 'MultiLineString':
                            new_line = linemerge(new_line)
                            if new_line.geom_type == 'MultiLineString':
                                for new_polyline in new_line.geoms:
                                    coords = np.array(new_polyline.coords)
                                    if not to_ego_coords:
                                        coords = coords @ e2g_rotation.T + e2g_translation
                                    local_maps[i].append(init_map_element_from(map_element, coords=coords[:, :2].tolist()))
                            else:
                                coords = np.array(new_line.coords)
                                if not to_ego_coords:
                                    coords = coords @ e2g_rotation.T + e2g_translation
                                local_maps[i].append(init_map_element_from(map_element, coords=coords[:, :2].tolist()))
                        else:
                            coords = np.array(new_line.coords)
                            if not to_ego_coords:
                                coords = coords @ e2g_rotation.T + e2g_translation
                            local_maps[i].append(init_map_element_from(map_element, coords=coords[:, :2].tolist()))
            else:
                for i, (box_, translation, patch_angle) in enumerate(boxes_with_trans_and_angle):
                    new_line = line.intersection(box_)
                    if not new_line.is_empty:
                        if to_ego_coords:
                            new_line = affinity.rotate(new_line, -patch_angle, origin=(translation[0], translation[1]), use_radians=False)
                            new_line = affinity.affine_transform(new_line,
                                                                [1.0, 0.0, 0.0, 1.0, -translation[0], -translation[1]])
                        if new_line.geom_type == 'MultiLineString':
                            new_line = linemerge(new_line)
                            if new_line.geom_type == 'MultiLineString':
                                for new_polyline in new_line.geoms:
                                    local_maps[i].append(init_map_element_from(map_element, coords=list(new_polyline.coords)))
                            else:
                                local_maps[i].append(init_map_element_from(map_element, coords=list(new_line.coords)))
                        else:
                            local_maps[i].append(init_map_element_from(map_element, coords=list(new_line.coords)))

        return local_maps
    
    def update_global_map(self, map_name, local_map, pose, patch_size=None, from_ego_coords=False, adjust_rot_angle=None,
                           replace_mode=MapReplaceMode.SIMPLE_REPLACE, 
                           nms_purge_mode=MapNMSPurgeMode.PURGE_BUFFER_ZONE, nms_score_mode=MapNMSScoreMode.LENGTH, **kwargs):
        self.assert_map_exists(map_name)

        # get patch from pose
        pose = formalize_pose(pose)
        if patch_size is None:
            patch_size = self.patch_size
        patch_box, translation, patch_angle = generate_patch_box(patch_size, pose, return_trans_and_angle=True)
        
        # transform local map from ego coords to global coords if necessary
        if from_ego_coords:
            local_map = copy.deepcopy(local_map)
            for map_element in local_map:
                line = LineString(map_element['coords'])
                line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, translation[0], translation[1]])
                line = affinity.rotate(line, patch_angle, origin=(translation[0], translation[1]), use_radians=False)
                if adjust_rot_angle is not None:
                    line = affinity.rotate(line, adjust_rot_angle, origin=(translation[0], translation[1]), use_radians=False)
                map_element['coords'] = list(line.coords)

        # map replace
        if replace_mode == MapReplaceMode.SIMPLE_REPLACE:
            threshold = kwargs.get('threshold', self.threshold) # get(key_name, default_value)

            self.global_maps[map_name]['map_elements'] = simple_replace(self.global_maps[map_name]['map_elements'], local_map, patch_box,
                                                                         threshold=threshold)
            self.global_maps[map_name]['poses'].append(pose)
        elif replace_mode == MapReplaceMode.MATCH_REPLACE:
            threshold = kwargs.get('threshold', self.threshold)
            sample_num = kwargs.get('sample_num', 100)
            simplify = kwargs.get('simplify', True)

            self.global_maps[map_name]['map_elements'] = match_replace(self.global_maps[map_name]['map_elements'], local_map, patch_box,
                                                                        threshold=threshold, sample_num=sample_num, simplify=simplify)
            self.global_maps[map_name]['poses'].append(pose)
        elif replace_mode == MapReplaceMode.MULTI_FRAME_VOTE:
            raise NotImplementedError("Multi-frame vote is not implemented yet")
        
        # map nms purge
        buffer_distance = kwargs.get('buffer_distance', 0.5)
        biou_threshold = kwargs.get('biou_threshold', 0.2)

        self.global_maps[map_name]['map_elements'] = map_nms_purge(self.global_maps[map_name]['map_elements'], buffer_distance=buffer_distance, threshold=biou_threshold, purge_mode=nms_purge_mode, nms_mode=nms_score_mode)

        if self.cache_patch_traces:
            self._update_patch_trace(map_name, pose, patch_size)


if __name__ == '__main__':
    # import datetime
    map_builder = MapBuilder(root_dir=r'./cache/global_maps')
    # map_builder.init_global_map('test_map', {'name': 'test_map', 'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    # map_builder.save_map('test_map')
    map_builder.init_global_map('test_gt', load_path=r'./cache/global_maps/boston-seaport_gt_test.json')
    print(map_builder.get_global_map('test_gt')['meta_info'])

    # test single local map
    pose = [427.96858346929594, 1622.1558281210846, 0.0, 0.27668389210179944, -0.0026796705507768723, 0.00471221945277763, 0.9609456999253227]
    local_map = map_builder.get_local_maps('test_gt', [pose])[0]
    print(local_map)
    with open(r'./cache/global_maps/local_map.json', 'w') as f:
        json.dump(local_map, f)
    local_map_ego = map_builder.get_local_maps('test_gt', [pose], to_ego_coords=True)[0]
    print(local_map_ego)
    with open(r'./cache/global_maps/local_map_ego.json', 'w') as f:
        json.dump(local_map_ego, f)

    # test sequential local maps
    with open(r'./cache/global_maps/seq_ego_poses.json', 'r') as f:
        poses = json.load(f)
    local_maps = map_builder.get_local_maps('test_gt', poses)
    with open(r'./cache/global_maps/local_maps.json', 'w') as f:
        json.dump(local_maps, f)
    print([len(local_map) for local_map in local_maps])