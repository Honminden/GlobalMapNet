
# global ID format:
# map(6,capital)+category(2,capital)+index(8,decimal)
# eg: NUSCSORB00000226 = map:nuscenes-singapore-onenorth, category:road boundary, index:226

# TODO add argoverse2 maps
NUSC = 'nuscenes'
valid_datasets = [NUSC]

def map_id_from_location(dataset, location):
    return f'{dataset}-{location}'

def map_id_to_location(map_id):
    return map_id.split('-')[-1]

# nuscenes_locations = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
map_mapper = {
    'NUSCBS': f'{NUSC}-boston-seaport',
    'NUSCSH': f'{NUSC}-singapore-hollandvillage',
    'NUSCSO': f'{NUSC}-singapore-onenorth',
    'NUSCSQ': f'{NUSC}-singapore-queenstown',
}
map_mapper_inv = {v: k for k, v in map_mapper.items()}

category_mapper = {
    'RB': 'road', # road boundary
    'LD': 'lane', # lane divider
    'PC': 'ped', # pedestrian crossing
}
category_mapper_inv = {v: k for k, v in category_mapper.items()}

def map_element2global_id(dataset, location, category, index):
    map_id = map_id_from_location(dataset, location)
    return f'{map_mapper_inv[map_id]}{category_mapper_inv[category]}{index:08d}'

def global_id2map_element(global_id):
    map_id = global_id[:6]
    category = global_id[6:8]
    index = int(global_id[8:16])
    return map_id_to_location(map_mapper[global_id[:6]]), category_mapper[global_id[6:8]], int(global_id[8:16])

def index_map_elements(global_road_lines, global_lane_lines, global_ped_lines, location, dataset=NUSC):
    if dataset not in valid_datasets:
        raise NotImplementedError(f"dataset {dataset} not implemented")

    line_dict = {
        'road': global_road_lines,
        'lane': global_lane_lines,
        'ped': global_ped_lines,
    }

    # sort lines by x+y of centroids
    line_indexed_dict = dict()
    for category, lines in line_dict.items():
        centroids = [list(line.centroid.coords)[0] for line in lines]
        lines_sort = sorted(zip(centroids, lines), key=lambda x: x[0][0] + x[0][1])
        lines = [line for _, line in lines_sort]
        global_ids = [map_element2global_id(dataset, location, category, idx) for idx in range(len(lines))]
        line_indexed_dict[category] = dict(zip(global_ids, lines))

    return line_indexed_dict['road'], line_indexed_dict['lane'], line_indexed_dict['ped']