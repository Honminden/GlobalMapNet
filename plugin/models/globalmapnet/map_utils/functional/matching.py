import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString


def line_interpolate(line, sample_num=100):
    if isinstance(line, list):
        line = LineString(line)
    distances = np.linspace(0, line.length, sample_num)
    sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
    return sampled_points


def line_interpolate_distance(srcs, dsts, sample_num=100):
    # if coords, transform to LineString
    src_lines = list()
    for src in srcs:
        if isinstance(src, list):
            src_line = LineString(src)
            src_lines.append(src_line)
    
    dst_lines = list()
    for dst in dsts:
        if isinstance(dst, list):
            dst_line = LineString(dst)
            dst_lines.append(dst_line)

    distances = list()
    for src_line in src_lines:
        intervals = np.linspace(0, src_line.length, sample_num)
        distances_s = list()
        for interval in intervals:
            src_point = src_line.interpolate(interval)
            distances_i = list()
            for dst_line in dst_lines:
                distance = src_point.distance(dst_line)
                distances_i.append(distance)
            distances_s.append(distances_i)
        distances.append(distances_s)

    distances = np.array(distances).reshape(len(srcs), sample_num, len(dsts)) # (N, K, M), force shape when list is empty
    return distances.mean(axis=1) # (N, M)


def chamfer_distance_match(src_lines,
                           dst_lines,
                           cd_threshold=1.0,
                           sample_num=100, 
                           one_way=False):
    """
    Computes the chamfer distance matching between two sets of lines.

    Args:
        src_lines (list): List of N source lines, with type of shapely.geometry.LineString.
        dst_lines (list): List of M destination lines, with type of shapely.geometry.LineString.
        cd_threshold (float, optional): Threshold for accepting a match. Defaults to 1.0.
        sample_num (int, optional): Number of samples to interpolate along each line. Defaults to 100.
        one_way (bool, optional): Whether to match one way (src to dst) or not. Defaults to False.

    Returns:
        list: List of matched line pairs, where each pair is represented as (src_index, dst_index).

    """
    distance = line_interpolate_distance(src_lines, dst_lines, sample_num=sample_num) # (N, M)
    if not one_way:
        distance_reverse = line_interpolate_distance(dst_lines, src_lines, sample_num=sample_num) # (M, N)
        distance = np.minimum(distance, distance_reverse.T) # (N, M)

    src_indices, dst_indices = linear_sum_assignment(distance)

    matched_pairs = list()
    for src_index, dst_index in zip(src_indices, dst_indices):
        if distance[src_index, dst_index] <= cd_threshold:
            matched_pairs.append((src_index, dst_index))
    
    return matched_pairs