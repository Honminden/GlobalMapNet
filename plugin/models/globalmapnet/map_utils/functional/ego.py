import numpy as np
from pyquaternion import Quaternion
from shapely import affinity
from shapely.geometry import box


def formalize_pose(pose):
    assert len(pose) == 3 or len(pose) == 7, "Pose should be a list of length 3 or 7"
    if len(pose) == 3:
        translation = pose[:2]
        patch_angle = pose[2]
        quaternion = list(Quaternion(axis=[0, 0, 1], degrees=patch_angle))
        return translation + [0.0] + quaternion
    else:
        return pose


def get_trans_and_angle_2d(pose, return_degree=False, concat=False):
    pose = formalize_pose(pose)
    translation = pose[:2]
    rotation = pose[3:]
    angle = Quaternion(rotation).yaw_pitch_roll[0]
    if return_degree:
        angle = angle / np.pi * 180.0
    if concat:
        return translation + [angle]
    else:
        return translation, angle


def generate_patch_box(patch_size, pose, return_trans_and_angle=False):
    translation, patch_angle = get_trans_and_angle_2d(pose, return_degree=True)

    patch_box = box(translation[0] - patch_size[0] / 2.0, translation[1] - patch_size[1] / 2.0, 
                        translation[0] + patch_size[0] / 2.0, translation[1] + patch_size[1] / 2.0)
    patch_box = affinity.rotate(patch_box, patch_angle, origin=(translation[0], translation[1]), use_radians=False)

    if return_trans_and_angle:
        return patch_box, translation, patch_angle
    else:
        return patch_box