# -*- coding: utf-8 -*-
"""
This code is to project bounding boxes to the lidar frame and 
visualize the projected bounding boxes in the lidar frame.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import numpy as np
import open3d as o3d

from tools.utils.yaml_utils import load_yaml


def global_to_local(global_pos, lidar_pose):
    """
    Converts global coordinates to a local coordinate system.
    :param global_pos: Global position [x, y, z]
    :param lidar_pose: Lidar pose [x, y, z, roll, pitch, yaw]
    :return: [x, y, z] in the local coordinate system
    """
    # Extract translation and rotation information
    lidar_translation = np.array(lidar_pose[:3])  # Extracts the translation vector
    lidar_rotation = np.array(lidar_pose[3:])    # Extracts the rotation (roll, pitch, yaw)
    
    # If roll, pitch, and yaw are all zero, only translation is considered
    if np.all(lidar_rotation == 0):
        return np.array(global_pos) - lidar_translation  # Simple subtraction for translation

    # Otherwise, account for rotation
    # Compute rotation matrix from roll, pitch, and yaw
    roll, pitch, yaw = lidar_rotation
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])  # Rotation around the x-axis (roll)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])  # Rotation around the y-axis (pitch)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])  # Rotation around the z-axis (yaw)
    R = Rz @ Ry @ Rx  # Combined rotation matrix in the order of yaw → pitch → roll
    
    # Convert to local coordinates
    local_pos = R.T @ (np.array(global_pos) - lidar_translation)  # Transform global position to local
    return local_pos

def decode_yaml(yaml_file):

    yaml_param = load_yaml(yaml_file)
    # get the key of yaml_param
    key_list = list(yaml_param.keys())
    #print("key:",key_list)

    lidar_key_list = []
    camera_key_list = []
    # get the lidar_pose from yaml_param
    for key in key_list:
        if "lidar_pose" in key:
            lidar_key_list.append(key)
        if "camera" in key or "cam" in key:
            camera_key_list.append(key)
    
    lidar_pose_list = []
    for key in lidar_key_list:
        lidar_pose = yaml_param[key]
        lidar_pose_list.append(lidar_pose)

    camera_list = []
    for key in camera_key_list:
        camera_pose = yaml_param[key]
        camera_list.append(camera_pose)

    if "vehicles" in yaml_param:
        vehicle_dict = yaml_param["vehicles"]
    else:
        vehicle_dict = {}
    if "pedestrians" in yaml_param:
        pedestrian_dict = yaml_param["pedestrians"]
    else:
        pedestrian_dict = {}


    return lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict

# =============================================================================
# Test Functions
# =============================================================================
def test():
    yaml_file = "data_dumping/example/2024_11_30_16_15_46/125/000031.yaml"
    lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict = decode_yaml(yaml_file)
    print(vehicle_dict)

    # load vehicle in vehiecle dict one by one
    for key in vehicle_dict.keys():
        vehicle = vehicle_dict[key]
        location = vehicle["location"]
        angle =  vehicle["angle"]
        extent = vehicle["extent"]
        # convert location into a list [x, y, z, roll, pitch, yaw]
        # location.extend([0,0,0])
        # print("location:",location)
        # convert global location to local location
        local_location = global_to_local(location, lidar_pose_list[0])
        print("local_location:",local_location)
        print("angle:",angle)
        print("extent:",extent)


if __name__ == "__main__":
    test()


        

