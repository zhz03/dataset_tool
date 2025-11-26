# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import time

import cv2
import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm

from tools.carla_dataset_utils import box_utils
from tools.carla_dataset_utils import common_utils

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

import open3d as o3d
import numpy as np
import yaml
import re
import yaml
import os
import math


def visualize_two_pcds(output_dir, lidar_pcd_xyz, radar_pcd_xyz, open_in_o3d):
    """
    Visualize two point clouds together.

    Parameters
    ----------
    lidar_pcd_xyz : np.ndarray
        Shape (N, 3)
    radar_pcd_xyz : np.ndarray
        Shape (M, 3)
    color1 : tuple
        RGB for first point cloud
    color2 : tuple
        RGB for second point cloud
    """
    color1=(1, 0, 0)
    color2=(0, 0, 1)

    # Convert to Open3D point cloud objects
    lidar = o3d.geometry.PointCloud()
    lidar.points = o3d.utility.Vector3dVector(lidar_pcd_xyz)
    lidar.paint_uniform_color(color1)

    radar = o3d.geometry.PointCloud()
    radar.points = o3d.utility.Vector3dVector(radar_pcd_xyz)
    radar.paint_uniform_color(color2)

    # Visualize
    if open_in_o3d:
        o3d.visualization.draw_geometries([lidar, radar])
    else: 
        save_o3d_visualization([lidar, radar], output_dir)

def save_o3d_visualization(element, save_path):
    """
    Save the open3d drawing to folder.

    Parameters
    ----------
    element : list
        List of o3d.geometry objects.

    save_path : str
        The save path.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    for i in range(len(element)):
        vis.add_geometry(element[i])
        vis.update_geometry(element[i])

    vis.poll_events()
    vis.update_renderer()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vis.capture_screen_image(save_path)
    vis.destroy_window()
    print("Saved to: ", save_path)

def load_yaml(file, opt=None):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    opt : argparser
         Argparser.
    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 'config.yaml')

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param

def load_pcd(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    # pcd.points is o3d.utility.Vector3dVector
    points_xyz = np.asarray(pcd.points)  # shape (N, 3)

    # If there's intensity, sometimes it's in pcd.colors or separate channels,
    # depends on how the PCD is formatted. For simplicity, we assume we only have xyz.
    # If you do have intensity, you might store it in a separate array.

    return points_xyz

def parse_vehicle_bbox(vehicle_info):
    """
    Parse vehicle info into bounding box representation [x, y, z, h, w, l, yaw].
    vehicle_info: dict with keys 'location', 'extent', 'angle'
    """
    location = vehicle_info['location']  # [x_world, y_world, z_world]
    extent = vehicle_info['extent']     # [l, w, h] (half-size or full-size depends on dataset)
    angle = vehicle_info['angle']       # [roll, yaw, pitch]

    # Extract values
    x, y, z = location
    l, w, h = extent
    roll, yaw, pitch = angle  # YAML order: roll, yaw, pitch

    # Convert yaw to radians
    yaw = np.deg2rad(yaw)

    # Double the dimensions (assumes half-extent is provided)
    l *= 2
    w *= 2
    h *= 2

    return np.array([x, y, z, h, w, l, yaw], dtype=np.float32)

def euler_to_rot(roll, yaw, pitch):
    """Build a 3x3 rotation matrix given roll, yaw, pitch in radians."""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    # Roll -> Yaw -> Pitch (Z-Y-X order)
    
    return Rz @ Ry @ Rx

def invert_pose(lidar_pose):
    """
    lidar_pose: [x, y, z, roll, yaw, pitch] in (left-handed) world frame (degrees)
    return: 4x4 transform from WORLD -> LIDAR in a right-handed convention
    """
    x, y, z, roll_deg, yaw_deg, pitch_deg = lidar_pose

    # Convert degrees to radians
    roll = np.deg2rad(roll_deg)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    # Build forward transform T (world -> lidar)
    T = np.eye(4)
    R = euler_to_rot(roll, yaw, pitch)  # Use the updated euler_to_rot
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    # Invert to get T_lidar_world
    T_inv = np.linalg.inv(T)
    return T_inv

def transform_points(points, T):
    """
    Transform points (N,3) using 4x4 matrix T.
    """
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])  # (N,4)
    points_transformed = (T @ points_h.T).T[:, :3]
    return points_transformed

def main(output_dir, yaml_file, lidar_pcd_file, radar_pcd_file, sensor_id, open_in_o3d):
    # (1) Load data
    data = load_yaml(yaml_file)
    lidar_pose = data['lidar_pose0']  # LiDAR pose in the world frame [x, y, z, roll, pitch, yaw]
    radar_id = 'radar_pose' + str(sensor_id)
    radar_pose = data[radar_id]

    # Load LiDAR points (already in LiDAR frame)
    lidar_pcd_xyz = load_pcd(lidar_pcd_file)  # Shape: (N, 3)
    radar_pcd_xyz = load_pcd(radar_pcd_file)


    # (3) Build transform from WORLD to LIDAR frame
    T_world_lidar = invert_pose(lidar_pose)  # shape (4,4)
    T_radar_world = np.linalg.inv(invert_pose(radar_pose))
    radar_pcd_align = transform_points(radar_pcd_xyz, T_world_lidar @ T_radar_world)

    # (6) Visualize bounding boxes and LiDAR points in Open3D
    #vis_pcd = o3d.geometry.PointCloud()
    visualize_two_pcds(output_dir, lidar_pcd_xyz, radar_pcd_align, open_in_o3d)



def main_with_args(root_dir, output_dir, agent, frame, sensor_id, open_in_o3d):
    """
    Build the file paths from root_dir, agent, frame,
    then call the original 'main' function.
    """
    agent_str = str(agent)
    frame_str = f"{int(frame):06}"  # e.g., 72 -> "000072"

    yaml_file = f"{root_dir}/{agent_str}/{frame_str}.yaml"
    lidar_pcd_file  = f"{root_dir}/{agent_str}/{frame_str}_lidar0.pcd"
    radar_pcd_file  = f"{root_dir}/{agent_str}/{frame_str}_radar" + str(sensor_id) + ".pcd"

    main(output_dir, yaml_file, lidar_pcd_file, radar_pcd_file, sensor_id, open_in_o3d)


if __name__ == "__main__":
    # root = "/media/carma/aebdc025-05c3-40fe-a0e9-f424cfe2ae03/home/mobility/data_dumping/confirm/"
    # config_yamls = ["town05_intersection1_4cam_radar", "town05_intersection3_4cam_radar"]
    # agent = -125

    # for config_yaml in config_yamls:
    #     root_dir = root + config_yaml
    #     for i in range(31, 41):
    #         frame = i
    #         for j in range(4):
    #             sensor_id = j
    #             output_dir =  "/home/carma/dg/results/radar2lidar_3d/" + config_yaml + "/" + str(sensor_id) + "/" + f"{int(frame):06}.png"
    #             main_with_args(root_dir, output_dir, agent, frame, sensor_id, False)

    root_dir = '/media/carma/aebdc025-05c3-40fe-a0e9-f424cfe2ae03/home/mobility/data_dumping/radar_dataset/test/bridgeentry_town07_med_infra_radar_t_c_day_s7'
    agent    = -125
    frame    = 31
    main_with_args(root_dir, "", agent, frame, 2, True)