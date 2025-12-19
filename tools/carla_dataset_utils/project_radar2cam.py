# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
import yaml
import matplotlib.pyplot as plt
import vis_pcd

try:
    import carla
    _CARLA_AVAILABLE = True
except ImportError:
    carla = None
    _CARLA_AVAILABLE = False

def create_transformation(x, y, z, roll, yaw, pitch):
    """
    Build forward and inverse homogeneous transforms from CARLA-style pose parameters.

    When the `carla` module is available, use its native API for consistency with the
    simulator; otherwise fall back to NumPy-based reconstruction.
    """
    if _CARLA_AVAILABLE and carla is not None:
        location = carla.Location(x=x, y=y, z=z)
        rotation = carla.Rotation(roll=roll, yaw=yaw, pitch=pitch)
        carla_transformation = carla.Transform(location, rotation)
        return carla_transformation.get_matrix(), carla_transformation.get_inverse_matrix()

    forward = to_homogeneous_matrix(x, y, z, roll, yaw, pitch)
    inverse = np.linalg.inv(forward)
    return forward, inverse

def to_homogeneous_matrix(x, y, z, roll, yaw, pitch):
    """
    Convert translation and rotation (Euler angles) into a 4x4 homogeneous transformation matrix.

    :param x: Translation in X.
    :param y: Translation in Y.
    :param z: Translation in Z.
    :param roll: Rotation about X-axis (degrees).
    :param yaw: Rotation about Z-axis (degrees).
    :param pitch: Rotation about Y-axis (degrees).
    :return: 4x4 NumPy array representing the homogeneous transformation matrix.
    """
    # Convert degrees to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # Rotation matrix components
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix (Rz * Ry * Rx)
    R = Rz @ Ry @ Rx

    # Create the 4x4 homogeneous transformation matrix
    T = np.eye(4)  # Initialize as identity matrix
    T[:3, :3] = R  # Top-left 3x3 block is the rotation matrix
    T[:3, 3] = [x, y, z]  # Top-right 3x1 block is the translation vector

    return T

def load_yaml(file):
    """Load sensor calibration YAML."""
    with open(file, 'r') as f:
        return yaml.safe_load(f)

def process_image(file_path):
    """Load image into numpy array (RGB)."""
    img = Image.open(file_path)
    arr = np.array(img,  dtype=np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[:,:,:3]
    return arr

def load_radar(file_path):
    """Load radar points: Nx4 (x, y, z, intensity/velocity)"""
    # Read the PCD file
    pcd = o3d.io.read_point_cloud(file_path)

    # Extract x, y, z coordinates
    points = np.asarray(pcd.points, dtype=np.float32)

    # Radar PCD may include intensity/rcs in colors or custom fields
    if hasattr(pcd, 'colors') and pcd.colors is not None and len(pcd.colors) > 0:
        rcs = np.mean(np.asarray(pcd.colors), axis=1).reshape(-1, 1)
    else:
        rcs = np.zeros((points.shape[0], 1), dtype=np.float32)

    radar_cloud = np.hstack((points, rcs))
    return radar_cloud

def project_radar_to_camera(image_w, image_h, im_array, radar_points, K,
                            radar_2_world, world_2_camera, output_dir, frame, text=""):
    """
    Project radar points onto camera image and save.
    """
    points_local = radar_points[:,:3].T
    points_h = np.vstack([points_local, np.ones(points_local.shape[1])])
    world_points = radar_2_world @ points_h
    sensor_points = world_2_camera @ world_points

    # CARLA UE4 -> camera standard
    points_cam = np.array([sensor_points[1],
                           -sensor_points[2],
                           sensor_points[0]])

    points_2d = K @ points_cam
    points_2d = np.array([points_2d[0,:]/points_2d[2,:],
                          points_2d[1,:]/points_2d[2,:],
                          points_2d[2,:]])

    points_2d = points_2d.T
    intensity = radar_points[:,3]
    mask = (points_2d[:,0]>=0) & (points_2d[:,0]<image_w) & \
           (points_2d[:,1]>=0) & (points_2d[:,1]<image_h) & \
           (points_2d[:,2]>0)
    points_2d = points_2d[mask]
    intensity = intensity[mask]

    # Color map based on intensity
    opacity = 0.2
    norm = (intensity - intensity.min()) / (intensity.ptp()+1e-6)
    cmap = plt.get_cmap('jet')
    colors = (cmap(norm)[:,:3]*255).astype(np.uint8)

    dot_extent = 2
    for i in range(len(points_2d)):
        x_min = max(0,int(points_2d[i,0])-dot_extent)
        x_max = min(image_w,int(points_2d[i,0])+dot_extent)
        y_min = max(0,int(points_2d[i,1])-dot_extent)
        y_max = min(image_h,int(points_2d[i,1])+dot_extent)
        im_array[y_min:y_max,x_min:x_max] = (opacity * colors[i] + (1-opacity) * \
                                                 im_array[y_min:y_max, x_min:x_max]
                                                ).astype(np.uint8)

    img = Image.fromarray(im_array)
    draw = ImageDraw.Draw(img)
    draw.text((10,10), text, fill=(255,255,255))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(output_dir,f"{frame}_{timestamp}.png")
    img.save(file_name)
    print("Saved radar overlay:", file_name)

def project_save_single_radar_frame(img_file, radar_file, yaml_file, output_dir, frame, cam_index, radar_index):
    """Full pipeline: load data, compute transforms, project radar points."""
    data = load_yaml(yaml_file)

    # camera
    camera_id = 'camera' + cam_index
    camera_param = data[camera_id]
    cam_pos = camera_param['cords']

    K = np.array(camera_param['intrinsic'])
    image_w, image_h = K[0,2]*2, K[1,2]*2
    
    _, world_2_camera = create_transformation(*cam_pos)

    # radar
    radar_id = 'radar_pose' + radar_index
    radar_param = data[radar_id]
    radar_pos = radar_param

    im_array = process_image(img_file)
    radar_points = load_radar(radar_file)   

    radar_2_world, _ = create_transformation(*radar_pos)

    #print(camera_id, radar_id)

    project_radar_to_camera(int(image_w), int(image_h), im_array, radar_points,
                            K, radar_2_world, world_2_camera, output_dir, frame, text=str(radar_pos[4]))

if __name__ == "__main__":
    root = "/media/carma/ui_4/data_transfer/data_dumping/new_test_dataset/"
    config_yamls = ["test_town04"]

    for config_yaml in config_yamls:
        for i in range(31, 51):
            for j in range(4):
                frame = f"{i:06d}"
                sensor_id = str(j)

                config_yaml_frame = config_yaml + "/-125/" + frame
                cam_file_path = root + config_yaml_frame + "_camera" + sensor_id + ".png"
                radar_file_path = root + config_yaml_frame + "_radar" + sensor_id + ".pcd"
                yaml_file = root + config_yaml_frame + ".yaml"
                output_dir =  "/home/carma/dg/results/radar2cam/" + config_yaml + "/" + sensor_id
                project_save_single_radar_frame(cam_file_path, radar_file_path, yaml_file, output_dir, 
                                                frame, cam_index=sensor_id, radar_index=sensor_id)
                
                #vis_pcd.test(radar_file_path)