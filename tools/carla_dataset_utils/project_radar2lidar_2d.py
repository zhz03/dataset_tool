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

def load_yaml(file):
    """Load sensor calibration YAML."""
    with open(file, 'r') as f:
        return yaml.safe_load(f)

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

def create_transformation(x, y, z, roll, yaw, pitch):
    """
    Build a homogeneous transformation matrix to convert from sensor coordinates to world coordinates.
    
    :param x: Translation in X.
    :param y: Translation in Y.
    :param z: Translation in Z.
    :param roll: Rotation about X-axis (in degrees).
    :param yaw: Rotation about Z-axis (in degrees).
    :param pitch: Rotation about Y-axis (in degrees).
    :return: A 4x4 transformation matrix.
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
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T

def transform_to_world_coordinates(radar_points, radar_position):
    """
    Apply a transformation matrix to convert radar points to world coordinates.
    
    :param radar_points: Nx4 numpy array with radar points (x, y, z, intensity/velocity).
    :param radar_position: List or array of [x, y, z] representing the radar position in world coordinates.
    :param radar_rotation: List or array of [roll, yaw, pitch] representing the radar rotation in degrees.
    :return: Transformed radar points in world coordinates.
    """
    # Create transformation matrix from radar position and rotation
    transformation_matrix = create_transformation(*radar_position)

    # Convert radar points to homogeneous coordinates (add a column of 1s for homogeneous transformation)
    points_local = radar_points[:, :3].T  # Only use the x, y, z columns
    points_homogeneous = np.vstack([points_local, np.ones(points_local.shape[1])])  # Add a row of ones for homogeneous coordinates

    # Apply transformation: multiply the transformation matrix with the radar points
    points_world = transformation_matrix @ points_homogeneous

    # Return the transformed points (world coordinates)
    return points_world[:3, :].T

def plot_projection(plane, radar_points, lidar_points, output_dir, frame, plane_name):
    """Project radar points onto the specified plane (XY, XZ, or YZ) and save the image."""
    fig, ax = plt.subplots()

    if plane == "xy":
        ax.scatter(radar_points[:, 0], radar_points[:, 1], color='blue', alpha=0.1, s=0.01, label='radar')
        ax.scatter(lidar_points[:, 0], lidar_points[:, 1], color='red', alpha=0.1, s=0.01, label='lidar')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
    elif plane == "xz":
        ax.scatter(radar_points[:, 0], radar_points[:, 2], color='blue', alpha=0.1, s=0.01, label='radar')
        ax.scatter(lidar_points[:, 0], lidar_points[:, 2], color='red', alpha=0.1, s=0.01, label='lidar')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
    elif plane == "yz":
        ax.scatter(radar_points[:, 1], radar_points[:, 2], color='blue', alpha=0.1, s=0.01, label='radar')
        ax.scatter(lidar_points[:, 1], lidar_points[:, 2], color='red', alpha=0.1, s=0.01, label='lidar')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')

    ax.set_title(f"Radar Projection on {plane_name}-Plane")
    ax.legend(loc="upper right", markerscale=100)

    # Save the figure as an image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, plane_name)
    file_name = os.path.join(output_dir, f"{frame}_{timestamp}.png")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(file_name)
    plt.close(fig)
    print(f"Saved radar projection on {plane_name}-plane:", file_name)

def project_radar_on_planes(radar_points, lidar_points, output_dir, frame):
    """Project radar points onto the XY, XZ, and YZ planes."""
    # Project points onto different planes
    plot_projection("xy", radar_points, lidar_points, output_dir, frame, "XY")
    plot_projection("xz", radar_points, lidar_points, output_dir, frame, "XZ")
    plot_projection("yz", radar_points, lidar_points, output_dir, frame, "YZ")

def project_save_single_frame(radar_file, lidar_file, yaml_file, output_dir, frame, radar_index, lidar_index):
    """Full pipeline: load radar data, apply transformation, and project points onto planes."""
    # Load data from PCD files
    radar_points = load_radar(radar_file)
    lidar_points = load_radar(lidar_file)

    # Load YAML configuration for sensor positions
    data = load_yaml(yaml_file)
    radar_id = "radar_pose" + radar_index
    lidar_id = "lidar_pose" + str(lidar_index)
    radar_pos = data[radar_id] 
    lidar_pos = data[lidar_id]

    # Transform radar points from sensor coordinates to world coordinates
    radar_points_world = transform_to_world_coordinates(radar_points, radar_pos)
    lidar_points_world = transform_to_world_coordinates(lidar_points, lidar_pos)

    # Project radar points onto XY, XZ, and YZ planes using world coordinates
    project_radar_on_planes(radar_points_world, lidar_points_world, output_dir, frame)
# ------------------------------------------------------------
# 5. Entry point example
# ------------------------------------------------------------
# if __name__ == "__main__":
#     # Example usage — replace with your paths.
#     lidar_file = "/media/carma/aebdc025-05c3-40fe-a0e9-f424cfe2ae03/home/mobility/data_dumping/confirm/town05_intersection1_4cam_radar/-125/000031_lidar0.pcd"
#     radar_file = "/media/carma/aebdc025-05c3-40fe-a0e9-f424cfe2ae03/home/mobility/data_dumping/confirm/town05_intersection1_4cam_radar/-125/000031_radar0.pcd"
#     yaml_file  = "/media/carma/aebdc025-05c3-40fe-a0e9-f424cfe2ae03/home/mobility/data_dumping/confirm/town05_intersection1_4cam_radar/-125/000031.yaml"
#     output_dir = "/home/carma/dg/results/radar2lidar/"

#     project_save_single_frame(radar_file, lidar_file, yaml_file, output_dir, radar_index=0, lidar_index=0)

if __name__ == "__main__":
    root = "/media/carma/ui_4/data_transfer/data_dumping/radar_dataset/test/"
    config_yamls = ["bridgeentry_town07_dense_infra_radar_t_c_day_s27"]

    for config_yaml in config_yamls:
        for i in range(31, 51):
            for j in range(4):
                frame = f"{i:06d}"
                sensor_id = str(j)

                config_yaml_frame = config_yaml + "/-125/" + frame
                radar_file_path = root + config_yaml_frame + "_radar" + sensor_id + ".pcd"
                lidar_file_path = root + config_yaml_frame + "_lidar0.pcd"
                yaml_file = root + config_yaml_frame + ".yaml"
                output_dir =  "/home/carma/dg/results/radar2lidar_2d/" + config_yaml + "/" + sensor_id
                project_save_single_frame(radar_file_path, lidar_file_path, yaml_file, output_dir, frame,
                                        radar_index=sensor_id, lidar_index=0)