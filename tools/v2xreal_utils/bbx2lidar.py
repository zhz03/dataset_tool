import numpy as np
import open3d as o3d

from tools.carla_dataset_utils.bbx_projection import decode_yaml
from tools.carla_dataset_utils.project_bbox_lidar import (
    parse_vehicle_bbox,
    transform_bbox_world_to_lidar,
    visualize_single_sample_dataloader,
)

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
    lidar_pose: [x, y, z, roll, yaw, pitch] in carla world frame (degrees)
    return: 4x4 transform from WORLD -> LIDAR in a right-handed convention
    """
    x, y, z, roll_deg, yaw_deg, pitch_deg = lidar_pose

    # Convert degrees to radians
    roll = np.deg2rad(-roll_deg)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(-pitch_deg)

    # Build forward transform T (world -> lidar)
    T = np.eye(4)
    R = euler_to_rot(roll, yaw, pitch)  # Use the updated euler_to_rot
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    # Invert to get T_lidar_world
    T_inv = np.linalg.inv(T)
    return T_inv

def load_bin_file(bin_file, return_attributes=False, dtype=np.float32):
    """
    Load a LiDAR point cloud stored in a binary (.bin) file.

    The binary file is expected to contain a flat sequence of 32-bit floats in
    either XYZI* (4 values per point) or XYZIR (5 values per point) format.

    Args:
        bin_file (str): Path to the binary LiDAR file.
        return_attributes (bool): If True, return both XYZ coordinates and any
            additional per-point attributes (e.g., intensity). If False, return
            only the XYZ coordinates. Defaults to False.
        dtype: NumPy dtype used to interpret the binary data. Defaults to
            ``np.float32``.

    Returns:
        np.ndarray or Tuple[np.ndarray, np.ndarray]:
            - If ``return_attributes`` is False: array of shape (N, 3) with XYZ coordinates.
            - If ``return_attributes`` is True: tuple ``(xyz, attributes)``, where
              ``xyz`` has shape (N, 3) and ``attributes`` has shape (N, C) for the
              remaining channels.

    Raises:
        FileNotFoundError: If the provided path does not exist.
        ValueError: If the data length cannot be reshaped into Nx4 or Nx5.
    """
    if not bin_file or not isinstance(bin_file, str):
        raise ValueError("A valid path to the LiDAR .bin file must be provided.")

    lidar_buffer = np.fromfile(bin_file, dtype=dtype)
    if lidar_buffer.size == 0:
        raise ValueError(f"LiDAR file '{bin_file}' appears to be empty.")

    if lidar_buffer.size % 5 == 0:
        lidar_data = lidar_buffer.reshape(-1, 5)
    elif lidar_buffer.size % 4 == 0:
        lidar_data = lidar_buffer.reshape(-1, 4)
    else:
        raise ValueError(
            f"Unsupported LiDAR point format in '{bin_file}': "
            f"{lidar_buffer.size} floats cannot be reshaped into Nx4 or Nx5."
        )

    xyz = lidar_data[:, :3]
    attributes = lidar_data[:, 3:] if lidar_data.shape[1] > 3 else np.empty((xyz.shape[0], 0))

    if return_attributes:
        return xyz, attributes

    if attributes.size == 0:
        intensity = np.ones((xyz.shape[0], 1), dtype=xyz.dtype)
    else:
        intensity = attributes[:, :1]
    return np.hstack([xyz, intensity])

def main(bin_file, yaml_file, lidar_index=0, visualize=True):
    """
    Load a LiDAR point cloud and its associated YAML metadata, transform all
    vehicle bounding boxes from the world frame into the LiDAR frame, and
    optionally visualize the result.

    Args:
        bin_file (str): Path to the LiDAR binary file (XYZI or XYZIR floats).
        yaml_file (str): Path to the YAML file containing poses and objects.
        lidar_index (int): Index of the LiDAR sensor to use from the YAML file.
        visualize (bool): If True, launch an Open3D window to visualize the
            LiDAR points and transformed bounding boxes.

    Returns:
        dict: Sample dictionary containing LiDAR points and box annotations
        in the LiDAR coordinate frame.
    """
    # (1) Load LiDAR points
    lidar_points = load_bin_file(bin_file)

    # (2) Parse YAML metadata
    lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict = decode_yaml(yaml_file)
    if not lidar_pose_list:
        raise ValueError(f"No LiDAR pose found in '{yaml_file}'.")
    if lidar_index >= len(lidar_pose_list):
        raise IndexError(
            f"Requested lidar_index={lidar_index}, but only {len(lidar_pose_list)} "
            f"LiDAR poses are available."
        )
    lidar_pose = lidar_pose_list[lidar_index]
    print("lidar_pose:",lidar_pose) # x,y,z,roll,yaw,pitch
    vehicles = vehicle_dict if vehicle_dict is not None else {}

    # (3) Build bounding boxes in world frame
    bboxes_world = []
    for vid, vehicle_info in vehicles.items():
        try:
            bbox = parse_vehicle_bbox(vehicle_info)
            bboxes_world.append(bbox)
        except KeyError as exc:
            print(f"Skipping vehicle '{vid}' missing field {exc}.")
    bboxes_world = (
        np.array(bboxes_world, dtype=np.float32)
        if bboxes_world else np.zeros((0, 7), dtype=np.float32)
    )

    # (4) Transform bounding boxes from world to LiDAR frame
    T_world_to_lidar = invert_pose(lidar_pose)
    bboxes_lidar = bboxes_world.copy()
    for i in range(bboxes_lidar.shape[0]):
        bboxes_lidar[i, :] = transform_bbox_world_to_lidar(
            bboxes_world[i, :], T_world_to_lidar
        )

    # (5) Assemble sample dictionary
    object_bbx_mask = np.ones((bboxes_lidar.shape[0],), dtype=bool)
    sample = {
        "ego": {
            "origin_lidar": lidar_points,
            "object_bbx_center": bboxes_lidar,
            "object_bbx_mask": object_bbx_mask,
        }
    }

    # (6) Optional visualization
    if visualize:
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd, box_lines = visualize_single_sample_dataloader(
            sample["ego"],
            o3d_pcd=vis_pcd,
            order="hwl",
            visualize=False,
            mode="constant",
        )

        red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        for lineset in box_lines:
            lineset.colors = o3d.utility.Vector3dVector(
                np.repeat(red[np.newaxis, :], len(lineset.lines), axis=0)
            )

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        visualizer.add_geometry(vis_pcd)
        for lineset in box_lines:
            visualizer.add_geometry(lineset)

        render_option = visualizer.get_render_option()
        if render_option is not None:
            render_option.line_width = 10.0

        visualizer.run()
        visualizer.destroy_window()

    return sample

def test1():
    bin_file = "/Users/zhaoliang/Documents/zhz03/github/dataset_tool/test_examples/v2x-real/vehicle_1/000030.bin"
    yaml_file = "/Users/zhaoliang/Documents/zhz03/github/dataset_tool/test_examples/v2x-real/vehicle_1/000030.yaml"
    main(bin_file, yaml_file)
    
if __name__ == "__main__":
    test1()