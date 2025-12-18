# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import numpy as np
from PIL import Image
import open3d as o3d
from matplotlib import cm

from datetime import datetime

from tools.carla_dataset_utils.bbx_projection import decode_yaml
from tools.carla_dataset_utils.box_utils import decode_wh, get_K
from PIL import Image, ImageDraw, ImageFont

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

try:
    import carla  # type: ignore
    _CARLA_AVAILABLE = True
except ImportError:
    carla = None  # type: ignore
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
    
    Matches CARLA's coordinate system and rotation order (Roll, Pitch, Yaw).
    
    :param x: Translation in X.
    :param y: Translation in Y.
    :param z: Translation in Z.
    :param roll: Rotation about X-axis (degrees).
    :param yaw: Rotation about Z-axis (degrees).
    :param pitch: Rotation about Y-axis (degrees).
    :return: 4x4 NumPy array representing the homogeneous transformation matrix.
    """
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.eye(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix



def process_jpeg_to_array0(file_path):
    """
    Reads a JPEG file and converts it into a numpy array with the desired structure.

    :param file_path: Path to the JPEG image file.
    :return: Numpy array of shape (height, width, 3), with channels reversed to BGR.
    """
    # Open the image using PIL
    image = Image.open(file_path)

    # Convert the image to a numpy array (shape: (height, width, 3) in RGB)
    im_array = np.array(image, dtype=np.uint8)

    # Ensure the image has 4 channels (RGBA), add an alpha channel if not present
    if im_array.shape[-1] == 3:
        im_array = np.concatenate([im_array, 255 * np.ones((*im_array.shape[:2], 1), dtype=np.uint8)], axis=-1)

    # Reshape and retain only the first 3 channels, reversing them to BGR order
    im_array = im_array[:, :, :3][:, :, ::-1]

    return im_array

def process_jpeg_to_array(file_path):
    """
    Reads a JPEG file and converts it into a numpy array with the correct RGB channel order.

    :param file_path: Path to the JPEG image file.
    :return: Numpy array of shape (height, width, 3) in correct RGB order.
    """
    # Open the image using PIL
    image = Image.open(file_path)

    # Convert the image to a numpy array (shape: (height, width, 3) in RGB)
    im_array = np.array(image, dtype=np.uint8)

    # Ensure the array has 3 channels (RGB). If there are 4 channels (RGBA), drop the alpha channel.
    if im_array.shape[-1] == 4:
        im_array = im_array[:, :, :3]  # Retain only the RGB channels

    # Return the array in its original RGB order without reversing channels
    return im_array

def process_pcd_to_array(file_path):
    """
    Reads a PCD file and converts it into a numpy array of shape (N, 4), 
    where each row represents (x, y, z, intensity).

    :param file_path: Path to the PCD file.
    :return: Numpy array of shape (N, 4).
    """
    # Read the PCD file
    pcd = o3d.io.read_point_cloud(file_path)

    # Extract point cloud data (x, y, z)
    points = np.asarray(pcd.points, dtype=np.float32)

    # Check if intensity or additional fields exist (mock for demonstration)
    # If no intensity, create a default column of zeros
    if hasattr(pcd, 'intensities') and pcd.intensities:
        intensities = np.asarray(pcd.intensities, dtype=np.float32).reshape(-1, 1)
    else:
        intensities = np.zeros((points.shape[0], 1), dtype=np.float32)

    # Combine points (x, y, z) with intensity into a single array of shape (N, 4)
    p_cloud = np.hstack((points, intensities))

    return p_cloud

def project_lidar_to_camera0(image_w, image_h, im_array, p_cloud, K, 
                            lidar_2_world, world_2_camera, output_dir):
    # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
    # focus on the 3D points.
    intensity = np.array(p_cloud[:, 3])

    # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
    local_lidar_points = np.array(p_cloud[:, :3]).T

    # Add an extra 1.0 at the end of each 3d point so it becomes of
    # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
    local_lidar_points = np.r_[
        local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

    # This (4, 4) matrix transforms the points from lidar space to world space.
    # lidar_2_world = lidar.get_transform().get_matrix() # ! need to replace

    # Transform the points from lidar space to world space.
    world_points = np.dot(lidar_2_world, local_lidar_points)

    # This (4, 4) matrix transforms the points from world to sensor coordinates.
    #　world_2_camera = np.array(camera.get_transform().get_inverse_matrix()) #! need to replace 

    # Transform the points from world space to camera space.
    sensor_points = np.dot(world_2_camera, world_points)

    # New we must change from UE4's coordinate system to an "standard"
    # camera coordinate system (the same used by OpenCV):

    # ^ z                       . z
    # |                        /
    # |              to:      +-------> x
    # | . x                   |
    # |/                      |
    # +-------> y             v y

    # This can be achieved by multiplying by the following matrix:
    # [[ 0,  1,  0 ],
    #  [ 0,  0, -1 ],
    #  [ 1,  0,  0 ]]

    # Or, in this case, is the same as swapping:
    # (x, y ,z) -> (y, -z, x)
    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2] * -1,
        sensor_points[0]])

    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, point_in_camera_coords)

    # Remember to normalize the x, y values by the 3rd value.
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]])

    # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
    # contains all the y values of our points. In order to properly
    # visualize everything on a screen, the points that are out of the screen
    # must be discarted, the same with points behind the camera projection plane.
    points_2d = points_2d.T
    intensity = intensity.T
    points_in_canvas_mask = \
        (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
        (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
        (points_2d[:, 2] > 0.0)
    points_2d = points_2d[points_in_canvas_mask]
    intensity = intensity[points_in_canvas_mask]

    # Extract the screen coords (uv) as integers.
    u_coord = points_2d[:, 0].astype(int)
    v_coord = points_2d[:, 1].astype(int)

    # Since at the time of the creation of this script, the intensity function
    # is returning high values, these are adjusted to be nicely visualized.
    intensity = 4 * intensity - 3
    color_map = np.array([
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(int).T

    dot_extent = 1 
    if dot_extent <= 0:
        # Draw the 2d points on the image as a single pixel using numpy.
        im_array[v_coord, u_coord] = color_map
    else:
        # Draw the 2d points on the image as squares of extent args.dot_extent.
        for i in range(len(points_2d)):
            # I'm not a NumPy expert and I don't know how to set bigger dots
            # without using this loop, so if anyone has a better solution,
            # make sure to update this script. Meanwhile, it's fast enough :)
            im_array[
                v_coord[i]- dot_extent : v_coord[i]+ dot_extent,
                u_coord[i]- dot_extent : u_coord[i]+ dot_extent] = color_map[i]

    # Save the image using Pillow module.
    image = Image.fromarray(im_array)

    # create output_dir if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{output_dir}/{timestamp}.png"

    image.save(file_name)
    print("0: saved image to:", file_name)

import matplotlib.pyplot as plt  # Import for color mapping

def project_lidar_to_camera1(image_w, image_h, im_array, p_cloud, K, 
                             lidar_2_world, world_2_camera, output_dir,image_txt):
    """
    Projects LiDAR point cloud data onto the camera image plane.

    :param image_w: Width of the camera image.
    :param image_h: Height of the camera image.
    :param im_array: The numpy array representing the image.
    :param p_cloud: The LiDAR point cloud data as a numpy array (N, 4).
    :param K: Intrinsic matrix of the camera.
    :param lidar_2_world: 4x4 transformation matrix from LiDAR to world coordinates.
    :param world_2_camera: 4x4 transformation matrix from world to camera coordinates.
    :param output_dir: Directory to save the resulting image.
    """
    # Extract the local coordinates of the point cloud (3, N)
    local_lidar_points = np.array(p_cloud[:, :3]).T  # x, y, z in the LiDAR frame

    # Extract z-values (height) for color mapping
    z_values = local_lidar_points[2, :]  # z-values in LiDAR coordinates

    # Convert point cloud to homogeneous coordinates (4, N)
    local_lidar_points_hom = np.r_[
        local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

    # Transform points from the LiDAR coordinate system to world coordinates
    world_points = np.dot(lidar_2_world, local_lidar_points_hom)

    # Transform points from world coordinates to camera coordinates
    sensor_points = np.dot(world_2_camera, world_points)

    # Adjust the coordinate system from UE4 to standard camera coordinates
    point_in_camera_coords = np.array([
        sensor_points[1], 
        sensor_points[2] * -1, 
        sensor_points[0]])

    # Project 3D points onto the 2D image plane using the intrinsic matrix
    points_2d = np.dot(K, point_in_camera_coords)

    # Normalize x and y coordinates
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]])

    # Filter points within the image bounds and in front of the camera
    points_2d = points_2d.T
    z_values = z_values.T
    points_in_canvas_mask = \
        (points_2d[:, 0] >= 0.0) & (points_2d[:, 0] < image_w) & \
        (points_2d[:, 1] >= 0.0) & (points_2d[:, 1] < image_h) & \
        (points_2d[:, 2] > 0.0)
    points_2d = points_2d[points_in_canvas_mask]
    z_values = z_values[points_in_canvas_mask]

    # Extract pixel coordinates and convert to integers
    u_coord = points_2d[:, 0].astype(int)
    v_coord = points_2d[:, 1].astype(int)

    # Normalize z-values to the range [0, 1]
    min_z = np.min(z_values)
    max_z = np.max(z_values)
    norm_z_values = (z_values - min_z) / (max_z - min_z)

    # Map normalized z-values to RGB colors using a colormap
    cmap = plt.get_cmap('jet')  # Choose a colormap, e.g., 'jet'
    colors = cmap(norm_z_values)[:, :3]  # Extract RGB values

    # Convert color values to integers in the range [0, 255]
    color_map = (colors * 255).astype(np.uint8)

    dot_extent = 1  # Size of the dot for each point
    if dot_extent <= 0:
        # Draw single-pixel points
        im_array[v_coord, u_coord] = color_map
    else:
        # Draw square dots of size `dot_extent`
        for i in range(len(points_2d)):
            # Ensure indices are within bounds
            x_min = max(0, u_coord[i] - dot_extent)
            x_max = min(image_w, u_coord[i] + dot_extent)
            y_min = max(0, v_coord[i] - dot_extent)
            y_max = min(image_h, v_coord[i] + dot_extent)
            im_array[y_min:y_max, x_min:x_max] = color_map[i]

    # Save the resulting image
    image = Image.fromarray(im_array)

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{output_dir}/{timestamp}.png"

    # put a txt on the left cornor of the saved image, like image_txt = "100"
    # Add text to the top-left corner of the saved image, e.g., image_txt = "100"
    # image_txt = "100"  # The text content to add
    draw = ImageDraw.Draw(image)

    # Optional: Specify font and size
    # Ensure that the font file (e.g., "arial.ttf") is available in your environment
    # font = ImageFont.truetype("arial.ttf", 20)

    # Specify the position for the text (top-left corner), adjust as needed
    position = (10, 10)  # (x, y) coordinates

    # Specify the text color, e.g., white
    text_color = (255, 255, 255)  # RGB tuple

    # Draw the text on the image
    # If using a custom font, add the 'font=font' parameter
    draw.text(position, image_txt, fill=text_color)  # , font=font)

    image.save(file_name)
    print("1: saved image to:", file_name)


def project_lidar_to_camera2(image_w, image_h, im_array, p_cloud, K, 
                             lidar_2_world, world_2_camera, output_dir, image_txt):
    """
    Projects LiDAR point cloud data onto the camera image plane.

    :param image_w: Width of the image.
    :param image_h: Height of the image.
    :param im_array: The numpy array representing the image.
    :param p_cloud: LiDAR point cloud data as a numpy array with shape (N, 4).
    :param K: Intrinsic matrix of the camera.
    :param lidar_2_world: 4x4 transformation matrix from LiDAR to world coordinates.
    :param world_2_camera: 4x4 transformation matrix from world to camera coordinates.
    :param output_dir: Directory to save the resulting image.
    :param image_txt: Text to add to the image.
    """
    # Extract the local coordinates of the point cloud (3, N)
    local_lidar_points = np.array(p_cloud[:, :3]).T  # x, y, z in the LiDAR frame
    # print("shape of local_lidar_points:",local_lidar_points.shape)

    # Extract x-values for color mapping
    x_values = local_lidar_points[0, :]  # x-values in LiDAR coordinates

    # Convert point cloud to homogeneous coordinates (4, N)
    local_lidar_points_hom = np.r_[
        local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

    # Transform points from the LiDAR coordinate system to world coordinates
    world_points = np.dot(lidar_2_world, local_lidar_points_hom)

    # Transform points from world coordinates to camera coordinates
    sensor_points = np.dot(world_2_camera, world_points)

    # Adjust the coordinate system from UE4 to standard camera coordinates
    point_in_camera_coords = np.array([
        sensor_points[1], 
        sensor_points[2] * -1, 
        sensor_points[0]])

    # Project 3D points onto the 2D image plane using the intrinsic matrix
    points_2d = np.dot(K, point_in_camera_coords)

    # Normalize x and y coordinates
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]])

    # Filter points within the image bounds and in front of the camera
    points_2d = points_2d.T
    x_values = x_values.T
    points_in_canvas_mask = (
        (points_2d[:, 0] >= 0.0) & (points_2d[:, 0] < image_w) &
        (points_2d[:, 1] >= 0.0) & (points_2d[:, 1] < image_h) &
        (points_2d[:, 2] > 0.0)
    )
    points_2d = points_2d[points_in_canvas_mask]
    x_values = x_values[points_in_canvas_mask]

    # Extract pixel coordinates and convert to integers
    u_coord = points_2d[:, 0].astype(int)
    v_coord = points_2d[:, 1].astype(int)

    # Normalize x-values to the range [0, 1]
    min_x = np.min(x_values)
    max_x = np.max(x_values)
    norm_x_values = (x_values - min_x) / (max_x - min_x)

    # Map normalized x-values to RGB colors using a colormap
    cmap = plt.get_cmap('jet')  # Choose a colormap, e.g., 'jet'
    colors = cmap(norm_x_values)[:, :3]  # Extract RGB values

    # Convert color values to integers in the range [0, 255]
    color_map = (colors * 255).astype(np.uint8)

    dot_extent = 1  # Size of the dot for each point
    if dot_extent <= 0:
        # Draw single-pixel points
        im_array[v_coord, u_coord] = color_map
    else:
        # Draw square dots of size `dot_extent`
        for i in range(len(points_2d)):
            # Ensure indices are within image bounds
            x_min = max(0, u_coord[i] - dot_extent)
            x_max = min(image_w, u_coord[i] + dot_extent)
            y_min = max(0, v_coord[i] - dot_extent)
            y_max = min(image_h, v_coord[i] + dot_extent)
            im_array[y_min:y_max, x_min:x_max] = color_map[i]

    # Convert the numpy array back to a PIL Image
    image = Image.fromarray(im_array)

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{output_dir}/{timestamp}.png"

    # Add text to the top-left corner of the saved image
    draw = ImageDraw.Draw(image)
    position = (10, 10)  # Text position (x, y)
    text_color = (255, 255, 255)  # Text color (white)
    draw.text(position, image_txt, fill=text_color)

    # Save the resulting image
    image.save(file_name)
    print("2: saved image to:", file_name)

def project_save_single_frame(img_file_path,pcd_file_path,yaml_file,output_dir,
                              cam_index,lidar_index):
    """
    Projects LiDAR point cloud data onto a 2D camera image and saves the resulting visualization.

    This function reads image and LiDAR data from the provided file paths, decodes the camera and 
    LiDAR parameters from a YAML file, computes the necessary transformations, and projects the 
    3D LiDAR points into the camera's 2D image plane. The final annotated image is saved to the 
    specified output directory.

    Args:
        img_file_path (str): Path to the input image file.
            Example: "./data_dumping/example/2024_12_01_18_14_28/125/camera1_000045.png"
        pcd_file_path (str): Path to the input LiDAR point cloud (.pcd) file.
            Example: "./data_dumping/example/2024_12_01_18_14_28/125/lidar1_000045.pcd"
        yaml_file (str): Path to the YAML file containing sensor and object parameters.
            Example: "./data_dumping/example/2024_12_01_18_14_28/125/000045.yaml"
        output_dir (str): Path to the directory where the resulting image will be saved.
            Example: "./data_dumping/example/2024_12_01_18_14_28/new_1"

    Workflow:
        1. Decode the YAML file to extract camera and LiDAR poses, intrinsic parameters, and object data.
        2. Compute the camera's intrinsic matrix and image dimensions.
        3. Process the input image into a NumPy array.
        4. Convert the LiDAR point cloud into a format suitable for projection.
        5. Compute transformation matrices:
           - LiDAR-to-world transformation.
           - World-to-camera transformation.
           - Combined LiDAR-to-camera transformation.
        6. Project the 3D LiDAR points into the camera's 2D image plane.
        7. Save the resulting annotated image to the output directory.

    Dependencies:
        - `decode_yaml`: Decodes the YAML file to retrieve sensor and object parameters.
        - `create_transformation`: Generates transformation matrices for the LiDAR and camera.
        - `process_jpeg_to_array`: Converts the image into a NumPy array.
        - `process_pcd_to_array`: Converts the LiDAR point cloud into a NumPy array.
        - `project_lidar_to_camera2`: Handles the projection of 3D LiDAR points into the image plane.

    Returns:
        None: The function saves the output directly to the specified directory.

    Raises:
        FileNotFoundError: If any of the input file paths do not exist.
        ValueError: If the input data is invalid or incomplete.

    Example:
        img_file_path = "./data_dumping/example/2024_12_01_18_14_28/125/camera1_000045.png"
        pcd_file_path = "./data_dumping/example/2024_12_01_18_14_28/125/lidar1_000045.pcd"
        yaml_file = "./data_dumping/example/2024_12_01_18_14_28/125/000045.yaml"
        output_dir = "./data_dumping/example/2024_12_01_18_14_28/new_1"
        cam_index = 1 
        lidar_index = 1         
        project_save_single_frame(img_file_path, pcd_file_path, yaml_file, output_dir,cam_index,lidar_index)
    """

    lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict = decode_yaml(yaml_file)
    # K = get_K(image_w, image_h, fov)

    # print("K:", K)


    camera_param = camera_list[cam_index]
    lidar_cords = lidar_pose_list[lidar_index]

    cam_cords = camera_param['cords']
    camera_intrinsic = camera_param['intrinsic']

    image_w, image_h = decode_wh(camera_intrinsic)

    print("cam_cords:", cam_cords)
    _,world_2_camera = create_transformation(cam_cords[0], cam_cords[1], cam_cords[2], \
                                              cam_cords[3], cam_cords[4], cam_cords[5]) 

    print("lidar_cords:", lidar_cords)    

    im_array = process_jpeg_to_array(img_file_path)
    p_cloud = process_pcd_to_array(pcd_file_path)

    print("lidar_cords_new:", lidar_cords)
    lidar_2_world,world_2_lidar = create_transformation(lidar_cords[0], lidar_cords[1], lidar_cords[2], \
                                            lidar_cords[3], lidar_cords[4], lidar_cords[5])
    # print("---")
    # print("lidar_2_world_old:",lidar_2_world_old)
    # print("lidar_2_world:",lidar_2_world)
    # print("world_2_lidar:",world_2_lidar)
    # print("---")

    lidar2camera = np.dot(world_2_camera, lidar_2_world)
    # print("lidar2camera:",lidar2camera)

    
    project_lidar_to_camera2(image_w, image_h, im_array, p_cloud, camera_intrinsic, 
                                lidar_2_world, world_2_camera, output_dir,image_txt=str(lidar_cords[4]))


# =============================================================================
# TEST CASES
# =============================================================================
def test1():
    # Example usage:
    file_path = "data_dumping/example/2024_11_30_16_15_46/125/camera0_000045.png"
    im_array = process_jpeg_to_array(file_path)
    print("Processed image shape:", im_array.shape)

def test2():
    # Example usage:
    file_path = "data_dumping/example/2024_11_30_16_15_46/125/lidar0_000037.pcd"
    p_cloud = process_pcd_to_array(file_path)
    print("Processed point cloud shape:", p_cloud.shape)

def test3():
    yaml_file = "data_dumping/example/2024_11_30_16_15_46/125/000045.yaml"
    lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict = decode_yaml(yaml_file)
    print(camera_list[0])

def test4():
    yaml_file = "data_dumping/example/2024_11_30_16_15_46/125/000045.yaml"
    lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict = decode_yaml(yaml_file)
    cam_cords = camera_list[0]['cords']
    x = cam_cords[0]
    y = cam_cords[1]
    z = cam_cords[2]
    roll = cam_cords[3]
    yaw = cam_cords[4]
    pitch = cam_cords[5]
    came_transform = create_transformation(x, y, z, roll, yaw, pitch) 
    came_transform_hom = to_homogeneous_matrix(x, y, z, roll, yaw, pitch)   
    print("Camera transformation:", came_transform.get_matrix()) # This one is correct in carla
    print("Camera transformation (homogeneous):", came_transform_hom)

def test_project_save_single_frame_based_on_diff_yaw():
    image_w = 1920 
    image_h = 1080 
    fov = 80
    img_file_path = "./data_dumping/example/2024_12_01_18_14_28/125/camera1_000045.png"
    pcd_file_path = "./data_dumping/example/2024_12_01_18_14_28/125/lidar1_000045.pcd"
    yaml_file = "./data_dumping/example/2024_12_01_18_14_28/125/000045.yaml"

    lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict = decode_yaml(yaml_file)
    K = get_K(image_w, image_h, fov)
    print("K:", K)

    cam_cords = camera_list[1]['cords']
    print("cam_cords:", cam_cords)
    _,world_2_camera = create_transformation(cam_cords[0], cam_cords[1], cam_cords[2], \
                                              cam_cords[3], cam_cords[4], cam_cords[5]) 
    lidar_cords = lidar_pose_list[1]
    print("lidar_cords:", lidar_cords)
    # lidar_2_world_old,world_2_lidar_old = create_transformation(lidar_cords[0], lidar_cords[1], lidar_cords[2], \
    #                                           lidar_cords[3], lidar_cords[4], lidar_cords[5])
    
    # create angle from -180 to 180,step 5 
    angles = np.arange(-180, 185, 5)
    for i in range(len(angles)):
        im_array = process_jpeg_to_array(img_file_path)
        p_cloud = process_pcd_to_array(pcd_file_path)
        print("angles[i]:",angles[i])
        
        lidar_cords[4] = angles[i] * 1.0

        print("lidar_cords_new:", lidar_cords)
        lidar_2_world,world_2_lidar = create_transformation(lidar_cords[0], lidar_cords[1], lidar_cords[2], \
                                                lidar_cords[3], lidar_cords[4], lidar_cords[5])
        # print("---")
        # print("lidar_2_world_old:",lidar_2_world_old)
        # print("lidar_2_world:",lidar_2_world)
        # print("world_2_lidar:",world_2_lidar)
        # print("---")

        lidar2camera = np.dot(world_2_camera, lidar_2_world)
        # print("lidar2camera:",lidar2camera)

        output_dir = "./data_dumping/example/2024_12_01_18_14_28/new"
        project_lidar_to_camera2(image_w, image_h, im_array, p_cloud, K, 
                                    lidar_2_world, world_2_camera, output_dir,image_txt=str(lidar_cords[4]))

def test5():
    # img_file_path = "/Users/zhaoliang/Documents/zhz03/github/v2x-real-example/2023-04-03-18-15-32_9_0/1/000030_cam3.jpeg"
    # pcd_file_path = "/Users/zhaoliang/Documents/zhz03/github/v2x-real-example/2023-04-03-18-15-32_9_0/1/000030.bin"
    # yaml_file = "/Users/zhaoliang/Documents/zhz03/github/v2x-real-example/2023-04-03-18-15-32_9_0/1/000030.yaml"
    # output_dir = "/Users/zhaoliang/Documents/zhz03/github/v2x-real-example/2023-04-03-18-15-32_9_0/example"

    img_file_path = "/home/zhaoliang/zzl/dataset_tool/data_examples/m2i_radar_dataset/000032_camera1.png"
    pcd_file_path = "/home/zhaoliang/zzl/dataset_tool/data_examples/m2i_radar_dataset/000032_lidar0.pcd"
    yaml_file = "/home/zhaoliang/zzl/dataset_tool/data_examples/m2i_radar_dataset/000032.yaml"
    output_dir = "/home/zhaoliang/zzl/dataset_tool/data_examples/m2i_radar_dataset/check_data1"
    project_save_single_frame(img_file_path, pcd_file_path, yaml_file, output_dir,
                              cam_index=1,lidar_index=0)

if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    # test4()
    # project_save_single_frame()
    test5()