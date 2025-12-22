# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import math
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
import common_utils #from tools.carla_dataset_utils 

def get_K(image_w, image_h, fov):
    # Build the K projection matrix:
    # K = [[Fx,  0, image_w/2],
    #      [ 0, Fy, image_h/2],
    #      [ 0,  0,         1]]
    """
    Get the camera intrinsic matrix K from the image width, image height, and field of view (fov).
    For example:     
        image_w = 1920
        image_h = 1080
        fov = 120
    """

    focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

    # In this case Fx and Fy are the same since the pixel aspect
    # ratio is 1
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = image_w / 2.0
    K[1, 2] = image_h / 2.0
    return K


def decode_wh(K):
    """
    Extracts the image width and height from the camera intrinsic matrix K.

    The intrinsic matrix K is assumed to have the following structure:
        K = [[Fx,  0, image_w / 2],
             [ 0, Fy, image_h / 2],
             [ 0,  0,          1]]

    :param K: A 3x3 camera intrinsic matrix (numpy array or list of lists).
    :return: A tuple containing (image_w, image_h) as integers.
    """
    # Convert K to a numpy array if it's not already
    K = np.array(K)
    
    # Validate the shape of K
    if K.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix K must be of shape (3, 3), but got {K.shape}")
    
    # Extract the (0, 2) and (1, 2) elements which correspond to image_w/2 and image_h/2 respectively
    image_w_half = K[0, 2]
    image_h_half = K[1, 2]
    
    # Compute the full image width and height
    image_w = 2 * image_w_half
    image_h = 2 * image_h_half
    
    # Optionally, round the values to the nearest integer
    image_w = int(round(image_w))
    image_h = int(round(image_h))
    
    return image_w, image_h

def convert_to_box(features):
    # example: [ 5.96192197 45.5479153  -4.09618628  4.84935665  2.88018813  1.85168338 -1.57597751  1.90349856 -6.3434773   1.        ]
    position = {
    "x": -features[1],  
    "y": features[0],  
    "z": features[2]   
    }
    scale = {
    "x": features[3],  
    "y": features[4],  
    "z": features[5]      
    }
    # features[6] is in [-pi,pi]
    # z_degree = features[6] * 180 / math.pi # degrees
    rotation = {
    "x": 0.0,  
    "y": 0.0,  
    "z": features[6] - 90 # [0,360]   # [-180, 180]   
    }

    return position, scale, rotation

def convert_carla_data_to_box(angle,extent,location):
    # example: [ 5.96192197 45.5479153  -4.09618628  4.84935665  2.88018813  1.85168338 -1.57597751  1.90349856 -6.3434773   1.        ]
    position = {
    "x": location[0],  
    "y": location[1],  
    "z": location[2]   
    }
    scale = {
    "x": extent[0] * 2,  
    "y": extent[1] * 2 ,  
    "z": extent[2] * 2      
    }
    # features[6] is in [-pi,pi]
    # z_degree = features[6] * 180 / math.pi # degrees
    rotation = {
    "x": angle[0],  
    "y": angle[2],  
    "z": angle[1] # [0,360]   # [-180, 180]   
    }

    return position, scale, rotation


def create_rotated_box(position, scale, rotation,color=(0, 1, 0),offset_deg=0):
    """Create a 3D rotated box represented as a LineSet in 3D space.

    This function generates a 3D rotated box with specified position, scale, and rotation, and
    returns it as an Open3D LineSet, which can be visualized as wireframe box geometry.

    Args:
        position (dict): A dictionary containing the X, Y, and Z coordinates of the box's center.
        scale (dict): A dictionary containing the scaling factors for the box along the X, Y, and Z axes.
        rotation (dict): A dictionary containing the rotation angles in degrees around the X, Y, and Z axes.
        color (tuple, optional): The RGB color tuple for the box's lines. Defaults to (0, 1, 0) (green).

    Returns:
        o3d.geometry.LineSet: An Open3D LineSet representing the rotated box.

    Example:
        >>> position = {"x": 1.0, "y": 2.0, "z": 3.0}
        >>> scale = {"x": 2.0, "y": 1.0, "z": 0.5}
        >>> rotation = {"x": 45.0, "y": 30.0, "z": 60.0}
        >>> box = create_rotated_box(position, scale, rotation, color=(1, 0, 0))  # Create a red rotated box
        >>> o3d.visualization.draw_geometries([box])  # Visualize the box using Open3D
    """
    # Define vertices of the box
    vertices = np.array([[-0.5, -0.5, -0.5],
                         [ 0.5, -0.5, -0.5],
                         [ 0.5,  0.5, -0.5],
                         [-0.5,  0.5, -0.5],
                         [-0.5, -0.5,  0.5],
                         [ 0.5, -0.5,  0.5],
                         [ 0.5,  0.5,  0.5],
                         [-0.5,  0.5,  0.5]])

    # Scale the vertices
    scaled_vertices = vertices * np.array([scale["x"], scale["y"], scale["z"]])

    # Apply rotation
    # rotation_matrix = Rotation.from_euler('xyz', [math.degrees(rotation["x"]), math.degrees(rotation["y"]), math.degrees(rotation["z"])+ offset_deg], degrees=True).as_matrix()
    rotation_matrix = Rotation.from_euler('xyz', [rotation["x"], rotation["y"], rotation["z"]], degrees=True).as_matrix()
    rotated_vertices = np.dot(scaled_vertices, rotation_matrix.T)
    
    # Apply translation
    translated_vertices = rotated_vertices + np.array([position["x"], position["y"], position["z"] + scale["z"]/2])

    # Apply global rotation 
    # rotation_matrix = Rotation.from_euler('xyz', [math.degrees(rotation["x"]), math.degrees(rotation["y"]), math.degrees(rotation["z"])], degrees=True).as_matrix()
    # rotation_matrix = Rotation.from_euler('xyz', [rotation["x"], rotation["y"], rotation["z"]], degrees=True).as_matrix()
    # rotated_vertices = np.dot(translated_vertices, rotation_matrix.T)

    # rotate rotated_vertices at its center for 45 degree

    # Define edges of the box
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    # Define heading arrow (we'll add an arrow from the center to one end of the box along the x-axis)
    arrow_length = scale["x"]
    arrow_start = np.array([position["x"], position["y"], position["z"]])
    arrow_end = np.dot([arrow_length, 0, 0], rotation_matrix.T) + arrow_start
    edges.append((8, 9))
    translated_vertices = np.vstack((translated_vertices, arrow_start, arrow_end))

    # Use the same color for all lines
    colors = [list(color) for _ in range(len(edges))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(translated_vertices)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def create_rotated_box_points(position, scale, rotation,color=(0, 1, 0),offset_deg=0):
    # Define vertices of the box
    vertices = np.array([[-0.5, -0.5, -0.5],
                         [ 0.5, -0.5, -0.5],
                         [ 0.5,  0.5, -0.5],
                         [-0.5,  0.5, -0.5],
                         [-0.5, -0.5,  0.5],
                         [ 0.5, -0.5,  0.5],
                         [ 0.5,  0.5,  0.5],
                         [-0.5,  0.5,  0.5]])

    # Scale the vertices
    scaled_vertices = vertices * np.array([scale["x"], scale["y"], scale["z"]])

    # Apply rotation
    # rotation_matrix = Rotation.from_euler('xyz', [math.degrees(rotation["x"]), math.degrees(rotation["y"]), math.degrees(rotation["z"])+ offset_deg], degrees=True).as_matrix()
    rotation_matrix = Rotation.from_euler('xyz', [rotation["x"], rotation["y"], rotation["z"]], degrees=True).as_matrix()
    rotated_vertices = np.dot(scaled_vertices, rotation_matrix.T)
    
    # Apply translation
    translated_vertices = rotated_vertices + np.array([position["x"], position["y"], position["z"] + scale["z"]/2])

    # Apply global rotation 
    # rotation_matrix = Rotation.from_euler('xyz', [math.degrees(rotation["x"]), math.degrees(rotation["y"]), math.degrees(rotation["z"])], degrees=True).as_matrix()
    # rotation_matrix = Rotation.from_euler('xyz', [rotation["x"], rotation["y"], rotation["z"]], degrees=True).as_matrix()
    # rotated_vertices = np.dot(translated_vertices, rotation_matrix.T)

    # rotate rotated_vertices at its center for 45 degree

    # Define edges of the box
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    # Define heading arrow (we'll add an arrow from the center to one end of the box along the x-axis)
    arrow_length = scale["x"]
    arrow_start = np.array([position["x"], position["y"], position["z"]])
    arrow_end = np.dot([arrow_length, 0, 0], rotation_matrix.T) + arrow_start
    edges.append((8, 9))
    translated_vertices = np.vstack((translated_vertices, arrow_start, arrow_end))

    # Use the same color for all lines
    colors = [list(color) for _ in range(len(edges))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(translated_vertices)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return translated_vertices, edges, colors

def proj_points_2_img(im_array, vertices, edges, points_2_world, world_2_img,
                      camera_intrinsics, color=(0, 255, 0)):
    # The shape of vertices is (3,N)

    # # Conver 3d points to homogeneous coordinates (4,N)
    vertices_points_hom = np.r_[
        vertices, [np.ones(vertices.shape[1])]]

    world_points = np.dot(points_2_world, vertices_points_hom)
    sensor_points = np.dot(world_2_img, world_points)

    # Adjust the coordinate system from UE4 to standard camera coordinates
    # UE4/Carla uses: X=forward, Y=right, Z=up
    # Camera coordinates need: X=right, Y=down, Z=forward
    # So we remap: X_cam = Y_carla, Y_cam = -Z_carla, Z_cam = X_carla
    point_in_camera_coords = np.array([
        sensor_points[1],        # X_camera = Y_carla (right)
        -sensor_points[2],       # Y_camera = -Z_carla (down)
        sensor_points[0]])       # Z_camera = X_carla (forward)

    points_2d = np.dot(camera_intrinsics, point_in_camera_coords)

    # Normalize x and y coordinates
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]])

    # Transpose points_2d for easier indexing
    points_2d = points_2d.T
    image_h, image_w = decode_wh(camera_intrinsics)

    # Convert image array to PIL Image for drawing
    if isinstance(im_array, np.ndarray):
        image = Image.fromarray(im_array)
    else:
        image = im_array

    draw = ImageDraw.Draw(image)

    # Extract pixel coordinates
    u_coord = points_2d[:, 0]
    v_coord = points_2d[:, 1]

    # Loop over edges and draw lines
    for edge in edges:
        idx0, idx1 = edge[0], edge[1]

        # Skip invalid endpoints 
        if not (valid_mask[idx0] and valid_mask[idx1]):
            continue

        # Get the coordinates of the vertices
        u0, v0 = u_coord[idx0], v_coord[idx0]
        u1, v1 = u_coord[idx1], v_coord[idx1]

        # Optionally, check if the points are in front of the camera
        if points_2d[idx0, 2] <= 0 or points_2d[idx1, 2] <= 0:
            continue  # Skip lines with points behind the camera

        # Optionally, check if both points are within image bounds
        if not ((0 <= u0 < image_w and 0 <= v0 < image_h) or (0 <= u1 < image_w and 0 <= v1 < image_h)):
            continue  # Skip lines completely outside the image

        # Draw the line
        draw.line([(u0, v0), (u1, v1)], fill=color, width=2)

    # If needed, convert the image back to numpy array
    im_array_result = np.array(image)

    return im_array_result

def save_img(im_array_result, save_path):
    """
    Saves the image array to the specified file path.

    :param im_array_result: The image array to be saved (numpy array).
    :param save_path: The file path where the image will be saved.
    """
    # Convert the numpy array to a PIL Image
    image = Image.fromarray(im_array_result)

    # Save the image to the specified path
    image.save(save_path)
    print(f"Image saved to {save_path}")

def boxes_to_corners2d(boxes3d, order):
    """
      0 -------- 1
      |          |
      |          |
      |          |
      3 -------- 2
    Parameters
    __________
    boxes3d: np.ndarray or torch.Tensor
        (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners2d: np.ndarray or torch.Tensor
        (N, 4, 3), the 4 corners of the bounding box.

    """
    corners3d = boxes_to_corners_3d(boxes3d, order)
    corners2d = corners3d[:, :4, :]
    return corners2d


def boxes2d_to_corners2d(boxes2d, order="lwh"):
    """
      0 -------- 1
      |          |
      |          |
      |          |
      3 -------- 2
    Parameters
    __________
    boxes2d: np.ndarray or torch.Tensor
        (..., 5) [x, y, dx, dy, heading], (x, y) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners2d: np.ndarray or torch.Tensor
        (..., 4, 2), the 4 corners of the bounding box.

    """
    assert order == "lwh", \
        "boxes2d_to_corners_2d only supports lwh order for now."
    boxes2d, is_numpy = common_utils.check_numpy_to_torch(boxes2d)
    template = boxes2d.new_tensor((
        [1, -1], [1, 1], [-1, 1], [-1, -1]
    )) / 2
    input_shape = boxes2d.shape
    boxes2d = boxes2d.view(-1, 5)
    corners2d = boxes2d[:, None, 2:4].repeat(1, 4, 1) * template[None, :, :]
    corners2d = common_utils.rotate_points_along_z_2d(corners2d.view(-1, 2),
                                                      boxes2d[:,
                                                      4].repeat_interleave(
                                                          4)).view(-1, 4,
                                                                   2)
    corners2d += boxes2d[:, None, 0:2]
    corners2d = corners2d.view(*(input_shape[:-1]), 4, 2)
    return corners2d


def boxes_to_corners_3d(boxes3d, order):
    """
        4 -------- 5
       /|         /|
      7 -------- 6 .
      | |        | |
      . 0 -------- 1
      |/         |/
      3 -------- 2
    Parameters
    __________
    boxes3d: np.ndarray or torch.Tensor
        (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center.

    order : str
        'lwh' or 'hwl'

    Returns:
        corners3d: np.ndarray or torch.Tensor
        (N, 8, 3), the 8 corners of the bounding box.

    """
    # ^ z
    # |
    # |
    # | . x
    # |/
    # +-------> y

    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    boxes3d_ = boxes3d

    if order == 'hwl':
        boxes3d_ = boxes3d[:, [0, 1, 2, 5, 4, 3, 6]]

    template = boxes3d_.new_tensor((
        [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
        [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
    )) / 2

    corners3d = boxes3d_[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3),
                                                   boxes3d_[:, 6]).view(-1, 8,
                                                                        3)
    corners3d += boxes3d_[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

if __name__ == "__main__":
    pass
