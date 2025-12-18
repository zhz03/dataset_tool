# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import carla
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tools.utils.yaml_utils import load_yaml
from tools.verify_dataset.img_utils import ImageLoader
from tools.verify_dataset.box_util import convert_carla_data_to_box

class ProjBBX2Cam:
    """
    Class to project 3D bounding boxes onto camera images.
    This class is used to verify the dataset by projecting 3D bounding boxes onto camera images.
    """

    def __init__(self, line_width=2):
        self.line_width = line_width
        self.cam_param = None
        self.img_path = None
        self.yaml_path = None

    def load_config(self, yaml_path, config_key="camera0"):
        if yaml_path.endswith(".yaml"):
            yaml_config = load_yaml(yaml_path)
            if config_key not in yaml_config:
                raise KeyError(f"Key '{config_key}' not found in the YAML file.")
            config_param = yaml_config[config_key]
        else:
            raise ValueError(f"Unsupported file extension: {yaml_path}")
        return config_param
        
    def decode_cam_param(self, cam_param):
        """
        Decode camera parameters from the loaded configuration.
        :param cam_param: Camera parameters dictionary.
        :return: Decoded camera parameters.
        """
        if not isinstance(cam_param, dict):
            raise TypeError("Camera parameters should be a dictionary.")
        
        # Extract necessary parameters
        cam_cords = cam_param['cords']
        cam_intrinsic = cam_param['intrinsic']

        return cam_cords, cam_intrinsic

    def proj_bbx2cam(self, img_array, bbx_dict, 
                     cam_intrinsic, bbx2word, word2cam, 
                     save_path=None, vis_flag=False):
        """
        Project 3D bounding boxes onto a 2D camera image.
        
        Args:
            img_array: numpy array of the camera image
            bbx_dict: dictionary containing bounding box information
            cam_intrinsic: camera intrinsic matrix
            bbx2word: transformation matrix from bounding box to world coordinates
            word2cam: transformation matrix from world to camera coordinates
            save_path: path to save the output image
            vis_flag: whether to visualize the result
        """
        K = cam_intrinsic
        image_w, image_h = ImageLoader.decode_wh(cam_intrinsic)

        # Create a PIL Image for drawing
        image = Image.fromarray(img_array)
        draw = ImageDraw.Draw(image)

        for bbx_id, bbx_data in bbx_dict.items():
            if isinstance(bbx_data, dict):
                # Extract bounding box parameters
                bbx_location = bbx_data['location']
                bbx_center = bbx_data['center']
                bbx_center_location = [bbx_location[0],
                                     bbx_location[1],
                                     bbx_location[2]]
                bbx_extent = bbx_data['extent']
                bbx_angle = bbx_data['angle']

                # Convert to box format
                position, scale, rotation = convert_carla_data_to_box(bbx_angle, bbx_extent, bbx_center_location)

                # Get the 8 corners of the bounding box
                corners = self._get_box_corners(position, scale, rotation)
                
                # Transform corners to camera coordinates
                corners_hom = np.r_[corners, np.ones((1, corners.shape[1]))]
                world_corners = np.dot(bbx2word, corners_hom)
                camera_corners = np.dot(word2cam, world_corners)

                # Adjust coordinate system from UE4 to standard camera coordinates
                camera_corners = np.array([
                    camera_corners[1],
                    camera_corners[2] * -1,
                    camera_corners[0]
                ])

                # Project 3D points to 2D image plane
                points_2d = np.dot(K, camera_corners)
                points_2d = points_2d[:2] / points_2d[2]

                # Filter points within image bounds and in front of camera
                valid_points = (
                    (points_2d[0] >= 0) & (points_2d[0] < image_w) &
                    (points_2d[1] >= 0) & (points_2d[1] < image_h) &
                    (camera_corners[2] > 0)
                )

                if np.any(valid_points):
                    # Draw the bounding box
                    points_2d = points_2d.T
                    self._draw_box(draw, points_2d, valid_points)

        if vis_flag:
            plt.imshow(image)
            plt.axis('off')
            plt.show()

        if save_path:
            if save_path.endswith('.png'):
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                image.save(save_path)
                print(f"Image saved to {save_path}")
            else:
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                if os.path.isdir(save_path):
                    img_name = os.path.basename(self.img_path)
                    save_path = os.path.join(save_path, img_name)
                    image.save(save_path)
                    print(f"Image saved to {save_path}")
                else:
                    print(f"Provided save path is not a valid directory: {save_path}")

    def _get_box_corners(self, position, scale, rotation):
        """
        Calculate the 8 corners of a 3D bounding box.
        
        Args:
            position: dict with keys ['x', 'y', 'z'] for box center
            scale: dict with keys ['x', 'y', 'z'] for box dimensions
            rotation: dict with keys ['x', 'y', 'z'] for box rotation angles
            
        Returns:
            corners: 3x8 numpy array of box corners
        """
        # Get box dimensions
        l = scale['x']
        w = scale['y']
        h = scale['z']

        # Create local corners
        corners = np.array([
            [l/2, w/2, h/2], [l/2, w/2, -h/2],
            [l/2, -w/2, h/2], [l/2, -w/2, -h/2],
            [-l/2, w/2, h/2], [-l/2, w/2, -h/2],
            [-l/2, -w/2, h/2], [-l/2, -w/2, -h/2]
        ]).T

        # Create rotation matrix
        roll = np.radians(rotation['x'])
        pitch = np.radians(rotation['y'])
        yaw = np.radians(rotation['z'])
        
        R_roll = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_pitch = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        R = R_yaw @ R_pitch @ R_roll
        
        # Rotate corners
        corners = R @ corners
        print(f"corners0: {corners[0]}")
        # Translate corners
        print(f"position['x']: {position['x']}")
        print(f"position['y']: {position['y']}")
        corners[0] += position['x']
        # corners[0] += scale['x'] * 0.5
        corners[1] += position['y']
        # print(f"position['z']: {position['z']}")
        # print(f"scale['z']: {scale['z']}")
        corners[2] += scale['z'] * 0.5 # -= position['z']
        print(f"corners: {corners[2]}")
        print("--------------------------------")
        return corners

    def _get_box_color(self, bbx_key):
        """
        Get the color for a specific bounding box class.
        
        Args:
            bbx_key: String indicating the class of the bounding box
            
        Returns:
            tuple: RGB color values (R, G, B)
        """
        color_map = {
            "vehicles": (255, 0, 0),      # Red
            "cars": (255, 0, 0),          # Red
            "trucks": (255, 165, 0),      # Orange
            "pedestrians": (0, 0, 255),   # Blue
            "cyclists": (0, 255, 0),      # Green
            "motorcycles": (255, 255, 0), # Yellow
            "buses": (128, 0, 128),       # Purple
            "traffic_lights": (0, 255, 255), # Cyan
            "traffic_signs": (255, 192, 203) # Pink
        }
        return color_map.get(bbx_key, (255, 255, 255))  # Default to white if class not found

    def _draw_box(self, draw, points_2d, valid_points, color=(255, 0, 0)):
        """
        Draw the bounding box on the image.
        
        Args:
            draw: PIL ImageDraw object
            points_2d: 8x2 numpy array of 2D points
            valid_points: boolean array indicating valid points
            color: RGB tuple for box color
        """
        # Define edges of the box (pairs of points to connect)
        edges = [
            (0, 1), (0, 2), (0, 4),  # Top face
            (1, 3), (1, 5),          # Front face
            (2, 3), (2, 6),          # Right face
            (3, 7),                  # Bottom face
            (4, 5), (4, 6),          # Back face
            (5, 7), (6, 7)           # Left face
        ]
        
        # Draw each edge if both endpoints are valid
        for edge in edges:
            if valid_points[edge[0]] and valid_points[edge[1]]:
                draw.line(
                    [(points_2d[edge[0]][0], points_2d[edge[0]][1]),
                     (points_2d[edge[1]][0], points_2d[edge[1]][1])],
                    fill=color,
                    width=self.line_width
                )

    def single_img_multi_bbx_proj(self, img_path, yaml_path, 
                            save_path=None,
                            cam_key="camera0", bbx_keys=["cars","pedestrians","cyclists"],
                            vis_flag=False):
        """
        Project multiple types of bounding boxes onto a single camera image with different colors.
        
        Args:
            img_path: Path to the camera image
            yaml_path: Path to the YAML file containing camera and bounding box parameters
            save_path: Path to save the output image
            cam_key: Key in the YAML file for camera configuration
            bbx_keys: List of keys in the YAML file for different types of bounding boxes
            vis_flag: Whether to visualize the result
        """
        self.img_path = img_path
        self.yaml_path = yaml_path
        
        # Load camera parameters
        cam_param = self.load_config(yaml_path, cam_key)
        cam_cords, cam_intrinsic = self.decode_cam_param(cam_param)

        image_w, image_h = ImageLoader.decode_wh(cam_intrinsic)
        
        # Create transformation matrices
        bbx2word = np.eye(4)  # Identity matrix for world coordinates
        word2cam = self.create_transformation(
            cam_cords[0], cam_cords[1], cam_cords[2],
            cam_cords[3], cam_cords[4], cam_cords[5]
        )[1]  # Get inverse matrix
        
        # Load and process image
        img_array = ImageLoader.process_jpeg_to_array(img_path)
        image = Image.fromarray(img_array)
        draw = ImageDraw.Draw(image)
        
        # Process each type of bounding box
        for bbx_key in bbx_keys:
            try:
                # Load bounding box data for this class
                bbx_dict = self.load_config(yaml_path, bbx_key)
                color = self._get_box_color(bbx_key)
                
                # Process each bounding box in this class
                for bbx_id, bbx_data in bbx_dict.items():
                    if isinstance(bbx_data, dict):
                        # Extract bounding box parameters
                        bbx_location = bbx_data['location']
                        bbx_center = bbx_data['center']
                        bbx_center_location = [bbx_location[0],
                                             bbx_location[1],
                                             bbx_location[2]]
                        bbx_extent = bbx_data['extent']
                        bbx_angle = bbx_data['angle']

                        # Convert to box format
                        position, scale, rotation = convert_carla_data_to_box(bbx_angle, bbx_extent, bbx_center_location)

                        # Get the 8 corners of the bounding box
                        corners = self._get_box_corners(position, scale, rotation)
                        
                        # Transform corners to camera coordinates
                        corners_hom = np.r_[corners, np.ones((1, corners.shape[1]))]
                        world_corners = np.dot(bbx2word, corners_hom)
                        camera_corners = np.dot(word2cam, world_corners)

                        # Adjust coordinate system from UE4 to standard camera coordinates
                        camera_corners = np.array([
                            camera_corners[1],
                            camera_corners[2] * -1,
                            camera_corners[0]
                        ])

                        # Project 3D points to 2D image plane
                        points_2d = np.dot(cam_intrinsic, camera_corners)
                        points_2d = points_2d[:2] / points_2d[2]

                        # Filter points within image bounds and in front of camera
                        valid_points = (
                            (points_2d[0] >= 0) & (points_2d[0] < image_w) &
                            (points_2d[1] >= 0) & (points_2d[1] < image_h) &
                            (camera_corners[2] > 0)
                        )

                        if np.any(valid_points):
                            # Draw the bounding box with class-specific color
                            points_2d = points_2d.T
                            self._draw_box(draw, points_2d, valid_points, color)
            except KeyError:
                print(f"Warning: No bounding boxes found for class '{bbx_key}'")
                continue

        if vis_flag:
            plt.imshow(image)
            plt.axis('off')
            plt.show()

        if save_path:
            if save_path.endswith('.png'):
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                image.save(save_path)
                print(f"Image saved to {save_path}")
            else:
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                if os.path.isdir(save_path):
                    img_name = os.path.basename(self.img_path)
                    save_path = os.path.join(save_path, img_name)
                    image.save(save_path)
                    print(f"Image saved to {save_path}")
                else:
                    print(f"Provided save path is not a valid directory: {save_path}")

    def single_img_bbx_proj(self, img_path, yaml_path, 
                           output_img_path=None,
                           cam_key="camera0", bbx_key="vehicles",
                           vis_flag=False):
        """
        Project bounding boxes onto a single camera image.
        
        Args:
            img_path: Path to the camera image
            yaml_path: Path to the YAML file containing camera and bounding box parameters
            output_img_path: Path to save the output image
            cam_key: Key in the YAML file for camera configuration
            bbx_key: Key in the YAML file for bounding box data
            vis_flag: Whether to visualize the result
        """
        self.img_path = img_path
        self.yaml_path = yaml_path
        
        # Load camera parameters
        cam_param = self.load_config(yaml_path, cam_key)
        cam_cords, cam_intrinsic = self.decode_cam_param(cam_param)
        
        # Load bounding box data
        bbx_dict = self.load_config(yaml_path, bbx_key)
        
        # Create transformation matrices
        bbx2word = np.eye(4)  # Identity matrix for world coordinates
        word2cam = self.create_transformation(
            cam_cords[0], cam_cords[1], cam_cords[2],
            cam_cords[3], cam_cords[4], cam_cords[5]
        )[1]  # Get inverse matrix
        
        # Load and process image
        img_array = ImageLoader.process_jpeg_to_array(img_path)
        
        # Project bounding boxes
        self.proj_bbx2cam(img_array, bbx_dict,
                         cam_intrinsic, bbx2word, word2cam,
                         save_path=output_img_path, vis_flag=vis_flag)

    def single_img_bbx_proj_smart(self, img_path, yaml_path, 
                                output_dir=None, vis_flag=False):
        """
        Smart version of single_img_bbx_proj that automatically determines camera key.
        
        Args:
            img_path: Path to the camera image
            yaml_path: Path to the YAML file
            output_dir: Directory to save output images
            vis_flag: Whether to visualize the result
        """
        img_name = os.path.basename(img_path)
        cam_key = img_name.split('.')[0].split('_')[1]  # Extract camera key from image name
        
        print(f"Using cam_key: {cam_key}")
        self.single_img_bbx_proj(img_path, yaml_path,
                                output_img_path=output_dir,
                                cam_key=cam_key, bbx_key="vehicles",
                                vis_flag=vis_flag)

    def single_img_bbx_proj_smart_index(self, yaml_path, index,
                                      output_dir=None, vis_flag=False):
        """
        Smart version that uses camera index to determine paths and keys.
        
        Args:
            yaml_path: Path to the YAML file
            index: Camera index
            output_dir: Directory to save output images
            vis_flag: Whether to visualize the result
        """
        yaml_keys = load_yaml(yaml_path).keys()
        camera_key = f"camera{index}"
        if camera_key not in yaml_keys:
            raise KeyError(f"Key '{camera_key}' not found in the YAML file.")

        yaml_base_name = os.path.basename(yaml_path).split('.')[0]
        img_path = os.path.join(os.path.dirname(yaml_path),
                              f"{yaml_base_name}_{camera_key}.png")
        
        print(f"Using cam_key: {camera_key}")
        self.single_img_bbx_proj(img_path, yaml_path,
                                output_img_path=output_dir,
                                cam_key=camera_key, bbx_key="vehicles",
                                vis_flag=vis_flag)

    @staticmethod
    def create_transformation(x, y, z, roll, yaw, pitch):
        """
        Create transformation matrices from camera parameters.
        
        Args:
            x, y, z: Camera position
            roll, yaw, pitch: Camera rotation angles in degrees
            
        Returns:
            tuple: (world2cam, cam2world) transformation matrices
        """
        location = carla.Location(x=x, y=y, z=z)
        rotation = carla.Rotation(roll=roll, yaw=yaw, pitch=pitch)
        carla_transformation = carla.Transform(location, rotation)
        return carla_transformation.get_matrix(), carla_transformation.get_inverse_matrix()

def test1():
    """
    Test function to verify the projection of bounding boxes onto a camera image.
    """
    proj = ProjBBX2Cam(line_width=2)
    cam_key = "camera3"
    img_path = f"data_examples/m2i_radar_dataset/000032_{cam_key}.png"
    yaml_path = "data_examples/m2i_radar_dataset/000032.yaml"
    output_img_path = "data_examples/m2i_radar_dataset/check_bbx2cam"
    
    # Test single class projection
    # proj.single_img_bbx_proj(img_path, yaml_path,
    #                         output_img_path=output_img_path,
    #                         cam_key=cam_key, bbx_key="cars",
    #                         vis_flag=True)
    
    # Test multi-class projection
    proj.single_img_multi_bbx_proj(img_path, yaml_path,
                                  save_path=output_img_path,
                                  cam_key=cam_key,
                                  bbx_keys=["cars", "trucks","pedestrians", "cyclists"],
                                  vis_flag=True)

if __name__ == "__main__":
    test1()
