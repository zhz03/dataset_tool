# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os, sys
sys.path.insert(0, os.path.abspath("..")) # .../dataset_tool/tools

import carla
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from utils.yaml_utils import load_yaml
from verify_dataset.pcd_utils import PCLoader
from verify_dataset.img_utils import ImageLoader

class ProjLidar2Cam:
    """
    Class to project LiDAR points onto camera images.
    This class is used to verify the dataset by projecting LiDAR points onto camera images.
    """

    def __init__(self, point_size=1):

        self.point_size = point_size
        self.cam_param = None
        self.img_path = None
        self.pcd_path = None
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

    def proj_lidar2cam(self, img_array, pc_array, 
                       cam_intrinsic, lidar2word, word2cam, 
                       save_path=None, vis_flag=False):
        K = cam_intrinsic
        image_w, image_h = ImageLoader.decode_wh(cam_intrinsic)

        # Extract the local coordinates of the point cloud (3, N)
        local_lidar_points = np.array(pc_array[:, :3]).T  # x, y, z in the LiDAR frame
        # Extract x-values for color mapping
        x_values = local_lidar_points[0, :]  # x-values in LiDAR coordinates

        # Convert point cloud to homogeneous coordinates (4, N)
        local_lidar_points_hom = np.r_[
            local_lidar_points, [np.ones(local_lidar_points.shape[1])]]  

        # Transform points from the LiDAR coordinate system to world coordinates
        world_points = np.dot(lidar2word, local_lidar_points_hom)         
        # Transform points from world coordinates to camera coordinates
        sensor_points = np.dot(word2cam, world_points)

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

        dot_extent = int(self.point_size)  # Size of the dot for each point
        if dot_extent <= 0:
            # Draw single-pixel points
            img_array[v_coord, u_coord] = color_map
        else:
            # Draw square dots of size `dot_extent`
            for i in range(len(points_2d)):
                # Ensure indices are within image bounds
                x_min = max(0, u_coord[i] - dot_extent)
                x_max = min(image_w, u_coord[i] + dot_extent)
                y_min = max(0, v_coord[i] - dot_extent)
                y_max = min(image_h, v_coord[i] + dot_extent)
                img_array[y_min:y_max, x_min:x_max] = color_map[i]

        # Convert the numpy array back to a PIL Image
        image = Image.fromarray(img_array)         

        if vis_flag:
            # Visualize the projected points on the image
            plt.imshow(image)
            plt.axis('off')
            plt.show()

        if save_path:
            if save_path.endswith('.png'):
                # check if the directory exists, if not create it
                save_dir = save_path.rsplit('/', 1)[0]
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                    print(f"Created directory: {save_dir}")
                else:
                    print(f"Directory already exists: {save_dir}")
                # Save the image with projected points
                image.save(save_path)
                print(f"Image saved to {save_path}")
            else:
                # check if save_path is a directory
                if save_path and not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                    print(f"Created directory: {save_path}")
                else:
                    print(f"Directory already exists: {save_path}")
                if os.path.isdir(save_path):
                    # Save the image with projected points in the specified directory
                    img_name = os.path.basename(self.img_path)
                    save_path = os.path.join(save_path, img_name)
                    image.save(save_path)
                    print(f"Image saved to {save_path}")
                else:
                    print(f"Provided save path is not a valid directory: {save_path}")
        else:
            print("No save path provided, image not saved.")       

    def single_img_lidar_proj(self, img_path, pcd_path, yaml_path, 
                              output_img_path=None,
                              cam_key="camera0", lidar_key="lidar_pose",
                              vis_flag=False):
        """
        Project LiDAR points onto a single camera image.
        
        :param img_path: Path to the camera image.
        :param yaml_path: Path to the YAML file containing camera parameters.
        :param config_key: Key in the YAML file for the camera configuration.
        :return: Projected LiDAR points on the camera image.
        """
        self.img_path = img_path
        self.pcd_path = pcd_path
        self.yaml_path = yaml_path
        # Load camera parameters from YAML
        cam_param = self.load_config(yaml_path, cam_key)
        lidar_cords = self.load_config(yaml_path, lidar_key)

        # Decode camera parameters
        cam_cords, cam_intrinsic = self.decode_cam_param(cam_param)
        image_w, image_h = ImageLoader.decode_wh(cam_intrinsic)

        print("cam_cords:", cam_cords)
        print("lidar_cords:", lidar_cords)  
        
        # Create carla.Transform from LiDAR parameters
        lidar2word, world2lidar = self.create_transformation(lidar_cords[0], lidar_cords[1], 
                                                             lidar_cords[2], lidar_cords[3], 
                                                             lidar_cords[4], lidar_cords[5])
        
        # Create carla.Transform from camera parameters
        cam2word, word2cam = self.create_transformation(cam_cords[0], cam_cords[1], 
                                                         cam_cords[2], cam_cords[3], 
                                                         cam_cords[4], cam_cords[5])
        
        # Compute the transformation from LiDAR to camera 
        # lidar2cam = np.dot(word2cam, lidar2word)

        img_array = ImageLoader.process_jpeg_to_array(img_path)
        pc_array = PCLoader.process_pcd_to_array(pcd_path)

        self.proj_lidar2cam(img_array, pc_array,
                            cam_intrinsic, lidar2word, word2cam, 
                            save_path=output_img_path, vis_flag=vis_flag)

    def single_img_lidar_proj_smart(self, img_path, pcd_path, yaml_path, 
                               output_dir=None,vis_flag=False):

        img_name = os.path.basename(img_path)
        pcd_name = os.path.basename(pcd_path)
        cam_key = img_name.split('.')[0].split('_')[1]  # Extract camera key from image name
        lidar_key = pcd_name.split('.')[0].split('_')[1]
        lidar_key = lidar_key.replace('lidar', 'lidar_pose')  # Ensure it matches the expected key format
        
        print(f"cam_key: {cam_key}, lidar_key: {lidar_key}")
        self.single_img_lidar_proj(img_path, pcd_path, yaml_path, 
                                   output_img_path=output_dir,
                                   cam_key=cam_key, lidar_key=lidar_key,
                                   vis_flag=vis_flag)

    def single_img_lidar_proj_smart_index(self, yaml_path, index, 
                               output_dir=None,vis_flag=False):
        yaml_keys = load_yaml(yaml_path).keys()
        camera_key = f"camera{index}"
        if camera_key not in yaml_keys:
            raise KeyError(f"Key '{camera_key}' not found in the YAML file.")
        lidar_name = f"lidar0"
        lidar_key = f"lidar_pose0"

        yaml_base_name = os.path.basename(yaml_path).split('.')[0]

        # convert yaml name into img_path and pcd_path
        img_path = os.path.join(os.path.dirname(yaml_path), f"{yaml_base_name}_{camera_key}.png")
        pcd_path = os.path.join(os.path.dirname(yaml_path), f"{yaml_base_name}_{lidar_name}.pcd")
        print(f"img_path: {img_path}, pcd_path: {pcd_path}")
        # img_name = os.path.basename(img_path)
        # pcd_name = os.path.basename(pcd_path)
        
        print(f"Using cam_key: {camera_key}, lidar_key: {lidar_key}")
        self.single_img_lidar_proj(img_path, pcd_path, yaml_path, 
                                   output_img_path=output_dir,
                                   cam_key=camera_key, lidar_key=lidar_key,
                                   vis_flag=vis_flag)
        
    @staticmethod
    def create_transformation(x, y, z, roll, yaw, pitch):
        """
        Reconstruct a carla.Transform object from provided transformation parameters.

        :param x: X-coordinate of the LiDAR location.
        :param y: Y-coordinate of the LiDAR location.
        :param z: Z-coordinate of the LiDAR location.
        :param roll: Roll angle of the LiDAR rotation (degrees).
        :param yaw: Yaw angle of the LiDAR rotation (degrees).
        :param pitch: Pitch angle of the LiDAR rotation (degrees).
        :return: carla.Transform object representing the LiDAR transformation.
        """
        location = carla.Location(x=x, y=y, z=z)
        rotation = carla.Rotation(roll=roll, yaw=yaw, pitch=pitch)
        carla_transformation = carla.Transform(location, rotation)
        return carla_transformation.get_matrix(), carla_transformation.get_inverse_matrix()
    

def test1():
    #* Test PASS
    """
    Test function to verify the projection of LiDAR points onto a camera image.
    """
    proj = ProjLidar2Cam(point_size=1.0)
    #img_path = "/media/carma/ui_4/data_transfer/data_dumping/radar_dataset_discard/train/000032_camera0.png"
    #pcd_path = "/media/carma/ui_4/data_transfer/data_dumping/radar_dataset_discard/train/000032_lidar0.pcd"
    yaml_path = "/media/carma/ui_4/data_transfer/data_dumping/radar_dataset_discard/train/bridgeentry_town07_dense_infra_radar_t_c_day_s16/-125/000032.yaml"
    output_img_path = "/home/carma/dg/results_new"

    # proj.single_img_lidar_proj(img_path, pcd_path, yaml_path, 
    #                            output_img_path=None,
    #                            cam_key="camera0", lidar_key="lidar_pose0",
    #                            vis_flag=True)
    
    # proj.single_img_lidar_proj_smart(img_path, pcd_path, yaml_path, vis_flag=True)
    proj.single_img_lidar_proj_smart_index(yaml_path, 3,vis_flag=True, output_dir=output_img_path)
    
if __name__ == "__main__":
    test1()
