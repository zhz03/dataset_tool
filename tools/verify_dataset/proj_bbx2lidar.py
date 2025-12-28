# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os, sys
sys.path.insert(0, os.path.abspath("..")) # .../dataset_tool/tools

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

from verify_dataset.pcd_utils import PCLoader
from utils.yaml_utils import load_yaml
from verify_dataset.box_util import convert_carla_data_to_box, create_rotated_box

class ProjBBX2Lidar:
    def __init__(self, point_size=1.0, pose_type="global", window_name="bbx2lidar"):
        self.point_size = point_size
        # For BBX 2 LiDAR projection, we need to convert all bbx 
        # # from global position to the local coordinate frame. 
        self.pose_type = pose_type # global: global position or "local" position
        self.window_name = window_name

        self.pcd = None # Place holder for o3d.geometry.PointCloud
        self.vis = None # Place holder for o3d.visualization.Visualizer

        self.init_o3d_vis()
        self.init_o3dpcd()
    
    def _get_box_color(self, bbx_key):
        """
        Get the color for a specific bounding box class.
        
        Args:
            bbx_key: String indicating the class of the bounding box
            
        Returns:
            tuple: RGB color values (R, G, B)
        """
        color_map = {
            "vehicles": (1, 0, 0),      # Red
            "cars": (1, 0, 0),          # Red
            "trucks": (1, 0.65, 0),      # Orange
            "pedestrians": (0, 0, 1),   # Blue
            "cyclists": (0, 1, 0),      # Green
            "motorcycles": (1, 1, 0), # Yellow
            "buses": (0.50, 0, 0.50),       # Purple
            "traffic_lights": (0, 1, 1), # Cyan
            "traffic_signs": (1, 0.75, 0.80) # Pink
        }
        return color_map.get(bbx_key, (1, 1, 1))  # Default to white if class not found

    def init_o3d_vis(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.window_name)
        self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        self.vis.get_render_option().point_size = self.point_size
        self.vis.get_render_option().show_coordinate_frame = True
        # Add coordinate frame to visualize point cloud coordinate system
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        self.vis.add_geometry(coordinate_frame)

    def init_o3dpcd(self):
        """
        pcd_points: numpy array
        """
        self.pcd = o3d.geometry.PointCloud()
        # self.pcd.points = o3d.utility.Vector3dVector(pcd_points)
        # self.pcd.colors = o3d.utility.Vector3dVector(pcd_intensity)
    
    def load_point_cloud(self, pcd_file_path):
        """
        Load a point cloud from a .pcd or .bin file.

        Args:
            pcd_file_path: Path to .pcd or .bin point cloud file.

        Returns:
            pcd_points: numpy array of XYZ points
            pcd_intensity: numpy array of intensities (or None if .bin)
        """
        if pcd_file_path.endswith('.pcd'):
            pcd_points, pcd_intensity = PCLoader.load_pcd_sep(pcd_file_path)
        elif pcd_file_path.endswith('.bin'):
            pcd_points = PCLoader.load_bin(pcd_file_path)
            pcd_intensity = None
        else:
            raise ValueError(f"Unsupported file extension: {pcd_file_path}")

        return pcd_points, pcd_intensity

    def load_pcd_pose(self, yaml_path, config_key = "lidar_pose"):
        """
        Load LiDAR pose (translation + rotation) from a YAML file.

        Args:
            yaml_path: Path to the YAML file.
            config_key: Key under which LiDAR pose is stored.

        Returns:
            lidar_pose: List [x, y, z, roll, yaw, pitch] in degrees.
        """
        if yaml_path.endswith(".yaml"):
            yaml_config = load_yaml(yaml_path)
            if config_key not in yaml_config:
                raise KeyError(f"Key '{config_key}' not found in the YAML file.")
            lidar_pose = yaml_config[config_key]
        else:
            raise ValueError(f"Unsupported file extension: {yaml_path}")

        return lidar_pose # [x,y,z,roll,yaw,pitch]

    def _wrap_angle(self, angle_deg):
        """
        Normalize an angle in degrees to the range [-180, 180].

        Args:
            angle_deg: A single angle in degrees.

        Returns:
            wrapped: The same angle wrapped into [-180, 180].
        """
        wrapped = ((angle_deg + 180) % 360) - 180
        return wrapped

    def convert_bbx_global2local(self, bbx_pos, bbx_scale, bbx_angle, lidar_pose):
        """
        Convert a bounding box definition from the global frame into the LiDAR’s local frame.

        Args:
            bbx_pos: dict with keys ['x', 'y', 'z'] representing the bounding box center in global coords.
            bbx_scale: dict with keys ['x', 'y', 'z'] representing the bounding box size.
            bbx_angle: dict with keys ['x'=roll, 'y'=yaw, 'z'=pitch] in degrees in the global frame.
            lidar_pose: list [x, y, z, roll, yaw, pitch] of the LiDAR in global coords, all in degrees.

        Returns:
            re_bbx_pos: dict with keys ['x', 'y', 'z'] for the box center in LiDAR local coords.
            re_bbx_scale: same as bbx_scale (size is unchanged).
            re_bbx_angle: dict with keys ['x'=roll, 'y'=yaw, 'z'=pitch] in degrees in LiDAR local frame.
        """
        # 1. POSITION TRANSFORMATION: global → LiDAR local
        lidar_t = np.array(lidar_pose[:3])  # translation [x, y, z]
        # LiDAR’s roll/pitch/yaw order must match how bbx_angle is stored.
        # lidar_pose is [x, y, z, roll, yaw, pitch]
        lidar_roll = lidar_pose[3]
        lidar_yaw = lidar_pose[4]
        lidar_pitch = lidar_pose[5]
        # Create rotation matrix for LiDAR: apply rotations in order (roll → pitch → yaw)
        lidar_rpy_deg = [lidar_roll, lidar_pitch, lidar_yaw]
        lidar_rpy_rad = np.radians(lidar_rpy_deg)
        R_lidar = R.from_euler('xyz', lidar_rpy_rad).as_matrix()

        # Global BBX center
        pos_global = np.array([bbx_pos['x'], bbx_pos['y'], bbx_pos['z']])
        # Translate into LiDAR-centered coordinates, then rotate into LiDAR frame
        relative_pos = pos_global - lidar_t
        pos_local = R_lidar.T @ relative_pos  # inverse rotation

        re_bbx_pos = {
            'x': float(pos_local[0]),
            'y': float(pos_local[1]),
            'z': float(pos_local[2])
        }

        # 2. ORIENTATION TRANSFORMATION: direct Euler-angle difference (global minus LiDAR)
        bbx_roll = bbx_angle['x']
        bbx_yaw = bbx_angle['y']
        bbx_pitch = bbx_angle['z']

        # Subtract LiDAR’s angles from BBX global angles, then wrap into [-180, 180]
        local_roll = self._wrap_angle(bbx_roll  - lidar_roll)
        local_yaw = self._wrap_angle(bbx_yaw - lidar_yaw)  # Adjust yaw to match LiDAR's frame
        local_pitch = self._wrap_angle(bbx_pitch - lidar_pitch)

        re_bbx_angle = {
            'x': float(local_roll),   # roll
            'y': float(local_yaw),    # yaw
            'z': float(local_pitch)   # pitch
        }

        # 3. SCALE remains unchanged
        re_bbx_scale = bbx_scale.copy()

        return re_bbx_pos, re_bbx_scale, re_bbx_angle


    def load_bbx(self, bbx_file_path, config_key="vehicles"):
        """
        Load bounding-box definitions from a YAML file.

        Args:
            bbx_file_path: Path to .yaml file containing bounding-box entries.
            config_key: Key under which box data is stored in the YAML.

        Returns:
            bbx_dict: Dictionary of bounding-box entries.
        """
        if bbx_file_path.endswith('.yaml'):
            yaml_config = load_yaml(bbx_file_path)
            if config_key not in yaml_config:
                raise KeyError(f"Key '{config_key}' not found in the YAML file.")
            bbx_dict = yaml_config[config_key]
        else:
            raise ValueError(f"Unsupported file extension: {bbx_file_path}")

        return bbx_dict

    def proj_bbx2lidar(self, yaml_file, pcd_file_path, 
                            bbx_classes=["vehicles", "cyclists", "pedestrians"], lidar_key="lidar_pose",
                            save_path='', vis_Flag=True):
        """
        Main routine: load LiDAR point cloud, load BBX definitions and LiDAR pose, convert each BBX
        from global frame into LiDAR local frame, then visualize them together.

        Args:
            yaml_file: Path to the YAML file containing both BBX data and LiDAR pose.
            pcd_file_path: Path to the point-cloud file (.pcd or .bin).
            bbx_class: Key under which the set of bounding boxes is stored (e.g. “vehicles”).
            lidar_key: Key under which LiDAR pose is stored (e.g. “lidar_pose”).
        """
        pcd_points, pcd_intensity = self.load_point_cloud(pcd_file_path)
        self.pcd.points = o3d.utility.Vector3dVector(pcd_points)
        self.vis.create_window(visible=vis_Flag)
        self.vis.add_geometry(self.pcd)

        bbx_dicts = {}
        for bbx_class in bbx_classes:
            try:
                bbx_dict = self.load_bbx(yaml_file, config_key=bbx_class)
                for bbx_id, bbx_data in bbx_dict.items():
                    if isinstance(bbx_data, dict):
                        bbx_data['class_type'] = bbx_class
                        bbx_dicts[f"{bbx_class}_{bbx_id}"] = bbx_data
            except KeyError:
                print(f"Warning: Class '{bbx_class}' not found in {yaml_file}")

        lidar_pose = self.load_pcd_pose(yaml_file, config_key=lidar_key)

        for bbx_id, bbx_data in bbx_dicts.items():
            bbx_class = bbx_data.get('class_type', 'unknown')
            box_offset_deg = 0 # for bounding box offset degree
            
            bbx_location = bbx_data['location']
            bbx_center = bbx_data['center']
            bbx_center_location = [bbx_location[0], # bbx_location[0] + bbx_center[0],
                                    bbx_location[1], # bbx_location[1] + bbx_center[1]
                                    bbx_location[2]]
            bbx_extent = bbx_data['extent']
            bbx_angle = bbx_data['angle']
            position, scale, rotation = convert_carla_data_to_box(bbx_angle, bbx_extent, bbx_center_location)
            
            if self.pose_type == "global":
                position, scale, rotation  = self.convert_bbx_global2local(position, scale, rotation,lidar_pose)
            elif self.pose_type == "local":
                pass
            else:
                raise ValueError("Not supported pose_type, only support 'global' and 'local'.")
            
            # Convert the data to a box format
            box = create_rotated_box(position, scale, rotation, color=self._get_box_color(bbx_class), 
                                     offset_deg=box_offset_deg, class_type=bbx_class)
            self.vis.add_geometry(box)

        if vis_Flag:
            self.vis.run()

        if save_path:
            print(f"Saving bbx2lidar viz at {save_path}.")
            self.vis.capture_screen_image(save_path)

        self.vis.destroy_window()


def test1():
    #! Test: Pass
    # test the class and load_bbx function
    bbx_file_path = "/home/zzl/zzl/bc/example_data/bridgeentry_town07_sparse_infra_t_c_day_s14/-125/000031.yaml"
    projector = ProjBBX2Lidar()
    bbx_dict = projector.load_bbx(bbx_file_path, config_key="cars")
    print(bbx_dict)

def savetopng():
    #! Test: Pass
    input_root = "/media/carma/ui_4/data_transfer/data_dumping/radar_dataset/town05_intersection3_4cam_radar/-125"
    output_root = "/home/carma/dg/results_new/town05_intersection3_4cam_radar/bbx2lidar"

    frame = 32
    pcd_file_path = f"{input_root}/{frame:06}_lidar0.pcd"
    yaml_file = f"{input_root}/{frame:06}.yaml"
    
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    projector = ProjBBX2Lidar()
    projector.proj_bbx2lidar(yaml_file, pcd_file_path, bbx_classes=["cars", "cyclists", "pedestrians", "trucks"], lidar_key="lidar_pose0", 
                             save_path=f"{output_root}/{frame:06}.png", vis_Flag=True)

def view_in_open3d():
    #! Test: Pass
    input_root = "/media/carma/ui_4/data_transfer/data_dumping/radar_dataset/bridgeentry_town07_dense_4cam_radar/-125"
    output_root = "/home/carma/dg/results_new/bridgeentry_town07_dense_4cam_radar"
    frame = 148
    pcd_file_path = f"{input_root}/{frame:06}_lidar0.pcd"
    yaml_file = f"{input_root}/{frame:06}.yaml"

    print(pcd_file_path, yaml_file)

    projector = ProjBBX2Lidar()
    projector.proj_bbx2lidar(yaml_file, pcd_file_path, bbx_classes=["cars", "cyclists", "pedestrians", "trucks"], lidar_key="lidar_pose0")


def test3():
    #! Test: Pass
    yaml_file = "./data_examples/m2i_radar_dataset/000032.yaml"
    pcd_file_path = "./data_examples/m2i_radar_dataset/000032_radar0.pcd"

    projector = ProjBBX2Lidar()
    projector.proj_bbx2lidar(yaml_file, pcd_file_path, bbx_class="cars", lidar_key="radar_pose0")

if __name__ == "__main__":
    view_in_open3d()