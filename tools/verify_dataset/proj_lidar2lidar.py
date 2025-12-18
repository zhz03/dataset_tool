# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import os
import open3d as o3d
import numpy as np
import carla
from tools.verify_dataset.pcd_utils import PCLoader
from tools.utils.yaml_utils import load_yaml

class ProjLidar2Lidar:
    """
    Class to project bounding boxes from one LiDAR frame to another.
    This class is a placeholder and does not implement any functionality.
    """

    def __init__(self, point_size=1.0, window_name="lidar2lidar"):
        """
        Initialize the ProjLidar2Lidar with configuration parameters.
        
        Parameters
        ----------
        config : dict
            Configuration parameters for the projection.
        """
        self.point_size = point_size
        self.axis_size = 2.0
        self.window_name = window_name
        self.pcd = None
        self.vis = None
        self.init_o3dpcd()
        self.init_o3d_vis()

    def init_o3dpcd(self):
        """
        pcd_points: numpy array
        """
        self.pcd = o3d.geometry.PointCloud()

    def init_o3d_vis(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.window_name)
        self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        self.vis.get_render_option().point_size = self.point_size
        self.vis.get_render_option().show_coordinate_frame = True
        # Add coordinate frame to visualize point cloud coordinate system
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.axis_size, origin=[0, 0, 0])
        self.vis.add_geometry(coordinate_frame)

    def load_point_cloud(self, pcd_file_path, mode="sep"):
        """
        Load a point cloud from a .pcd or .bin file.

        Args:
            pcd_file_path: Path to .pcd or .bin point cloud file.
            mode: "xyzi" for .pcd with intensity, "sep" for separate XYZ and intensity arrays,

        Returns:
            pcd_points: numpy array of XYZ points
            pcd_intensity: numpy array of intensities (or None if .bin)
            or
            pcd_xyzi: open3d point cloud with intensity information (if .pcd and mode="xyzi")

        """
        if pcd_file_path.endswith('.pcd'):
            if mode == "xyzi":
                pcd_xyzi = PCLoader.load_pcd_with_intensity(pcd_file_path)
                return pcd_xyzi
            elif mode == "sep":
                raise NotImplementedError("Mode 'sep' is not implemented for .pcd files in this method.")
                pcd_points, pcd_intensity = PCLoader.load_pcd_sep(pcd_file_path)
                return pcd_points, pcd_intensity
            else:
                raise ValueError(f"Unsupported mode: {mode}. Use 'xyzi' or 'sep'.")
        elif pcd_file_path.endswith('.bin'):
            pcd_points = PCLoader.load_bin(pcd_file_path)
            pcd_intensity = None
            return pcd_points, pcd_intensity
        else:
            raise ValueError(f"Unsupported file extension: {pcd_file_path}")

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

        return lidar_pose
    
    def combine_point_clouds(self, pcd1, pcd2):
        combined_pcd = pcd1 + pcd2
        return combined_pcd
    
    def save_point_cloud(self, pcd, save_path):
        """
        Save the point cloud to a file.

        Args:
            pcd: Point cloud to save (open3d point cloud).
            save_path: Path to save the point cloud file.
        """
        o3d.t.io.write_point_cloud(save_path, pcd, write_ascii=True)
        print(f"Point cloud saved to {save_path}")

    def visualize_point_cloud(self, pcd):
        """
        Visualize the point cloud using Open3D.

        Args:
            pcd: Point cloud to visualize (open3d point cloud).
        """
        self.vis.add_geometry(pcd)
        self.vis.run()
        self.vis.destroy_window()

    def single_radar2lidar(self, pcd_path1, radar_path, yaml_file,
                    config_key1="lidar_pose1",
                    config_key2=["radar_pose0","radar_pose1","radar_pose2","radar_pose3"], 
                    save_path=None, vis_flag=False):
        """
        Project multiple radar point clouds onto a single LiDAR frame.
        
        Args:
            pcd_path1: Path to the LiDAR point cloud file
            radar_path: Path to the radar point cloud file, example: "/home/zhaoliang/zhaoliang/example_data/fiveway_town03_med_infraset/-125/000035_radar0.pcd"
            yaml_file: Path to the YAML file containing sensor poses
            config_key1: Key for the LiDAR pose in the YAML file
            config_key2: List of keys for radar poses in the YAML file
            save_path: Path to save the combined point cloud
            vis_flag: Whether to visualize the result
        """
        # Load LiDAR point cloud
        pcd1 = self.load_point_cloud(pcd_path1, mode="xyzi")
        pcd1.paint_uniform_color([1, 0, 0])  # Red for LiDAR

        # Load LiDAR pose
        lidar_pose = self.load_pcd_pose(yaml_file, config_key=config_key1)
        print(f"Loaded LiDAR pose: {lidar_pose}")

        # Initialize combined point cloud with LiDAR points
        combined_pcd = pcd1

        # Process each radar point cloud
        for i, radar_key in enumerate(config_key2):
            try:
                # Load radar pose
                radar_pose = self.load_pcd_pose(yaml_file, config_key=radar_key)
                print(f"Loaded Radar pose {i}: {radar_pose}")

                print(f"radar_key: {radar_key}")
                radar_path_dir = os.path.dirname(radar_path)
                radar_file_name = os.path.basename(radar_path).split("_")[0] # 000035

                radar_name = radar_key.replace("radar_pose", "radar")
                pcd_path2 = os.path.join(radar_path_dir, f"{radar_file_name}_{radar_name}.pcd")
                print(f"radar_path2: {pcd_path2}")


                # Construct transformation matrix from radar to LiDAR
                tf_matrix = self.construct_tf_matrix(radar_pose, lidar_pose)
                print(f"Transformation Matrix from Radar {i} to LiDAR:")
                print(tf_matrix)

                # Calculate relative position
                rel_pos = self.convert_tf2pose(tf_matrix)
                print(f"Relative Position for Radar {i} (x, y, z, roll, yaw, pitch):")
                print(rel_pos)

                # Convert to CARLA coordinate system
                rel_pos = self.carla_xyz_conversion(rel_pos)
                print(f"New Relative Position for Radar {i} (x, y, z, roll, yaw, pitch):")
                print(rel_pos)

                # Convert back to transformation matrix
                tf_matrix_new = self.convert_pose2tf(rel_pos)

                # Load and transform radar point cloud
                radar_pcd = self.load_point_cloud(pcd_path2, mode="xyzi")
                # Assign different colors to each radar point cloud
                colors = [
                    [0, 1, 0],    # Green
                    [0, 0, 1],    # Blue
                    [1, 1, 0],    # Yellow
                    [1, 0, 1]     # Magenta
                ]
                radar_pcd.paint_uniform_color(colors[i % len(colors)])



                # # Compute tf matrices from (0,0,0,0,0,0) to lidar/radar position
                # lidar_tf = self.convert_pose2tf(lidar_pose)   # T_lidar
                # radar_tf = self.convert_pose2tf(radar_pose)   # T_radar

                # # T_from_radar_to_lidar = (T_lidar) * (T_radar)^-1
                # tf_matrix = np.linalg.inv(lidar_tf) @ radar_tf

                transformed_radar_pcd = self.apply_transformation(radar_pcd, tf_matrix)
                
                # Combine with existing point cloud
                combined_pcd = self.combine_point_clouds(combined_pcd, transformed_radar_pcd)

            except KeyError as e:
                print(f"Warning: Could not find pose for {radar_key} in YAML file. Skipping...")
                continue

        # Save combined point cloud
        if save_path is not None:
            # self.save_point_cloud(combined_pcd, save_path)
            o3d.io.write_point_cloud(save_path, combined_pcd, write_ascii=True)

        else:
            print("No save path provided, skipping save operation.")

        # Visualize combined point cloud
        if vis_flag:
            self.visualize_point_cloud(combined_pcd)

    def single_comb(self, pcd_path1, pcd_path2, yaml_file,
                    config_key1="lidar_pose1",
                    config_key2="lidar_pose2", 
                    save_path=None, vis_flag=False):
        # Load point clouds
        pcd1 = self.load_point_cloud(pcd_path1, mode="xyzi")
        pcd2 = self.load_point_cloud(pcd_path2, mode="xyzi")
        # add different color to pcd1 and pcd2
        pcd1.paint_uniform_color([1, 0, 0])  # Red for pcd1
        pcd2.paint_uniform_color([0, 1, 0])  # Green for pcd2
        # Load LiDAR poses
        lidar_pose1 = self.load_pcd_pose(yaml_file, config_key=config_key1)
        lidar_pose2 = self.load_pcd_pose(yaml_file, config_key=config_key2)

        print(f"Loaded LiDAR pose 1: {lidar_pose1}")
        print(f"Loaded LiDAR pose 2: {lidar_pose2}")

        # Construct transformation matrix
        # tf_matrix = self.construct_tf_matrix(lidar_pose2, lidar_pose1)
        tf_matrix = self.construct_tf_matrix(lidar_pose2, lidar_pose1)

        print("Transformation Matrix from LiDAR 1 to LiDAR 2:")
        print(tf_matrix)
        rel_pos = self.convert_tf2pose(tf_matrix)
        print("Relative Position (x, y, z, roll, yaw, pitch):")
        print(rel_pos)

        rel_pos = self.carla_xyz_conversion(rel_pos)  # Exchange x and y for Carla coordinate system
        print("zzl New Relative Position (x, y, z, roll, yaw, pitch):")
        print(rel_pos)
        tf_matrix_new = self.convert_pose2tf(rel_pos)

        # Apply transformation to pcd1
        transformed_pcd2 = self.apply_transformation(pcd2, tf_matrix)
        # Combine point clouds
        combined_pcd = self.combine_point_clouds(pcd1, transformed_pcd2)
        # Save combined point cloud
        if save_path is not None:
            self.save_point_cloud(combined_pcd, save_path)
        else:
            print("No save path provided, skipping save operation.")
        # Visualize combined point cloud
        if vis_flag:
            self.visualize_point_cloud(combined_pcd)

    def comb_2lidars(self, pcd_path1, pcd_path2, yaml1_file,yaml2_file,
                     config_key1="lidar_pose0",
                     config_key2="lidar_pose0",
                     save_path=None, vis_flag=False, lidar_mode="xyzi"):

        pcd1 = self.load_point_cloud(pcd_path1, mode=lidar_mode)
        pcd2 = self.load_point_cloud(pcd_path2, mode=lidar_mode)
    # add different color to pcd1 and pcd2
        pcd1.paint_uniform_color([1, 0, 0])  # Red for pcd1
        pcd2.paint_uniform_color([0, 1, 0])  # Green for pcd2
        # Load LiDAR poses
        lidar_pose1 = self.load_pcd_pose(yaml1_file, config_key=config_key1)
        lidar_pose2 = self.load_pcd_pose(yaml2_file, config_key=config_key2)        
        print(f"Loaded LiDAR pose 1: {lidar_pose1}")
        print(f"Loaded LiDAR pose 2: {lidar_pose2}")

        # Construct transformation matrix
        tf_matrix = self.construct_tf_matrix(lidar_pose1, lidar_pose2)

        print("Transformation Matrix from LiDAR 1 to LiDAR 2:")
        print(tf_matrix)
        rel_pos = self.convert_tf2pose(tf_matrix)
        print("Relative Position (x, y, z, roll, yaw, pitch):")
        print(rel_pos)

        # rel_pos = self.carla_xyz_conversion2(rel_pos)  # Exchange x and y for Carla coordinate system
        print("New Relative Position (x, y, z, roll, yaw, pitch):")
        print(rel_pos)
        # rel_pos[4] = 
        tf_matrix_new = self.convert_pose2tf(rel_pos)

        # Apply transformation to pcd1
        transformed_pcd2 = self.apply_transformation(pcd2, tf_matrix_new)
        # Combine point clouds
        combined_pcd = self.combine_point_clouds(pcd1, transformed_pcd2)
        # Save combined point cloud
        if save_path is not None:
            self.save_point_cloud(combined_pcd, save_path)
        else:
            print("No save path provided, skipping save operation.")
        # Visualize combined point cloud
        if vis_flag:
            self.visualize_point_cloud(combined_pcd)

    def single_vis(self, pcd_path1, vis_flag=False):
        """
        Visualize a single point cloud.

        Args:
            pcd_path1: Path to the point cloud file.
            vis_flag: Whether to visualize the point cloud.
        """
        
        pcd_points,_ = self.load_point_cloud(pcd_path1, mode="sep")
        self.pcd.points = o3d.utility.Vector3dVector(pcd_points)

        pcd = self.load_point_cloud(pcd_path1, mode="xyzi")

        if vis_flag:
            self.visualize_point_cloud(pcd)
        else:
            print("Visualization flag is set to False, skipping visualization.")

    @staticmethod
    def carla_xyz_conversion(rel_pos):

        x = rel_pos[0]
        y = rel_pos[1]
        z = rel_pos[2]
        roll = rel_pos[3]
        yaw = rel_pos[4]
        pitch = rel_pos[5]
        rel_pos[0] = -y
        rel_pos[1] = x
        # rel_pos[0] = -x
        # rel_pos[1] = -y
        rel_pos[2] = z
        rel_pos[3] = pitch
        rel_pos[4] = yaw
        rel_pos[5] = roll
        return rel_pos
    
    @staticmethod
    def convert_pose2tf(pose):
        """
        Convert a pose (translation + rotation) to a transformation matrix.

        Args:
            pose: List [x, y, z, roll, yaw, pitch] in degrees.

        Returns:
            tf_matrix: 4x4 transformation matrix.
        """
        x, y, z = pose[0], pose[1], pose[2]
        roll_deg = pose[3]
        yaw_deg = pose[4]
        pitch_deg = pose[5]
        # Convert angles from degrees to radians
        roll = np.deg2rad(roll_deg)
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(-pitch_deg)
        # Compute rotation matrices
        cos_r = np.cos(roll)
        sin_r = np.sin(roll)
        R_x = np.array([
            [1,      0,       0],
            [0, cos_r, -sin_r],
            [0, sin_r,  cos_r]
        ])          
        cos_p = np.cos(pitch)
        sin_p = np.sin(pitch)
        R_y = np.array([
            [ cos_p, 0, sin_p],
            [      0, 1,      0],
            [-sin_p, 0, cos_p]
        ])
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        R_z = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [     0,       0, 1]
        ])
        # Combined rotation: first roll, then pitch, then yaw
        R = R_z @ R_y @ R_x
        # Build the 4x4 homogeneous transformation matrix
        tf_matrix = np.eye(4)
        tf_matrix[0:3, 0:3] = R
        tf_matrix[0:3, 3] = np.array([x, y, z])
        return tf_matrix
    
    @staticmethod
    def convert_tf2pose(tf_matrix):
        """
        Convert a transformation matrix to a pose (translation + rotation).

        Args:
            tf_matrix: 4x4 transformation matrix.

        Returns:
            pose: List [x, y, z, roll, yaw, pitch] in degrees.
        """
        translation = tf_matrix[:3, 3]
        rotation = tf_matrix[:3, :3]

        # Extract Euler angles from the rotation matrix
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1]**2 + rotation[2, 2]**2))
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])

        # Convert radians to degrees
        roll_deg = np.rad2deg(roll)
        pitch_deg = np.rad2deg(pitch)
        yaw_deg = np.rad2deg(yaw)

        pose = [translation[0], translation[1], translation[2], roll_deg, yaw_deg, pitch_deg]
        return pose
    
    @staticmethod
    def apply_transformation(pcd, tf_matrix):
        pcd.transform(tf_matrix)
        return pcd
    
    @staticmethod
    def apply_transformation2(pcd1, tf_matrix):
        """
        Apply a transformation matrix to a point cloud.

        Args:
            pcd1: Point cloud 1 (numpy array of shape Nx3).
            pcd2: Point cloud 2 (numpy array of shape Nx3).
            tf_matrix: 4x4 transformation matrix.

        Returns:
            transformed_pcd: Transformed point cloud.
        """
        # Convert point clouds to homogeneous coordinates
        if pcd1.shape[1] == 3:
            pcd1_h = np.hstack((pcd1, np.ones((pcd1.shape[0], 1))))
        elif pcd1.shape[1] == 4:
            pcd1_h = pcd1
        else:
            raise ValueError("Point cloud must have 3 or 4 columns (XYZ or XYZI).")

        # Apply the transformation
        transformed_pcd = (tf_matrix @ pcd1_h.T).T[:, :3]

        return transformed_pcd
    
    @staticmethod
    def construct_tf_matrix_old(lidar_pose1, lidar_pose2):
        """
        Construct a transformation matrix from lidar_pose1 to lidar_pose2.

        Args:
            lidar_pose1: Pose of the first LiDAR [x, y, z, roll, yaw, pitch] (angles in degrees).
            lidar_pose2: Pose of the second LiDAR [x, y, z, roll, yaw, pitch] (angles in degrees).

        Returns:
            tf_matrix: 4x4 transformation matrix that transforms points from
                    the coordinate frame of lidar_pose1 into the frame of lidar_pose2.
        """
        # Extract translation and rotation (in degrees) for pose1
        x1, y1, z1 = lidar_pose1[0], lidar_pose1[1], lidar_pose1[2]
        roll1_deg  = lidar_pose1[3]   # rotation about X-axis, in degrees
        yaw1_deg   = lidar_pose1[4]   # rotation about Z-axis, in degrees
        pitch1_deg = lidar_pose1[5]   # rotation about Y-axis, in degrees

        # Convert pose1 angles from degrees to radians for computation
        roll1  = np.deg2rad(roll1_deg)
        yaw1   = np.deg2rad(yaw1_deg)
        pitch1 = np.deg2rad(pitch1_deg)

        # Compute rotation matrices for pose1 (R_z(yaw) * R_y(pitch) * R_x(roll))
        cos_r1 = np.cos(roll1)
        sin_r1 = np.sin(roll1)
        R_x1 = np.array([
            [1,      0,       0],
            [0, cos_r1, -sin_r1],
            [0, sin_r1,  cos_r1]
        ])

        cos_p1 = np.cos(pitch1)
        sin_p1 = np.sin(pitch1)
        R_y1 = np.array([
            [ cos_p1, 0, sin_p1],
            [      0, 1,      0],
            [-sin_p1, 0, cos_p1]
        ])

        cos_y1 = np.cos(yaw1)
        sin_y1 = np.sin(yaw1)
        R_z1 = np.array([
            [cos_y1, -sin_y1, 0],
            [sin_y1,  cos_y1, 0],
            [     0,       0, 1]
        ])

        # Combined rotation for lidar_pose1: first roll, then pitch, then yaw
        R1 = R_z1 @ R_y1 @ R_x1

        # Build the 4x4 homogeneous transform matrix for pose1
        T1 = np.eye(4)
        T1[0:3, 0:3] = R1
        T1[0:3, 3]   = np.array([x1, y1, z1])

        # Invert T1 to go from the world frame into the frame of lidar_pose1
        R1_inv = R1.T
        t1_inv = -R1_inv @ np.array([x1, y1, z1])
        T1_inv = np.eye(4)
        T1_inv[0:3, 0:3] = R1_inv
        T1_inv[0:3, 3]   = t1_inv

        # Extract translation and rotation (in degrees) for pose2
        x2, y2, z2 = lidar_pose2[0], lidar_pose2[1], lidar_pose2[2]
        roll2_deg  = lidar_pose2[3]   # rotation about X-axis, in degrees
        yaw2_deg   = lidar_pose2[4]   # rotation about Z-axis, in degrees
        pitch2_deg = lidar_pose2[5]   # rotation about Y-axis, in degrees

        # Convert pose2 angles from degrees to radians
        roll2  = np.deg2rad(roll2_deg)
        yaw2   = np.deg2rad(yaw2_deg)
        pitch2 = np.deg2rad(pitch2_deg)

        # Compute rotation matrices for pose2 (R_z(yaw) * R_y(pitch) * R_x(roll))
        cos_r2 = np.cos(roll2)
        sin_r2 = np.sin(roll2)
        R_x2 = np.array([
            [1,      0,       0],
            [0, cos_r2, -sin_r2],
            [0, sin_r2,  cos_r2]
        ])

        cos_p2 = np.cos(pitch2)
        sin_p2 = np.sin(pitch2)
        R_y2 = np.array([
            [ cos_p2, 0, sin_p2],
            [      0, 1,      0],
            [-sin_p2, 0, cos_p2]
        ])

        cos_y2 = np.cos(yaw2)
        sin_y2 = np.sin(yaw2)
        R_z2 = np.array([
            [cos_y2, -sin_y2, 0],
            [sin_y2,  cos_y2, 0],
            [     0,       0, 1]
        ])

        # Combined rotation for lidar_pose2: first roll, then pitch, then yaw
        R2 = R_z2 @ R_y2 @ R_x2

        # Build the 4x4 homogeneous transform matrix for pose2
        T2 = np.eye(4)
        T2[0:3, 0:3] = R2
        T2[0:3, 3]   = np.array([x2, y2, z2])

        # The relative transform from frame1 to frame2 is T_rel = T2 * inv(T1)
        tf_matrix = T2 @ T1_inv

        return tf_matrix
    
    @staticmethod
    def construct_tf_matrix(lidar_pose1, lidar_pose2):
        """
        Construct a transform that maps points from lidar_pose1 frame into lidar_pose2 frame.
        
        Args:
            lidar_pose1: [x1, y1, z1, roll1, yaw1, pitch1] (degrees, global frame)
            lidar_pose2: [x2, y2, z2, roll2, yaw2, pitch2] (degrees, global frame)

        Returns:
            tf_matrix: 4x4, so that
                p_in_pose2 = tf_matrix @ [p_in_pose1; 1]
        """
        # Convert poses to 4x4 transformation matrices
        tf1 = ProjLidar2Lidar.convert_pose2tf(lidar_pose1)
        tf2 = ProjLidar2Lidar.convert_pose2tf(lidar_pose2)

        # Compute relative transformation: T_1->2 = inv(T2) @ T1
        tf_matrix = np.linalg.inv(tf2) @ tf1

        return tf_matrix
    
def test1():
    #* PassTest
    # test construct_tf_matrix function 
    lidar_pose1 = [0, 0, 0, 0, 0, 0]  # [x, y, z, roll, yaw, pitch]
    lidar_pose2 = [1, 1, 1, 0, 0, 0]  # [x, y, z, roll, yaw, pitch]
    tf_matrix = ProjLidar2Lidar.construct_tf_matrix(lidar_pose1, lidar_pose2)
    print("Transformation Matrix from LiDAR 1 to LiDAR 2:")
    print(tf_matrix)

def test2():
    pcd1_path = "data_examples/i2i_radar/roundabout_town03_med/-125/000031.pcd"
    pcd2_path = "data_examples/i2i_radar/roundabout_town03_med/-125/000031_radar0.pcd"
    yaml_file = "data_examples/i2i_radar/roundabout_town03_med/-125/000031.yaml"
    # save_path = "combined_pcd.pcd"
    proj = ProjLidar2Lidar(point_size=1.0)
    proj.single_comb(pcd1_path, pcd2_path, yaml_file, 
                     config_key1="lidar_pose0", 
                     config_key2="radar_pose0",
                     vis_flag=True)

def test3():
    pcd1_path = "./data_examples/m2i_radar_dataset/000032_lidar0.pcd"
    pcd2_path = "./data_examples/m2i_radar_dataset/000032_radar0.pcd"
    yaml_file = "./data_examples/m2i_radar_dataset/000032.yaml"
    proj = ProjLidar2Lidar(point_size=1.0)
    proj.single_radar2lidar(pcd1_path, pcd2_path, yaml_file,
                    config_key1="lidar_pose0",
                    config_key2=["radar_pose0","radar_pose1","radar_pose2","radar_pose3"], 
                    save_path=None, vis_flag=True)
    
if __name__ == "__main__":
    # test1()
    # test2()
    test3()
