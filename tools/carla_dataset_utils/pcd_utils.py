# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt  # for colormapping

class PCLoader:
    def __init__(self):
        pass
    
    @staticmethod
    def load_pcd_with_intensity(file_path , color='white'):
        """
        Load a point cloud from a PCD file, ensure it has an intensity channel,
        and store it as a legacy Open3D PointCloud (self.pcd) so that
        self.pcd.transform(...) can be used directly.

        Args:
            file_path (str): Path to the .pcd file.

        After calling this method, self.pcd will be an o3d.geometry.PointCloud
        whose points and colors (grayscale from intensity) are set.
        """
        # 1. Read the file into a tensor-based PointCloud
        pcd_t = o3d.t.io.read_point_cloud(file_path)

        # 2. If there is no "intensity" channel or its shape is wrong, create a dummy channel of zeros
        if "intensity" not in pcd_t.point or pcd_t.point["intensity"].shape[1] != 1:
            num_pts = pcd_t.point["positions"].shape[0]
            zeros = np.zeros((num_pts, 1), dtype=np.float32)
            pcd_t.point["intensity"] = o3d.core.Tensor(zeros, dtype=o3d.core.Dtype.Float32)

        # 3. Extract raw data as NumPy arrays
        positions = np.asarray(pcd_t.point["positions"].numpy())   # shape: [N, 3]
        intensity = np.asarray(pcd_t.point["intensity"].numpy()).squeeze()  # shape: [N,]

        # 4. Normalize intensity to [0,1] for coloring
        if intensity.max() > 1.0 or intensity.min() < 0.0:
            intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())

        # 5. Build a legacy PointCloud and assign points
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(positions)

        if color == 'gray':
            # Convert intensity → grayscale RGB colors and assign to self.pcd.colors
            #    Each point’s color is (i, i, i) where i ∈ [0,1].
            gray = np.vstack([intensity, intensity, intensity]).T.astype(np.float64)
            new_pcd.paint_uniform_color(gray)
        elif color == 'white':
            # If 'white' is specified, set all points to white color
            color = [1.0, 1.0, 1.0]
            new_pcd.paint_uniform_color(color)
        elif isinstance(color, (list, tuple)) and len(color) == 3:
            # If a specific RGB color is provided, apply it to the point cloud
            new_pcd.paint_uniform_color(color)
        else:
            raise ValueError("Invalid color option. Use 'gray', 'hot', or a specific RGB tuple.")
        
        return new_pcd

    @staticmethod
    def load_pcd_with_intensity_tensor(file_path):
        """
        Load a point cloud from a PCD file and ensure it has an intensity channel.
        Returns a tensor-based PointCloud on CPU with both positions and intensity.
        """
        # Read the point cloud from disk into a tensor-based PointCloud
        pcd = o3d.t.io.read_point_cloud(file_path)

        # If there is no "intensity" attribute or its shape is incorrect, create a dummy channel
        if "intensity" not in pcd.point or pcd.point["intensity"].shape[1] != 1:
            # Create a zero-valued intensity array of shape [N, 1]
            num_points = pcd.point["positions"].shape[0]
            dummy_intensity = np.zeros((num_points, 1), dtype=np.float32)
            # Assign the dummy intensity to the point cloud
            pcd.point["intensity"] = o3d.core.Tensor(dummy_intensity, dtype=o3d.core.Dtype.Float32)

        # Extract the raw intensity tensor (shape: [N, 1])
        pcd_intensity = pcd.point["intensity"]
        # Extract the positions tensor (shape: [N, 3])
        pcd_points = pcd.point["positions"]

        # Create a new tensor-based PointCloud on CPU to store positions and intensity
        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        new_pcd = o3d.t.geometry.PointCloud(device)

        # Copy the position data into the new PointCloud
        new_pcd.point["positions"] = o3d.core.Tensor(pcd_points, dtype, device)
        # Copy the intensity data into the new PointCloud
        new_pcd.point["intensity"] = o3d.core.Tensor(pcd_intensity, dtype, device)

        return new_pcd

    @staticmethod
    def load_pcd_sep(file_path):
        pcd = o3d.t.io.read_point_cloud(file_path)
        # verify the shape, if shape is 1,3, then no intensity
        # Check if the point cloud has intensity information
        # pcd.point["positions"] always exists, but "intensity" may not
        # If only positions (shape Nx3), set intensity to None
        if "intensity" not in pcd.point or pcd.point["intensity"].shape[1] != 1:
            # No intensity channel, create dummy intensity as None
            pcd.point["intensity"] = o3d.core.Tensor(np.zeros((pcd.point["positions"].shape[0], 1)), dtype=o3d.core.Dtype.Float32)
        # Access intensity and position
        pcd_intensity = pcd.point["intensity"]  # Access intensity
        pcd_points = pcd.point["positions"]  # Access position
        # Convert to Numpy array (if needed)
        pcd_intensity = pcd_intensity[:, :].numpy()
        pcd_points = pcd_points[:, :].numpy()

        return pcd_points, pcd_intensity    

    @staticmethod
    def load_bin(bin_path):
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points
    
    @staticmethod
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

    def get_points_info(self, points):
        print(len(points[:]))
        print(points.shape)
        print(points[:20])        

if __name__ == "__main__":
    pass