# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
import open3d as o3d
import numpy as np

def load_pcd2op3d(file_path):
    # Read the point cloud
    pcd_points = o3d.t.io.read_point_cloud(file_path)
    
    # Check if the intensity attribute exists
    if "intensity" in pcd_points.point:
        pcd_intensity = pcd_points.point["intensity"].numpy()
        print("Intensity values loaded:", pcd_intensity)
    else:
        print("Warning: Intensity attribute not found!")

    # Extract point data and create a traditional point cloud object
    points = pcd_points.point.positions.numpy()  # Extract point coordinates
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)  # Convert to Open3D point format
    return pcd

def visualize_pcd(file_path):
    # Create a visualization window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]  # Set background color
    vis.get_render_option().point_size = 1.0  # Set the point size
    vis.get_render_option().show_coordinate_frame = True  # Display the coordinate frame

    # Load the point cloud data
    pcd = load_pcd2op3d(file_path)

    # Add the point cloud to the visualization window
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def test(pcd_path):
    # Specify the PCD file path
    pcd_file_path = pcd_path
    # Call the visualization function
    visualize_pcd(pcd_file_path)

if __name__ == "__main__":
    #test("/home/carma/dg/dg_dump/dg_t5_4cam/-126/000031_radar3.pcd")
    pass