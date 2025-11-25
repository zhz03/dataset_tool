# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Registers the 3D projection
import sys
import os

def read_radar_bin0(file_path):
    """
    Reads the radar bin file and returns a NumPy array of shape (N, 4).
    It assumes that the bin file contains data as float32 values arranged
    in the order [x, y, z, value] for each radar point.
    
    Parameters:
        file_path (str): Path to the radar bin file.
        
    Returns:
        np.ndarray: An array of shape (N, 4) containing the radar data.
    """
    try:
        # Read binary data from the file as float32.
        data = np.fromfile(file_path, dtype=np.float32)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Check that the total number of floats is a multiple of 4
    if data.size % 4 != 0:
        print("Warning: The data size is not a multiple of 4. The file may be corrupted or in an unexpected format.")
    
    # Reshape to (-1, 4)
    radar_data = data.reshape((-1, 4))
    return radar_data
    
def read_radar_bin(file_path):
    """
    Reads the radar bin file and returns a NumPy array of shape (N, 4).
    It assumes that the bin file contains data as float32 values arranged
    in the order [x, y, z, value] for each radar point.
    
    If the total number of floats is not a multiple of 4, the extra values
    at the end are discarded.
    
    Parameters:
        file_path (str): Path to the radar bin file.
        
    Returns:
        np.ndarray: An array of shape (N, 4) containing the radar data.
    """
    try:
        # Read binary data from the file as float32.
        data = np.fromfile(file_path, dtype=np.float32)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    total_floats = data.size
    remainder = total_floats % 4
    if remainder != 0:
        print(f"Warning: The data size ({total_floats}) is not a multiple of 4. "
              f"Ignoring the last {remainder} elements.")
        data = data[:total_floats - remainder]
    
    # Now reshape the array to have 4 columns per radar point.
    radar_data = data.reshape((-1, 4))
    return radar_data

def visualize_4d_radar(radar_data):
    """
    Visualizes 4D radar data in a 3D scatter plot.
    
    Parameters:
        radar_data (np.ndarray): An array of shape (N, 4) where each row is [x, y, z, value].
                                 The value (4th dimension) is mapped to color.
    """
    # Extract coordinates and the fourth value
    x = radar_data[:, 0]
    y = radar_data[:, 1]
    z = radar_data[:, 2]
    values = radar_data[:, 3]
    
    # Create a new figure and a 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a scatter plot, mapping the 4th dimension to color using the 'viridis' colormap
    scatter = ax.scatter(x, y, z, c=values, cmap='viridis', marker='o', s=20)
    
    # Add a color bar to show the value mapping
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label("Radar Value (Intensity/Velocity)")
    
    # Set plot title and axis labels
    ax.set_title("4D Radar Visualization")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    
    plt.show()

def main():
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python visualize_4d_radar.py <radar_bin_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        sys.exit(1)
    
    # Read the radar bin file
    radar_data = read_radar_bin(file_path)
    print(f"Read radar data with shape: {radar_data.shape}")
    
    # Visualize the radar data
    visualize_4d_radar(radar_data)

if __name__ == "__main__":
    main()


