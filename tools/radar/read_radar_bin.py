# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import sys
import os
import numpy as np

def main():
    # Check command-line arguments
    # Ensure that the user has provided at least one argument (the radar bin file path)
    if len(sys.argv) < 2:
        print("Usage: python read_radar_bin.py <radar_bin_file>")
        sys.exit(1)
    
    # Get the file path from the first command-line argument
    file_path = sys.argv[1]
    
    # Check if the specified file exists
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found!")
        sys.exit(1)
    
    try:
        # Read the binary file assuming that the data type is float32.
        # If the file uses a different data type, adjust the dtype parameter accordingly.
        data = np.fromfile(file_path, dtype=np.float32)
        
        # Determine the number of data points in the file
        num_points = data.size
        
        # Print the total number of data points in the file
        print(f"The file '{file_path}' contains {num_points} data points.")
    except Exception as e:
        # Print any errors encountered during file reading
        print("Error reading file:", e)

if __name__ == "__main__":
    main()

