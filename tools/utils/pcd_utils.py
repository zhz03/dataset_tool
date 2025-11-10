# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import open3d as o3d
from typing import List

class LidarProcessor:
    def __init__(self, save_parent_folder: str):
        """
        Initializes the LidarProcessor with the directory containing PCD files.
        
        Parameters:
        - save_parent_folder (str): Path to the directory containing PCD files.
        """
        self.save_parent_folder = save_parent_folder

    def merge_pcd_files(self, pcd_file_list: List[str], output_filename: str):
        """
        Reads the PCD files specified in the list, merges them, and saves the result as a new PCD file.

        Parameters:
        - pcd_file_list (list of str): A list containing the paths to the PCD files.
        - output_filename (str): The name (including path) of the output PCD file to save the merged point cloud.
        """
        if not pcd_file_list:
            print("The provided PCD file list is empty.")
            return

        # Initialize an empty PointCloud object to store the merged point cloud
        combined_pcd = o3d.geometry.PointCloud()

        for file in pcd_file_list:
            try:
                # Read each PCD file
                pcd = o3d.io.read_point_cloud(file)

                if pcd.is_empty():
                    print(f"Warning: The file {file} is empty and has been skipped.")
                    continue

                # print(f"Loaded {file}, containing {len(pcd.points)} points.")

                # Add the current point cloud to the combined point cloud
                combined_pcd += pcd
            except Exception as e:
                print(f"Error: Unable to read the file {file}. Error message: {e}")
                continue

        if len(combined_pcd.points) == 0:
            print("The merged point cloud is empty. No valid points were added.")
            return

        # print(f"The merged point cloud contains {len(combined_pcd.points)} points.")

        # Save the merged point cloud to the new PCD file
        success = o3d.io.write_point_cloud(output_filename, combined_pcd)

        if success:
            # print(f"The merged point cloud has been successfully saved as {output_filename}")
            pass
        else:
            print(f"An error occurred while saving the point cloud to {output_filename}.")

    def post_process(self):
        """
        Post-processes the Lidar data by merging internal PCD files with their corresponding non-internal PCD files,
        saving the merged PCD, and deleting the internal PCD files.
        """
        # Step 1: Read all PCD files in the directory
        pcd_files = os.listdir(self.save_parent_folder)
        pcd_file_list = [os.path.join(self.save_parent_folder, file)
                         for file in pcd_files if file.endswith('.pcd')]

        if not pcd_file_list:
            print("No PCD files found in the specified directory.")
            return

        # Step 2: Sort the PCD files to ensure proper pairing
        pcd_file_list.sort()

        # Step 3: Initialize a list to hold internal PCD files
        internal_files = []

        # Step 4: Iterate through the sorted PCD files and merge accordingly
        for file in pcd_file_list:
            basename = os.path.basename(file)
            if 'internal' in basename:
                # Collect internal PCD files
                internal_files.append(file)
            else:
                # Current file is a non-internal PCD file
                if internal_files:
                    # Files to merge: internal files + current non-internal file
                    files_to_merge = internal_files + [file]

                    # print(f"Merging {len(internal_files)} internal files with {file}.")

                    # Perform the merge
                    self.merge_pcd_files(files_to_merge, file)

                    # Step 5: Delete the internal PCD files after successful merge
                    for internal_file in internal_files:
                        try:
                            os.remove(internal_file)
                            # print(f"Deleted internal file: {internal_file}")
                        except Exception as e:
                            print(f"Error deleting file {internal_file}: {e}")

                    # Reset the internal_files list for the next batch
                    internal_files = []
                else:
                    print(f"No internal files to merge with {file}.")

        # Optional: Handle any remaining internal files that do not have a corresponding non-internal file
        if internal_files:
            # print("Warning: There are internal PCD files without a corresponding non-internal file.")
            # for internal_file in internal_files:
            #     print(f"Unmerged internal file: {internal_file}")
            # Decide whether to delete them or keep them
            # For example, to delete:
            for internal_file in internal_files:
                os.remove(internal_file)
                # print(f"Deleted unmerged internal file: {internal_file}")

def merge_pcd_files(pcd_file_list, output_filename):
    """
    Reads the PCD files specified in the list, merges them, and saves the result as a new PCD file.

    Parameters:
    - pcd_file_list (list of str): A list containing the paths to the PCD files.
    - output_filename (str): The name (including path) of the output PCD file to save the merged point cloud.
    """
    if not pcd_file_list:
        print("The provided PCD file list is empty.")
        return

    # Initialize an empty PointCloud object to store the merged point cloud
    combined_pcd = o3d.geometry.PointCloud()

    for file in pcd_file_list:
        try:
            # Read each PCD file
            pcd = o3d.io.read_point_cloud(file)

            if pcd.is_empty():
                print(f"Warning: The file {file} is empty and has been skipped.")
                continue

            print(f"Loaded {file}, containing {len(pcd.points)} points.")

            # Add the current point cloud to the combined point cloud
            combined_pcd += pcd
        except Exception as e:
            print(f"Error: Unable to read the file {file}. Error message: {e}")
            continue

    if len(combined_pcd.points) == 0:
        print("The merged point cloud is empty. No valid points were added.")
        return

    print(f"The merged point cloud contains {len(combined_pcd.points)} points.")

    # Save the merged point cloud to the new PCD file
    success = o3d.io.write_point_cloud(output_filename, combined_pcd)

    if success:
        print(f"The merged point cloud has been successfully saved as {output_filename}")
    else:
        print(f"An error occurred while saving the point cloud to {output_filename}.")

if __name__ == "__main__":
    pass
