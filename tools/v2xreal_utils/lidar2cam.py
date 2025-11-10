import numpy as np
import os 
import cv2
import yaml
import shutil

import matplotlib.pyplot as plt
from tools.carla_dataset_utils.bbx_projection import decode_yaml
from tools.carla_dataset_utils.box_utils import decode_wh, get_K

def lidar2cam_single_z():
    image_name = "000030_cam3"
    img_file = f"/Users/zhaoliang/Documents/zhz03/github/v2x-real-example/2023-04-03-18-15-32_9_0/1/{image_name}.jpeg"
    lidar_file = "/Users/zhaoliang/Documents/zhz03/github/v2x-real-example/2023-04-03-18-15-32_9_0/1/000030.bin"
    calib_file = "/Users/zhaoliang/Documents/zhz03/github/v2x-real-example/2023-04-03-18-15-32_9_0/1/000030.yaml"
    output_file = f"/Users/zhaoliang/Documents/zhz03/github/dataset_tool/test_examples/v2x-real/veh_1_out/{image_name}.jpeg"
    
    # image_name = "000000_cam2"
    # img_file = f"/Users/zhaoliang/Documents/zhz03/github/dataset_tool/test_examples/v2x-real/infra_2/{image_name}.jpeg"
    # lidar_file = "/Users/zhaoliang/Documents/zhz03/github/dataset_tool/test_examples/v2x-real/infra_2/000000.bin"
    # calib_file = "/Users/zhaoliang/Documents/zhz03/github/dataset_tool/test_examples/v2x-real/infra_2/000000.yaml"
    # output_file = f"/Users/zhaoliang/Documents/zhz03/github/dataset_tool/test_examples/v2x-real/infra_2_out/{image_name}.jpeg"
    
    cam_index = 2 # 0,1,2,3
        
    lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict = decode_yaml(calib_file)
    
    camera_param = camera_list[cam_index]
    extrinsic = np.array(camera_param['extrinsic'])
    intrinsic = np.array(camera_param['intrinsic'])
    image_w, image_h = decode_wh(intrinsic)

    camera_extrinsic_mat = extrinsic
    camera_mat = intrinsic
    print(camera_extrinsic_mat)
    print(camera_mat)

    # Calculate the inverse matrix of the camera-to-LIDAR extrinsic matrix for LIDAR-to-camera coordinate transformation
    camera_to_lidar_mat_inv = np.linalg.inv(camera_extrinsic_mat)

    # Read the LIDAR point cloud data
    lidar_buffer = np.fromfile(lidar_file, dtype=np.float32)
    if lidar_buffer.size % 5 == 0:
        lidar_data = lidar_buffer.reshape(-1, 5)
    elif lidar_buffer.size % 4 == 0:
        lidar_data = lidar_buffer.reshape(-1, 4)
    else:
        raise ValueError(
            f"Unsupported LiDAR point format: got {lidar_buffer.size} floats, "
            "which cannot be reshaped into Nx4 or Nx5."
        )

    # Read the camera image
    image = cv2.imread(img_file)

    # Project LIDAR points onto the camera image
    points_3d = lidar_data[:, :3]
    ones = np.ones((points_3d.shape[0], 1))
    points_3d_hom = np.hstack([points_3d, ones])

    # Transform LIDAR points to camera coordinate system using the inverse matrix
    points_cam = camera_to_lidar_mat_inv.dot(points_3d_hom.T).T

    # Project 3D points to 2D image plane using camera intrinsic matrix
    points_2d = camera_mat.dot(points_cam[:, :3].T).T
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]

    z_values = points_cam[:, 1]

    colormap = plt.cm.jet


    # Define a normalization range that enhances the color difference
    min_z = np.min(z_values)
    max_z = np.max(z_values)
    mid_z = (min_z + max_z) / 2.0
    quarter_range = (max_z - min_z) / 4.0

    # Calculate normalized Z values with enhanced color difference
    z_normalized = np.zeros_like(z_values)

    # Define discrete color ranges based on distance
    num_colors = 20  # Number of distinct colors
    color_ranges = np.linspace(min_z, max_z, num_colors + 1)

    # Map each distance to a specific color range
    for i in range(num_colors):
        z_normalized[(z_values >= color_ranges[i]) & (z_values < color_ranges[i + 1])] = i / num_colors

    # Map normalized Z values to colors using colormap
    colors = colormap(z_normalized)


    point_radius = 2  # Change this value to adjust point size
    alpha = 0.6  # Change this value to adjust transparency (0: fully transparent, 1: fully opaque)

    # Create a blank image for drawing circles with transparency
    overlay = np.zeros_like(image)

    # Plot projected points on the overlay image
    for i in range(points_2d.shape[0]):
        if points_cam[i, 2] > 0:  # Ensure points are in front of the camera
            
            # Get color based on Z value from colormap
            color = (int(colors[i][0] * 255), int(colors[i][1] * 255), int(colors[i][2] * 255))  # Convert RGB components to 0-255 range
            
            # Get coordinates of the point
            x, y = int(points_2d[i, 0]), int(points_2d[i, 1])
            
            # Check if the coordinates are within the image range before drawing
            if x >= 0 and y >= 0 and x < overlay.shape[1] and y < overlay.shape[0]:
                # Draw circle on the overlay image with modified radius
                cv2.circle(overlay, (x, y), point_radius, color, -1, lineType=cv2.LINE_AA)  # Change -1 to adjust circle's border thickness

    # Blend the overlay image with the original image using transparency
    image_with_overlay = cv2.addWeighted(image, 1.0, overlay, alpha, 0)

    # Display the resulting image
    resized_image = cv2.resize(image_with_overlay, (1920, 1080))
    # Save the resulting image as a JPG file
    cv2.imwrite(output_file, resized_image)
    
if __name__ == "__main__":
    lidar2cam_single_z()