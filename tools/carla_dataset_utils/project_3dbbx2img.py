# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import numpy as np
import yaml
import cv2
import os
import math
from datetime import datetime
from tools.carla_dataset_utils.bbx_projection import decode_yaml
from tools.carla_dataset_utils.project_lidar2cam import create_transformation, process_jpeg_to_array
from tools.carla_dataset_utils.box_utils import convert_carla_data_to_box, \
    create_rotated_box, create_rotated_box_points, proj_points_2_img, save_img, get_K

# Construct rotation matrix
def rotation_matrix(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def project_bounding_box():
    yaml_file = "./data_dumping/example/2024_11_30_16_15_46/125/000045.yaml"
    image_path = "./data_dumping/example/2024_11_30_16_15_46/125/camera1_000045.png"  # Replace with your actual image path
    output_dir="./data_dumping/example/2024_11_30_16_15_46/bbx_projection"

    image_w = 1920 
    image_h = 1080
    fov = 120

    lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict = decode_yaml(yaml_file)
    K = get_K(image_w, image_h, fov)

    my_cam = camera_list[1]
    camera_extrinsic = my_cam['extrinsic']
    camera_intrinsic = my_cam['intrinsic']

    # Extract vehicle parameters
    # vehicle_dict = data.get('vehicles', {})
    vehicle_keys = list(vehicle_dict.keys())

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    image_height, image_width = image.shape[0], image.shape[1]

    # Camera intrinsic matrix
    K = camera_intrinsic

    # Camera extrinsic matrix (world to camera)
    world_to_camera = np.linalg.inv(camera_extrinsic)

    # Transformation matrix for coordinate system change (UE4 to standard camera)
    ue_to_cam = np.array([
        [0, 1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])

    for key in vehicle_keys:
        vehicle = vehicle_dict[key]
        vehicle_location = vehicle['location']
        vehicle_rotation = vehicle['angle']
        vehicle_extent = vehicle['extent']

        # Vehicle dimensions
        length = vehicle_extent[0] * 2
        width = vehicle_extent[1] * 2
        height = vehicle_extent[2] * 2

        # Bounding box corners in vehicle's local coordinate system
        x_corners = [length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2]
        y_corners = [width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]
        z_corners = [-height/2, -height/2, -height/2, -height/2, height/2, height/2, height/2, height/2]

        corners_3D = np.array([x_corners, y_corners, z_corners])

        # Vehicle rotation angles (convert degrees to radians if necessary)
        roll, pitch, yaw = vehicle_rotation
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

        # Vehicle transformation matrix (vehicle to world)
        veh_to_world,_ = create_transformation(
            vehicle_location[0], vehicle_location[1], vehicle_location[2],
            roll, pitch, yaw
        )

        # Transform corners from vehicle to world coordinate system
        corners_3D_hom = np.vstack((corners_3D, np.ones((1, corners_3D.shape[1]))))
        corners_world = np.dot(veh_to_world, corners_3D_hom)

        # Transform points from world to camera coordinate system
        corners_camera = np.dot(world_to_camera, corners_world)

        # Apply UE4 to camera coordinate system transformation
        corners_camera[:3, :] = np.dot(ue_to_cam, corners_camera[:3, :])

        # Project points onto image plane using camera intrinsic matrix
        projected_2D = np.dot(K, corners_camera[:3, :])

        # Normalize homogeneous coordinates
        projected_2D[0, :] /= projected_2D[2, :]
        projected_2D[1, :] /= projected_2D[2, :]

        # Check if any point is in front of the camera
        if not np.any(corners_camera[2, :] > 0):
            print(f"Vehicle {key}: Bounding box is not visible (all points are behind the camera).")
            continue

        # Check if any projected point is within image bounds
        u = projected_2D[0, :]
        v = projected_2D[1, :]
        within_width = np.logical_and(u >= 0, u < image_width)
        within_height = np.logical_and(v >= 0, v < image_height)
        within_image = np.logical_and(within_width, within_height)

        if not np.any(within_image):
            print(f"Vehicle {key}: Bounding box is not visible (all points are outside the image frame).")
            continue

        # Convert to integer pixel coordinates
        u = u.astype(int)
        v = v.astype(int)

        # Define bounding box edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom square
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top square
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical lines
        ]

        # Draw bounding box
        for edge in edges:
            pt1 = (u[edge[0]], v[edge[0]])
            pt2 = (u[edge[1]], v[edge[1]])
            # Only draw if both points are within image bounds
            if (0 <= pt1[0] < image_width and 0 <= pt1[1] < image_height) and \
               (0 <= pt2[0] < image_width and 0 <= pt2[1] < image_height):
                cv2.line(image, pt1, pt2, color=(0, 255, 0), thickness=2)

    # Save and display the image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"projected_bounding_box_{timestamp}.jpg")
    cv2.imwrite(output_path, image)
    print(f"Saved projected bounding box image to {output_path}")

    cv2.imshow('3D Bounding Box Projection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 1. Load and parse input data
    yaml_data = """
    [Your YAML data here, omitted for brevity]
    """
    yaml_file = "./data_dumping/example/2024_11_30_16_15_46/125/000045.yaml"
    image_path = "./data_dumping/example/2024_11_30_16_15_46/125/camera1_000045.png"  # Replace with your actual image path
    
    image_w = 1920 
    image_h = 1080
    fov = 120

    lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict = decode_yaml(yaml_file)
    K = get_K(image_w, image_h, fov)

    my_cam = camera_list[1]
    camera_extrinsic = my_cam['extrinsic']
    camera_intrinsic = my_cam['intrinsic']

    print("camera_extrinsic:",camera_extrinsic)
    print("K:",K)
    print("camera_intrinsic:",camera_intrinsic)

    # data = yaml.safe_load(yaml_data)

    # Extract camera parameters
    # camera = data['camera_1']
    # camera_cords = camera['cords']
    # camera_extrinsic = camera['extrinsic']
    # camera_intrinsic = camera['intrinsic']

    # # Extract vehicle parameters
    # get vehicle data from vehicle_dict
    # get the key of vehicle_dict
    key_list = list(vehicle_dict.keys())
    print("key:",key_list)

    for key in key_list:
        print("--------")
        print("key:",key)
        vehicle = vehicle_dict[key]

        # vehicle = vehicle_dict[key_list[0]]

        # vehicle = data['vehicles']['420']
        vehicle_location = vehicle['location']
        vehicle_rotation = vehicle['angle']
        vehicle_extent = vehicle['extent']
        vehicle_center = vehicle['center']

        # 2. Construct camera intrinsic and extrinsic matrices
        K = np.array(camera_intrinsic)  # Camera intrinsic matrix

        extrinsic_matrix = np.array(camera_extrinsic)  # Camera extrinsic matrix
        world_to_camera = np.linalg.inv(extrinsic_matrix)  # World-to-camera transformation

        # 3. Calculate the 3D bounding box vertices of the vehicle
        length = vehicle_extent[0] * 2
        width = vehicle_extent[1] * 2
        height = vehicle_extent[2] * 2

        x_corners = [length/2, length/2, -length/2, -length/2, length/2, length/2, -length/2, -length/2]
        y_corners = [width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]
        z_corners = [0, 0, 0, 0, height, height, height, height]

        corners_3D = np.array([x_corners, y_corners, z_corners])

        # Vehicle rotation angles (assumed in radians; convert from degrees if necessary)
        # roll, yaw, pitch = vehicle_rotation
        x, y, z = vehicle_location
        roll, yaw, pitch = vehicle_rotation
        print("roll:",roll)
        print("yaw:",yaw)
        print("pitch:",pitch)
        # Uncomment below lines if angles are in degrees
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

        R = rotation_matrix(roll, pitch, yaw)
        veh2world,world2veh = create_transformation(x, y, z, roll, yaw, pitch)
        print("R:",R)
        print("veh2world:",veh2world)

        # Rotate and translate the vertices
        corners_3D_world = np.dot(R, corners_3D)
        vehicle_pos = np.array(vehicle_location).reshape((3,1))
        corners_3D_world = corners_3D_world + vehicle_pos

        # 4. Transform the points into the camera coordinate system
        corners_3D_world_hom = np.vstack((corners_3D_world, np.ones((1, corners_3D_world.shape[1]))))
        corners_3D_camera = np.dot(world_to_camera, corners_3D_world_hom)

        # Filter points in front of the camera (positive Z-values)
        valid_indices = corners_3D_camera[2, :] > 0  # Z > 0 for visibility

        # If no points are in front of the camera, the bounding box is not visible
        if not np.any(valid_indices):
            print("Bounding box is not visible in the image (all points are behind the camera).")
        else:
            # 5. Project points onto the 2D image plane
            projected_2D = np.dot(K, corners_3D_camera[:3, :])
            projected_2D[0, :] /= projected_2D[2, :]
            projected_2D[1, :] /= projected_2D[2, :]

            # 6. Check if the projected points are within the image boundaries

            image = cv2.imread(image_path)
            image_height, image_width = image.shape[0], image.shape[1]

            u = projected_2D[0, :]
            v = projected_2D[1, :]

            # Check if points are within image bounds
            within_width = np.logical_and(u >= 0, u < image_width)
            within_height = np.logical_and(v >= 0, v < image_height)
            within_image = np.logical_and(within_width, within_height)

            # If no points are in the image, the bounding box is not visible
            if not np.any(within_image):
                print("Bounding box is not visible in the image (all points are outside the image frame).")
            else:
                # 7. Draw the 2D bounding box on the image
                u = u.astype(int)
                v = v.astype(int)

                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)
                ]

                for edge in edges:
                    pt1 = (u[edge[0]], v[edge[0]])
                    pt2 = (u[edge[1]], v[edge[1]])
                    # Only draw edges within the image bounds
                    if (0 <= pt1[0] < image_width and 0 <= pt1[1] < image_height) and \
                    (0 <= pt2[0] < image_width and 0 <= pt2[1] < image_height):
                        cv2.line(image, pt1, pt2, color=(0, 255, 0), thickness=2)

                # Display and save the resulting image
                cv2.imshow('3D Bounding Box Projection', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                output_dir = "./data_dumping/example/2024_11_30_16_15_46/bbx_projection"
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"{output_dir}/{timestamp}.png"

                cv2.imwrite('projected_bounding_box.jpg', image)



# zzl
def opv2v_bbx_projection():
    img_path = "/home/carma/dg/dataset_example/test_town04/-125/000032_camera2.png"
    yaml_file = "/home/carma/dg/dataset_example/test_town04/-125/000032.yaml"
    output_dir = "/home/carma/dg/results2/bbx2image"
    lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict = decode_yaml(yaml_file)

    lidar_index = 0
    cam_index = 2
    camera_param = camera_list[cam_index]
    # lidar_cords = lidar_pose_list[lidar_index]
    
    cam_cords = camera_param['cords']
    camera_intrinsics = camera_param['intrinsic']

    im_array = process_jpeg_to_array(img_path)
    _,world_2_camera = create_transformation(cam_cords[0], cam_cords[1], cam_cords[2], \
                                              cam_cords[3], cam_cords[4], cam_cords[5]) 

    # print("lidar_cords:", lidar_cords)    

    im_array = process_jpeg_to_array(img_path)

    # print("lidar_cords_new:", lidar_cords)
    # lidar_2_world,world_2_lidar = create_transformation(lidar_cords[0], lidar_cords[1], lidar_cords[2], \
    #                                         lidar_cords[3], lidar_cords[4], lidar_cords[5])
    # print("---")

    # Process all vehicles and project their bounding boxes
    for key in vehicle_dict.keys():
        vehicle = vehicle_dict[key]
        # print(f"Processing vehicle: {key}")
        # print(vehicle)
        location = vehicle["location"]
        angle =  vehicle["angle"]
        extent = vehicle["extent"]
        position, scale, rotation = convert_carla_data_to_box(angle,extent,location)
        translated_vertices, edges, colors = create_rotated_box_points(position, scale, rotation)
        
        # translated_vertices are already in world coordinates, so we use identity matrix
        points_2_world = np.eye(4)  # Identity matrix - no transformation needed

        # print("translated_vertices:",translated_vertices)
        # print("shape of translated_vertices:",translated_vertices.shape)

        # convert translated_vertices from (N,3) to (3,N)
        translated_vertices = translated_vertices.T
        # print("shape of translated_vertices:",translated_vertices.shape)
        # print("edges:",edges)

        # Project each vehicle's bounding box onto the image
        im_array = proj_points_2_img(im_array, translated_vertices, edges, points_2_world, 
                          world_2_camera, camera_intrinsics, color=(0, 255, 0))

    # Save the final image with all bounding boxes after processing all vehicles
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file_name = f"{output_dir}/{timestamp}.png"
    save_img(im_array, save_file_name)
    print(f"All bounding boxes projected and saved to {save_file_name}") 

if __name__ == "__main__":
    # main()
    # project_bounding_box()
    opv2v_bbx_projection()