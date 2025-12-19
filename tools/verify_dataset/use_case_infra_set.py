# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os
import numpy as np
from opencda_infra.utils.yaml_utils import load_yaml
from opencda_infra.utils.verify_dataset.proj_lidar2cam import ProjLidar2Cam
from opencda_infra.utils.verify_dataset.proj_bbx2lidar import ProjBBX2Lidar
from opencda_infra.utils.verify_dataset.proj_lidar2lidar import ProjLidar2Lidar

def get_other_files_v2xset(yaml_file, index= 0): # v2xset data
    lidar_file = yaml_file.replace(".yaml", ".pcd")
    base_name = yaml_file.split("/")[-1].split(".")[0]
    parent_dir = os.path.dirname(yaml_file)
    # lidar_file = os.path.join(parent_dir, base_name + "_lidar0.pcd")
    radar_file = os.path.join(parent_dir, base_name + "_radar0.pcd")

    camera_index = f"camera{index}"
    img_file = os.path.join(parent_dir, base_name + f"_{camera_index}.png")

    print(f"radar_file: {radar_file}")
    print(f"lidar_file: {lidar_file}")
    
    return lidar_file, radar_file, img_file, camera_index

def get_other_files_infraset(yaml_file, index= 0): # vinfra-set data
    lidar_file = yaml_file.replace(".yaml", ".pcd")
    base_name = yaml_file.split("/")[-1].split(".")[0]
    parent_dir = os.path.dirname(yaml_file)
    lidar_file = os.path.join(parent_dir, base_name + "_lidar0.pcd")
    radar_index = f"radar{index}"
    radar_file = os.path.join(parent_dir, base_name + f"_{radar_index}.pcd")

    camera_index = f"camera{index}"
    img_file = os.path.join(parent_dir, base_name + f"_{camera_index}.png")

    print(f"radar_file: {radar_file}")
    print(f"lidar_file: {lidar_file}")
    
    return lidar_file, radar_file, img_file, camera_index, radar_index

def test1():
    # yaml_file1 = "data_dumping/fourway_cav_town10_dense/-126/000050.yaml" # looks fine
    # yaml_file1 = "/home/zhaoliang/zhaoliang/example_data/fiveway_town03_med_infraset_c_c_day_s10/-125/000031.yaml"
    # yaml_file1 = "./data_dumping/fourway_cav_town10_dense/1/000035.yaml" # (problematic in lidar2cam)
    yaml_file1 = "data_dumping/fourway_nocav_town10_dense/-126/000070.yaml" # looks fine
    yaml_file1 = "/home/zhaoliang/zhaoliang/2021_08_20_20_39_00/-1/000407.yaml" # good: v2xset data
    yaml_file1 = "data_dumping/fourway_cav_town10_dense/-126/000080.yaml" # good: infraset dataset
    yaml_file1 = "./data_dumping/fourway_cav_town10_dense/1/000042.yaml"
    yaml_file1 = "./data_dumping/fourway_cav_town10_dense/-126/000069.yaml"
    yaml_file1 = "/home/zhaoliang/zhaoliang/example_data/fiveway_town03_med_infraset/-125/000031.yaml"
    # yaml_file1 = "/home/zhaoliang/zhaoliang/2021_08_20_20_39_00/-1/000167.yaml" # good
    # yaml_file1 = "/home/zhaoliang/zhaoliang/example_data/fiveway_town03_med_infraset_c_c_day_s10/-125/000031.yaml"
    # yaml_file1 = "./data_dumping/fourway_cav_town10_dense/1/000039.yaml"
    yaml_file2 = "./data_dumping/fourway_cav_town10_dense/1/000035.yaml"
    yaml_file2 = "/home/zhaoliang/zhaoliang/2021_08_20_20_39_00/1752/000407.yaml" # good: v2xset data
    yaml_file2 = "./data_dumping/fourway_cav_town10_dense/1/000080.yaml"
    yaml_file2 = "./data_dumping/fourway_cav_town10_dense/1/000069.yaml"
    yaml_file2 = "/home/zhaoliang/zhaoliang/example_data/fiveway_town03_med_infraset/-125/000031.yaml"
    # yaml_file = "/home/zhaoliang/zhaoliang/example_data/bridgeentry_town07_med_infra_t_c_day_s16/-125/000031.yaml"
    # get lidar file from yaml file name

    dataset_type = "infraset"

    if dataset_type == "v2xset":  # "vinfra-set" or "v2xset"
        lidar_file, radar_file, img_file, camera_index = get_other_files_v2xset(yaml_file1, index=1)
        lidar_file2, radar_file2, img_file2, camera_index2 = get_other_files_v2xset(yaml_file2)
    elif dataset_type == "infraset":
        lidar_file, radar_file, img_file, camera_index, radar_index = get_other_files_infraset(yaml_file1, index=3)
        lidar_file2, radar_file2, img_file2, camera_index2, _ = get_other_files_infraset(yaml_file2)

    # print(f"lidar_file: {lidar_file}")
    # print(f"radar_file: {lidar_file2}")

    yaml_file = yaml_file1  # or yaml_file2
    
    project_type = "radar2lidar"  # "radar2lidar" or "bbx2lidar" or "lidar2lidar"
    if project_type == "radar2lidar":
        proj_radar2lidar = ProjLidar2Lidar(point_size=1.0)
        proj_radar2lidar.single_comb(lidar_file, radar_file, yaml_file, 
                        config_key1="lidar_pose0", 
                        config_key2="radar_pose1",
                        vis_flag=True)   
    elif project_type == "bbx2lidar":
        proj_bbx2lidar= ProjBBX2Lidar()
        proj_bbx2lidar.proj_bbx2lidar(yaml_file, lidar_file, 
                                bbx_class="cars", lidar_key="lidar_pose0")    

    elif project_type == "lidar2lidar":
        proj_lidar2lidar = ProjLidar2Lidar(point_size=1.0)
        proj_lidar2lidar.comb_2lidars(lidar_file, lidar_file2, 
                                      yaml_file, yaml_file2,
                        config_key1="lidar_pose0", 
                        config_key2="lidar_pose0",
                        vis_flag=True)
        
    elif project_type == "lidar2cam":
        proj_lidar2cam = ProjLidar2Cam(point_size=1.0)
        proj_lidar2cam.single_img_lidar_proj(img_file, lidar_file, yaml_file, 
                                      cam_key=camera_index, lidar_key="lidar_pose0", 
                                      vis_flag=True)

def test_case(yaml_file1,yaml_file2=None, save_dir = None, 
                dataset_type="infraset", project_type="lidar2cam"):
    # dataset_type = "v2xset"

    if dataset_type == "v2xset":  # "vinfra-set" or "v2xset"
        lidar_file, radar_file, img_file, camera_index = get_other_files_v2xset(yaml_file1, index=2)
        lidar_file2, radar_file2, img_file2, camera_index2 = get_other_files_v2xset(yaml_file2)
    elif dataset_type == "infraset":
        lidar_file, radar_file, img_file, camera_index, radar_index = get_other_files_infraset(yaml_file1, index=2)
        lidar_file2, radar_file2, img_file2, camera_index2, _ = get_other_files_infraset(yaml_file2)

    # print(f"lidar_file: {lidar_file}")
    # print(f"radar_file: {lidar_file2}")

    yaml_file = yaml_file1  # or yaml_file2
    
    project_type = "lidar2cam"  # "radar2lidar" or "bbx2lidar" or "lidar2lidar"
    if project_type == "radar2lidar":
        proj_radar2lidar = ProjLidar2Lidar(point_size=1.0)
        proj_radar2lidar.single_comb(lidar_file, radar_file, yaml_file, 
                        config_key1="lidar_pose0", 
                        config_key2="radar_pose2",
                        vis_flag=True)   
    elif project_type == "bbx2lidar":
        proj_bbx2lidar= ProjBBX2Lidar()
        proj_bbx2lidar.proj_bbx2lidar(yaml_file, lidar_file, 
                                bbx_class="vehicles", lidar_key="lidar_pose")    

    elif project_type == "lidar2lidar":
        proj_lidar2lidar = ProjLidar2Lidar(point_size=1.0)
        proj_lidar2lidar.comb_2lidars(lidar_file, lidar_file2, 
                                      yaml_file, yaml_file2,
                        config_key1="lidar_pose0", 
                        config_key2="lidar_pose0",
                        vis_flag=True)
        
    elif project_type == "lidar2cam":
        proj_lidar2cam = ProjLidar2Cam(point_size=1.0)
        proj_lidar2cam.single_img_lidar_proj(img_file, lidar_file, yaml_file, 
                                      cam_key=camera_index, lidar_key="lidar_pose", 
                                      output_img_path=save_dir,
                                      vis_flag=False)
def test2():

    root_dir = "/home/zhaoliang/zhaoliang/2021_08_20_20_39_00/-1"
    save_dir = "/home/zhaoliang/zhaoliang/2021_08_20_20_39_00/process_-12"

    # root_dir = "/home/zhaoliang/zhaoliang/2021_08_20_20_39_00/1752"
    # save_dir = "/home/zhaoliang/zhaoliang/2021_08_20_20_39_00/process_1"

    root_dir = "/home/zhaoliang/zhaoliang/zhz03_github/opencda_old/OpenCDA/data_dumping/2025_06_09_20_52_20/811"
    save_dir = "/home/zhaoliang/zhaoliang/zhz03_github/opencda_old/OpenCDA/data_dumping/2025_06_09_20_52_20/811_process_2"

    yaml_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.yaml')]
    yaml_files.sort()  # Sort the files if needed
    count = 0

    for yaml_file in yaml_files:
        if count >= 300:
            break
        count += 1
        print(f"Processing {yaml_file}")

        test_case(yaml_file, yaml_file2=yaml_file,save_dir=save_dir, 
                dataset_type="v2xset", project_type="lidar2cam")

def test3(yaml_file):
    # yaml_file = "./data_dumping/fourway_cav_town10_dense/1/000036.yaml"
    print(yaml_file)
    yaml_config = load_yaml(yaml_file)
    true_ego_pos = yaml_config['true_ego_pos']
    lidar_pose0 = yaml_config['lidar_pose0']
    cam1 = yaml_config['camera1']
    cam2 = yaml_config['camera2']
    cam1_pos = cam1['cords']
    cam2_pos = cam2['cords']


    # do some calculations
    cam1_diff = np.array(cam1_pos) - np.array(true_ego_pos)
    cam2_diff = np.array(cam2_pos) - np.array(true_ego_pos)

    cam1_lidar_diff = np.array(cam1_pos) - np.array(lidar_pose0)
    cam2_lidar_diff = np.array(cam2_pos) - np.array(lidar_pose0)
    # Convert to regular float and format without scientific notation
    cam1_diff = [float(f"{x:.6f}") for x in cam1_diff]
    cam2_diff = [float(f"{x:.6f}") for x in cam2_diff]

    cam1_lidar_diff = [float(f"{x:.6f}") for x in cam1_lidar_diff]
    cam2_lidar_diff = [float(f"{x:.6f}") for x in cam2_lidar_diff]
    print(f"Camera 1 position relative to true ego position: {cam1_diff}")
    print(f"Camera 2 position relative to true ego position: {cam2_diff}")
    print("---")

    print(f"Camera 1 position relative to lidar position: {cam1_lidar_diff}")
    print(f"Camera 2 position relative to lidar position: {cam2_lidar_diff}")

    return true_ego_pos,lidar_pose0, cam1_pos, cam2_pos,\
        cam1_diff, cam2_diff, cam1_lidar_diff, cam2_lidar_diff


def test4():
    yaml1 = "./data_dumping/fourway_cav_town10_dense/1/000034.yaml"
    yaml2 = "./data_dumping/fourway_cav_town10_dense/1/000035.yaml"

    true_ego_pos1, lidar_pose01, cam1_pos1, cam2_pos1, \
        cam1_diff1, cam2_diff1, cam1_lidar_diff1, cam2_lidar_diff1 = test3(yaml1)
    true_ego_pos2, lidar_pose02, cam1_pos2, cam2_pos2, \
        cam1_diff2, cam2_diff2, cam1_lidar_diff2, cam2_lidar_diff2 = test3(yaml2)
    
    # Compare the differences
    true_pos_diff = np.array(true_ego_pos1) - np.array(true_ego_pos2)
    lidar_pos_diff = np.array(lidar_pose01) - np.array(lidar_pose02)
    cam1_pos_diff = np.array(cam1_pos1) - np.array(cam1_pos2)
    cam2_pos_diff = np.array(cam2_pos1) - np.array(cam2_pos2)


    # Convert to regular float and format without scientific notation
    true_pos_diff = [float(f"{x:.6f}") for x in true_pos_diff]
    lidar_pos_diff = [float(f"{x:.6f}") for x in lidar_pos_diff]
    cam1_pos_diff = [float(f"{x:.6f}") for x in cam1_pos_diff]
    cam2_pos_diff = [float(f"{x:.6f}") for x in cam2_pos_diff]

    print("===")

    print(f"True ego position difference: {true_pos_diff}")
    print(f"Lidar position difference: {lidar_pos_diff}")
    print(f"Camera 1 position difference: {cam1_pos_diff}")
    print(f"Camera 2 position difference: {cam2_pos_diff}")


if __name__ == "__main__":
    test1()
    # test2()
