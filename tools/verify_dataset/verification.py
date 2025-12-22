# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os, sys
sys.path.insert(0, os.path.abspath("..")) # .../dataset_tool/tools

import numpy as np
from utils.yaml_utils import load_yaml
from verify_dataset.proj_lidar2cam import ProjLidar2Cam
from verify_dataset.proj_bbx2lidar import ProjBBX2Lidar
from verify_dataset.proj_radar2lidar import ProjLidar2Lidar
from verify_dataset.proj_bbx2cam import ProjBBX2Cam
from verify_dataset.proj_radar2cam import ProjRadar2Cam

def get_other_files_v2xset(yaml_file, index= 0): # v2xset data
    lidar_file = yaml_file.replace(".yaml", ".pcd")
    base_name = yaml_file.split("/")[-1].split(".")[0]
    parent_dir = os.path.dirname(yaml_file)
    radar_index = f"radar_pose{index}"
    # lidar_file = os.path.join(parent_dir, base_name + "_lidar0.pcd")
    radar_file = os.path.join(parent_dir, base_name + f"_radar{index}.pcd")

    camera_index = f"camera{index}"
    img_file = os.path.join(parent_dir, base_name + f"_{camera_index}.png")

    print(f"radar_file: {radar_file}")
    print(f"lidar_file: {lidar_file}")
    
    return lidar_file, radar_file, img_file, camera_index, radar_index

def get_other_files_infraset(yaml_file, index= 0): # vinfra-set data
    lidar_file = yaml_file.replace(".yaml", ".pcd")
    base_name = yaml_file.split("/")[-1].split(".")[0]
    parent_dir = os.path.dirname(yaml_file)
    lidar_file = os.path.join(parent_dir, base_name + "_lidar0.pcd")
    radar_index = f"radar_pose{index}"
    radar_file = os.path.join(parent_dir, base_name + f"_radar{index}.pcd")

    camera_index = f"camera{index}"
    img_file = os.path.join(parent_dir, base_name + f"_{camera_index}.png")

    print(f"radar_file: {radar_file}")
    print(f"lidar_file: {lidar_file}")
    
    return lidar_file, radar_file, img_file, camera_index, radar_index

def test_case(yaml_file1,yaml_file2=None, save_dir = None, 
                dataset_type="infraset", project_type="lidar2cam",
                sensor_index=0,bbx_class="cars"):
    # dataset_type = "v2xset"
    print("hello")
    if dataset_type == "v2xset":  # "vinfra-set" or "v2xset"
        lidar_file, radar_file, img_file, camera_index, radar_index = get_other_files_v2xset(yaml_file1, index=sensor_index)
        lidar_file2, radar_file2, img_file2, camera_index2, radar_index2 = get_other_files_v2xset(yaml_file2)
    elif dataset_type == "infraset":
        lidar_file, radar_file, img_file, camera_index, radar_index = get_other_files_infraset(yaml_file1, index=sensor_index)
        lidar_file2, radar_file2, img_file2, camera_index2, radar_index2 = get_other_files_infraset(yaml_file2, index=sensor_index)

    # print(f"lidar_file: {lidar_file}")
    # print(f"radar_file: {lidar_file2}")

    yaml_file = yaml_file1  # or yaml_file2
    
    project_type = project_type  # "radar2lidar" or "bbx2lidar" or "lidar2lidar"
    if project_type == "radar2lidar":
        print(radar_index2)
        proj_radar2lidar = ProjLidar2Lidar(point_size=1.0)
        proj_radar2lidar.single_comb(lidar_file, radar_file, yaml_file, 
                        # save_path=save_dir,
                        config_key1="lidar_pose0", 
                        config_key2=radar_index2,
                        vis_flag=True)   
    elif project_type == "radar2lidar_batch":
        proj_radar2lidar = ProjLidar2Lidar(point_size=1.0)
        proj_radar2lidar.single_radar2lidar(lidar_file, radar_file, yaml_file, 
                        save_path = save_dir,
                        config_key1="lidar_pose0", 
                        config_key2=["radar_pose0","radar_pose1","radar_pose2","radar_pose3"],
                        vis_flag=False)   
    elif project_type == "bbx2lidar":
        proj_bbx2lidar= ProjBBX2Lidar()
        proj_bbx2lidar.proj_bbx2lidar(yaml_file, lidar_file, 
                                bbx_class=bbx_class, lidar_key="lidar_pose")    

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
                                      output_img_path=save_dir,
                                      vis_flag=False)
    
    elif project_type == "bbx2cam":
        proj_bbx2cam = ProjBBX2Cam(line_width=2)
        proj_bbx2cam.single_img_multi_bbx_proj(img_file, yaml_file,
                                  save_path=save_dir,
                                  cam_key=camera_index,
                                  bbx_keys=["cars", "trucks","pedestrians", "cyclists"],
                                  vis_flag=True)

def test1_single_lidar2cam(root_dir,sensor_index,sensor_type="lidar2cam"):
    # root_dir = "/home/zhaoliang/zhaoliang/2021_08_20_20_39_00/-1"
    # sensor_index = 0
    # sensor_type = "lidar2cam"
    save_dir = root_dir + f"_{sensor_type}_{sensor_index}"

    yaml_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.yaml')]
    yaml_files.sort()  # Sort the files if needed

    i = 0
    for yaml_file in yaml_files:
        print(f"Processing {yaml_file}")
        test_case(yaml_file, yaml_file2=yaml_file,save_dir=save_dir, 
                dataset_type="v2xset", project_type="lidar2cam", sensor_index=sensor_index) 
        if i > 30:
            break
        i += 1   

def test1_batch_lidar2cam():
    root_dir = "/media/guest/pred_1/v2x_new/2021_08_21_21_35_56/1"

    for sensor_index in range(4):

        test1_single_lidar2cam(root_dir,sensor_index,sensor_type="lidar2cam")
   

def test2_single_radar2lidar():
    sensor_index = 0
    root_dir = "/home/carma/dg/results_new"
    save_dir = f"{root_dir}/radar2lidar_{sensor_index}"
    print(save_dir)
    yaml_file = "/media/carma/ui_4/data_transfer/data_dumping/radar_dataset/town05_intersection3_4cam_radar/-125/000031.yaml"
    yaml_file1= "/media/carma/ui_4/data_transfer/data_dumping/radar_dataset/town05_intersection3_4cam_radar/-125/000031.yaml"

    test_case(yaml_file,yaml_file2=yaml_file1, save_dir = save_dir, 
                    dataset_type="infraset", project_type="radar2lidar",
                    sensor_index=sensor_index)   

def test3_single_bbx2cam():
    yaml_file = "/home/zhaoliang/zhaoliang/example_data/fiveway_town03_med_infraset/-125/000035.yaml"
    yaml_file = "/media/guest/pred_1/v2x_new/2021_08_21_22_21_37/1/000082.yaml"
    sensor_index = 2

    test_case(yaml_file, yaml_file2=yaml_file, save_dir=None,
             dataset_type="infraset", project_type="bbx2cam",
             sensor_index=sensor_index)
    
def test4_single_radar2lidar_batch(yaml_file, save_path):

    # yaml_file= "/media/guest/pred_1/v2x_new/2021_08_21_21_35_56/-1/000431.yaml"

    test_case(yaml_file, yaml_file2=yaml_file, save_dir=save_path,
             dataset_type="v2xset", project_type="radar2lidar_batch",
             sensor_index=0)

def test5_single_bbx2lidar():
    yaml_file = "/home/zhaoliang/zhaoliang/zhz03_github/OpenCDA-Infra/data_dumping/fourway_cav_town10_dense_noinfra/1/000070.yaml"
    yaml_file = "/media/guest/pred_1/v2x_new/2021_08_21_21_35_56/-1/000031.yaml"
    # yaml_file = "/home/zhaoliang/zhaoliang/example_data/fiveway_town03_med_infraset/-126/000038.yaml"
    sensor_index = 2

    test_case(yaml_file, yaml_file2=yaml_file, save_dir=None,
             dataset_type="v2xset", project_type="bbx2lidar",
             sensor_index=sensor_index,bbx_class="vehicles")
    
def test_many(input_root, output_root, classes, start_frame, end_frame, single_frame, flags):
    # lidar2cam
    if flags["lidar2cam"]:
        lidar2cam_proj = ProjLidar2Cam(point_size=1.0)
        for frame in range(start_frame, end_frame + 1):
            for cam in range(0, 4):
                yaml_path = f"{input_root}/{frame:06}.yaml"
                output_img_path = f"{output_root}/lidar2cam/camera{cam}"
                lidar2cam_proj.single_img_lidar_proj_smart_index(yaml_path, index=cam, vis_flag=False, output_dir=output_img_path)
    
    # radar2cam
    if flags["radar2cam"]:
        radar2cam_proj = ProjRadar2Cam(point_size=0.7)
        for frame in range(start_frame, end_frame + 1):
            for cam in range(0, 4):
                yaml_path = f"{input_root}/{frame:06}.yaml"
                output_img_path = f"{output_root}/radar2cam/camera{cam}"
                radar2cam_proj.single_img_all_radar_proj_smart_index(yaml_path, index=cam, vis_flag=False, output_dir=output_img_path)
                radar2cam_proj.single_img_radar_proj_smart_index(yaml_path, index=cam, vis_flag=False, output_dir=output_img_path)
            
    # bbx2cam
    if flags["bbx2cam"]:
        bbx2cam_proj = ProjBBX2Cam(line_width=2)
        for frame in range(start_frame, end_frame + 1):
            for cam in range(0, 4):
                img_path = f"{input_root}/{frame:06}_camera{cam}.png"
                yaml_path = f"{input_root}/{frame:06}.yaml"
                output_img_path = f"{output_root}/bbx2cam/camera{cam}"
                bbx2cam_proj.single_img_multi_bbx_proj(img_path, yaml_path, save_path=output_img_path, cam_key=f"camera{cam}", bbx_keys=classes, vis_flag=False)
    
    # radar2lidar
    if flags["radar2lidar"]:
        lidar_path = f"{input_root}/{single_frame:06}_lidar0.pcd"
        radar_path = f"{input_root}/{single_frame:06}_radar0.pcd"
        yaml_file = f"{input_root}/{single_frame:06}.yaml"
        output_img_path = f"{output_root}/radar2lidar"

        if not os.path.isdir(f"{output_root}/radar2lidar"):
            os.makedirs(f"{output_root}/radar2lidar")

        radar_keys = ["radar_pose0","radar_pose1","radar_pose2","radar_pose3"]
        radar2lidar_proj = ProjLidar2Lidar(point_size=1.0)
        radar2lidar_proj.single_radar2lidar(lidar_path, radar_path, yaml_file, config_key1="lidar_pose0", config_key2=radar_keys, save_path=f"{output_img_path}/{single_frame:06}.pcd", vis_flag=False)

    # bbx2lidar
    if flags["bbx2lidar"]:
        lidar_path = f"{input_root}/{single_frame:06}_lidar0.pcd"
        yaml_file = f"{input_root}/{single_frame:06}.yaml"
        output_img_path = f"{output_root}/bbx2lidar"
    
    if not os.path.isdir(output_img_path):
        os.makedirs(output_img_path)

    bbx2lidar_proj = ProjBBX2Lidar()
    bbx2lidar_proj.proj_bbx2lidar(yaml_file, lidar_path, bbx_classes=classes, lidar_key="lidar_pose0", save_path=f"{output_img_path}/{single_frame:06}.png", vis_Flag=True)

if __name__ == "__main__":
    # test1_batch_lidar2cam()
    # test2_single_radar2lidar()
    # test3_single_bbx2cam()
    # test4_single_radar2lidar_batch()
    # test5_single_bbx2lidar()
    input_root = "/media/carma/ui_4/data_transfer/data_dumping/radar_dataset/town05_intersection3_4cam_radar/-125"
    output_root = "/home/carma/dg/results_new/town05_intersection3_4cam_radar"
    classes = ["cars", "trucks","pedestrians", "cyclists"]
    batch_start_frame = 250
    batch_end_frame = 300
    single_frame = 148
    flags = {"lidar2cam": False, "radar2cam": False, "bbx2cam": True, "radar2lidar": True, "bbx2lidar": True}
    test_many(input_root, output_root, classes, batch_start_frame, batch_end_frame, single_frame, flags)

