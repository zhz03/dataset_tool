# -*- coding: utf-8 -*-
"""
Code description.
"""
# Author: Zhaoliang Zheng <zhz03@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import open3d as o3d
from data_post_process.vis_pcd import load_pcd2op3d
from data_post_process.box_utils import convert_carla_data_to_box, create_rotated_box
from data_post_process.bbx_projection import decode_yaml,global_to_local

def visualize_single_sample_data(yaml_file,pcd_file):
    try: 
        # load open3d settings
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # load pcd _file 
        pcd = load_pcd2op3d(pcd_file)
        vis.add_geometry(pcd)

        #　load yaml file object 
        lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict = decode_yaml(yaml_file)

        for key in vehicle_dict.keys():
            vehicle = vehicle_dict[key]
            location = vehicle["location"]
            angle =  vehicle["angle"]
            extent = vehicle["extent"]
            local_location = global_to_local(location, lidar_pose_list[0])
            position, scale, rotation = convert_carla_data_to_box(angle,extent,local_location)
            box = create_rotated_box(position, scale, rotation)
            vis.add_geometry(box)

        vis.run()

    finally:
        vis.destroy_window()

if __name__ == "__main__":
    yaml_file = "/home/zzl/zzl/Multi-Mod_Sensor_Config_Lib/data_dumping/example/2024_11_30_16_15_46/125/000017.yaml"
    pcd_file = "/home/zzl/zzl/Multi-Mod_Sensor_Config_Lib/data_dumping/example/2024_11_30_16_15_46/125/lidar0_000017.pcd"
    visualize_single_sample_data(yaml_file,pcd_file)
