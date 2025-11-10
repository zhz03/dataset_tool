def decode_yaml(yaml_file):

    yaml_param = load_yaml(yaml_file)
    # get the key of yaml_param
    key_list = list(yaml_param.keys())
    print("key:",key_list)

    lidar_key_list = []
    camera_key_list = []
    # get the lidar_pose from yaml_param
    for key in key_list:
        if "lidar_pose" in key:
            lidar_key_list.append(key)
        if "cam" in key:
            camera_key_list.append(key)
    
    lidar_pose_list = []
    for key in lidar_key_list:
        lidar_pose = yaml_param[key]
        lidar_pose_list.append(lidar_pose)

    camera_list = []
    for key in camera_key_list:
        camera_pose = yaml_param[key]
        camera_list.append(camera_pose)

    if "vehicles" in yaml_param:
        vehicle_dict = yaml_param["vehicles"]
    else:
        vehicle_dict = {}
    if "pedestrians" in yaml_param:
        pedestrian_dict = yaml_param["pedestrians"]
    else:
        pedestrian_dict = {}


    return lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict