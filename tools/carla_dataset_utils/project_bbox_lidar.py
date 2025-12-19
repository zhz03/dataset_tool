# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import time

import cv2
import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm

from tools.carla_dataset_utils import box_utils
from tools.carla_dataset_utils import common_utils

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def bbx2linset(bbx_corner, order='hwl', color=(0, 1, 0)):
    """
    Convert the torch tensor bounding box to o3d lineset for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    line_set : list
        The list containing linsets.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner,
                                                   order)

    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [list(color) for _ in range(len(lines))]
    bbx_linset = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbx)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bbx_linset.append(line_set)

    return bbx_linset


def bbx2oabb(bbx_corner, order='hwl', color=(0, 0, 1)):
    """
    Convert the torch tensor bounding box to o3d oabb for visualization.

    Parameters
    ----------
    bbx_corner : torch.Tensor
        shape: (n, 8, 3).

    order : str
        The order of the bounding box if shape is (n, 7)

    color : tuple
        The bounding box color.

    Returns
    -------
    oabbs : list
        The list containing all oriented bounding boxes.
    """
    if not isinstance(bbx_corner, np.ndarray):
        bbx_corner = common_utils.torch_tensor_to_numpy(bbx_corner)

    if len(bbx_corner.shape) == 2:
        bbx_corner = box_utils.boxes_to_corners_3d(bbx_corner,
                                                   order)
    oabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        oabb = tmp_pcd.get_oriented_bounding_box()
        oabb.color = color
        oabbs.append(oabb)

    return oabbs


def bbx2aabb(bbx_center, order):
    """
    Convert the torch tensor bounding box to o3d aabb for visualization.

    Parameters
    ----------
    bbx_center : torch.Tensor
        shape: (n, 7).

    order: str
        hwl or lwh.

    Returns
    -------
    aabbs : list
        The list containing all o3d.aabb
    """
    if not isinstance(bbx_center, np.ndarray):
        bbx_center = common_utils.torch_tensor_to_numpy(bbx_center)
    bbx_corner = box_utils.boxes_to_corners_3d(bbx_center, order)

    aabbs = []

    for i in range(bbx_corner.shape[0]):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(bbx)

        aabb = tmp_pcd.get_axis_aligned_bounding_box()
        aabb.color = (0, 0, 1)
        aabbs.append(aabb)

    return aabbs


def linset_assign_list(vis,
                       lineset_list1,
                       lineset_list2,
                       update_mode='update'):
    """
    Associate two lists of lineset.

    Parameters
    ----------
    vis : open3d.Visualizer
    lineset_list1 : list
    lineset_list2 : list
    update_mode : str
        Add or update the geometry.
    """
    for j in range(len(lineset_list1)):
        index = j if j < len(lineset_list2) else -1
        lineset_list1[j] = \
            lineset_assign(lineset_list1[j],
                                     lineset_list2[index])
        if update_mode == 'add':
            vis.add_geometry(lineset_list1[j])
        else:
            vis.update_geometry(lineset_list1[j])


def lineset_assign(lineset1, lineset2):
    """
    Assign the attributes of lineset2 to lineset1.

    Parameters
    ----------
    lineset1 : open3d.LineSet
    lineset2 : open3d.LineSet

    Returns
    -------
    The lineset1 object with 2's attributes.
    """

    lineset1.points = lineset2.points
    lineset1.lines = lineset2.lines
    lineset1.colors = lineset2.colors

    return lineset1


def color_encoding(intensity, mode='constant'):
    """
    Encode the single-channel intensity to 3-channel RGB color.

    Parameters
    ----------
    intensity : np.ndarray
        Lidar intensity, shape (n,).

    mode : str
        The color rendering mode. Supported: 'intensity', 'z-value', 'constant'.

    Returns
    -------
    color : np.ndarray
        Encoded Lidar color, shape (n, 3).
    """
    assert mode in ['intensity', 'z-value', 'constant']

    if mode == 'intensity':
        intensity_col = np.clip(1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 50)), 0, 1)
        int_color = np.c_[
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
            np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]
    
    elif mode == 'z-value':
        min_value, max_value = -1.5, 0.5
        norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        colors = m.to_rgba(intensity)
        int_color = colors[:, :3]  # Drop alpha channel

    elif mode == 'constant':
        int_color = np.ones((intensity.shape[0], 3))
        int_color[:, 0] *= 0.8  # R
        int_color[:, 1] *= 0.8  # G
        int_color[:, 2] *= 0.0  # B


    return int_color



def visualize_single_sample_output_gt(pred_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis=True,
                                      save_path='',
                                      mode='constant'):
    """
    Visualize the prediction, groundtruth with point cloud together.

    Parameters
    ----------
    pred_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.

    mode : str
        Color rendering mode.
    """

    def custom_draw_geometry(pcd, pred, gt):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        opt = vis.get_render_option()
        opt.background_color = np.asarray([1,1,1])
        opt.point_size = 1.0

        vis.add_geometry(pcd)
        for ele in pred:
            vis.add_geometry(ele)
        for ele in gt:
            vis.add_geometry(ele)

        vis.run()
        vis.destroy_window()

    if len(pcd.shape) == 3:
        pcd = pcd[0]
    origin_lidar = pcd
    if not isinstance(pcd, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(pcd)

    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)
    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    oabbs_pred = bbx2oabb(pred_tensor, color=(1, 0, 0))
    oabbs_gt = bbx2oabb(gt_tensor, color=(0, 1, 0))

    visualize_elements = [o3d_pcd] + oabbs_pred + oabbs_gt
    if show_vis:
        custom_draw_geometry(o3d_pcd, oabbs_pred, oabbs_gt)
    if save_path:
        save_o3d_visualization(visualize_elements, save_path)


def visualize_single_sample_output_bev(pred_box, gt_box, pcd, dataset,
                                       show_vis=True,
                                       save_path=''):
    """
    Visualize the prediction, groundtruth with point cloud together in
    a bev format.

    Parameters
    ----------
    pred_box : torch.Tensor
        (N, 4, 2) prediction.

    gt_box : torch.Tensor
        (N, 4, 2) groundtruth bbx

    pcd : torch.Tensor
        PointCloud, (N, 4).

    show_vis : bool
        Whether to show visualization.

    save_path : str
        Save the visualization results to given path.
    """

    if not isinstance(pcd, np.ndarray):
        pcd = common_utils.torch_tensor_to_numpy(pcd)
    if pred_box is not None and not isinstance(pred_box, np.ndarray):
        pred_box = common_utils.torch_tensor_to_numpy(pred_box)
    if gt_box is not None and not isinstance(gt_box, np.ndarray):
        gt_box = common_utils.torch_tensor_to_numpy(gt_box)

    ratio = dataset.params["preprocess"]["args"]["res"]
    L1, W1, H1, L2, W2, H2 = dataset.params["preprocess"]["cav_lidar_range"]
    bev_origin = np.array([L1, W1]).reshape(1, -1)
    # (img_row, img_col)
    bev_map = dataset.project_points_to_bev_map(pcd, ratio)
    # (img_row, img_col, 3)
    bev_map = \
        np.repeat(bev_map[:, :, np.newaxis], 3, axis=-1).astype(np.float32)
    bev_map = bev_map * 255

    if pred_box is not None:
        num_bbx = pred_box.shape[0]
        for i in range(num_bbx):
            bbx = pred_box[i]

            bbx = ((bbx - bev_origin) / ratio).astype(int)
            bbx = bbx[:, ::-1]
            cv2.polylines(bev_map, [bbx], True, (0, 0, 255), 1)

    if gt_box is not None and len(gt_box):
        for i in range(gt_box.shape[0]):
            bbx = gt_box[i][:4, :2]
            bbx = (((bbx - bev_origin)) / ratio).astype(int)
            bbx = bbx[:, ::-1]
            cv2.polylines(bev_map, [bbx], True, (255, 0, 0), 1)

    if show_vis:
        plt.axis("off")
        plt.imshow(bev_map)
        plt.show()
    if save_path:
        plt.axis("off")
        plt.imshow(bev_map)
        plt.savefig(save_path)


def visualize_single_sample_dataloader(batch_data,
                                       o3d_pcd,
                                       order,
                                       key='origin_lidar',
                                       visualize=False,
                                       save_path='',
                                       oabb=False,
                                       mode='constant'):
    """
    Visualize a single frame of a single CAV for validation of data pipeline.

    Parameters
    ----------
    o3d_pcd : o3d.PointCloud
        Open3d PointCloud.

    order : str
        The bounding box order.

    key : str
        origin_lidar for late fusion and stacked_lidar for early fusion.

    visualize : bool
        Whether to visualize the sample.

    batch_data : dict
        The dictionary that contains current timestamp's data.

    save_path : str
        If set, save the visualization image to the path.

    oabb : bool
        If oriented bounding box is used.
    """

    origin_lidar = batch_data[key]
    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    # we only visualize the first cav for single sample
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)

    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    object_bbx_center = batch_data['object_bbx_center']
    object_bbx_mask = batch_data['object_bbx_mask']
    object_bbx_center = object_bbx_center[object_bbx_mask == 1]

    aabbs = bbx2linset(object_bbx_center, order) if not oabb else \
        bbx2oabb(object_bbx_center, order)
    visualize_elements = [o3d_pcd] + aabbs
    if visualize:
        o3d.visualization.draw_geometries(visualize_elements)

    if save_path:
        save_o3d_visualization(visualize_elements, save_path)

    return o3d_pcd, aabbs


def visualize_inference_sample_dataloader(pred_box_tensor,
                                          gt_box_tensor,
                                          origin_lidar,
                                          o3d_pcd,
                                          mode='constant'):
    """
    Visualize a frame during inference for video stream.

    Parameters
    ----------
    pred_box_tensor : torch.Tensor
        (N, 8, 3) prediction.

    gt_box_tensor : torch.Tensor
        (N, 8, 3) groundtruth bbx

    origin_lidar : torch.Tensor
        PointCloud, (N, 4).

    o3d_pcd : open3d.PointCloud
        Used to visualize the pcd.

    mode : str
        lidar point rendering mode.
    """

    if not isinstance(origin_lidar, np.ndarray):
        origin_lidar = common_utils.torch_tensor_to_numpy(origin_lidar)
    # we only visualize the first cav for single sample
    if len(origin_lidar.shape) > 2:
        origin_lidar = origin_lidar[0]
    # this is for 2-stage origin lidar, it has different format
    if origin_lidar.shape[1] > 4:
        origin_lidar = origin_lidar[:, 1:]

    origin_lidar_intcolor = \
        color_encoding(origin_lidar[:, -1] if mode == 'intensity'
                       else origin_lidar[:, 2], mode=mode)

    if not isinstance(pred_box_tensor, np.ndarray):
        pred_box_tensor = common_utils.torch_tensor_to_numpy(pred_box_tensor)
    if not isinstance(gt_box_tensor, np.ndarray):
        gt_box_tensor = common_utils.torch_tensor_to_numpy(gt_box_tensor)

    # left -> right hand
    origin_lidar[:, :1] = -origin_lidar[:, :1]

    o3d_pcd.points = o3d.utility.Vector3dVector(origin_lidar[:, :3])
    o3d_pcd.colors = o3d.utility.Vector3dVector(origin_lidar_intcolor)

    gt_o3d_box = bbx2linset(gt_box_tensor, order='hwl', color=(0, 1, 0))
    pred_o3d_box = bbx2linset(pred_box_tensor, color=(1, 0, 0))

    return o3d_pcd, pred_o3d_box, gt_o3d_box


def visualize_sequence_dataloader(dataloader, order, color_mode='constant'):
    """
    Visualize the batch data in animation.

    Parameters
    ----------
    dataloader : torch.Dataloader
        Pytorch dataloader

    order : str
        Bounding box order(N, 7).

    color_mode : str
        Color rendering mode.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().show_coordinate_frame = True

    # used to visualize lidar points
    vis_pcd = o3d.geometry.PointCloud()
    # used to visualize object bounding box, maximum 50
    vis_aabbs = []
    for _ in range(50):
        vis_aabbs.append(o3d.geometry.LineSet())

    while True:
        for i_batch, sample_batched in enumerate(dataloader):
            print(i_batch)
            pcd, aabbs = \
                visualize_single_sample_dataloader(sample_batched['ego'],
                                                   vis_pcd,
                                                   order,
                                                   mode=color_mode)
            if i_batch == 0:
                vis.add_geometry(pcd)
                for i in range(len(vis_aabbs)):
                    index = i if i < len(aabbs) else -1
                    vis_aabbs[i] = lineset_assign(vis_aabbs[i], aabbs[index])
                    vis.add_geometry(vis_aabbs[i])

            for i in range(len(vis_aabbs)):
                index = i if i < len(aabbs) else -1
                vis_aabbs[i] = lineset_assign(vis_aabbs[i], aabbs[index])
                vis.update_geometry(vis_aabbs[i])

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)

    vis.destroy_window()


def save_o3d_visualization(element, save_path):
    """
    Save the open3d drawing to folder.

    Parameters
    ----------
    element : list
        List of o3d.geometry objects.

    save_path : str
        The save path.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in range(len(element)):
        vis.add_geometry(element[i])
        vis.update_geometry(element[i])

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(save_path)
    vis.destroy_window()


def visualize_bev(batch_data):
    bev_input = batch_data["processed_lidar"]["bev_input"]
    label_map = batch_data["label_dict"]["label_map"]
    if not isinstance(bev_input, np.ndarray):
        bev_input = common_utils.torch_tensor_to_numpy(bev_input)

    if not isinstance(label_map, np.ndarray):
        label_map = label_map[0].numpy() if not label_map[0].is_cuda else \
            label_map[0].cpu().detach().numpy()

    if len(bev_input.shape) > 3:
        bev_input = bev_input[0, ...]

    plt.matshow(np.sum(bev_input, axis=0))
    plt.axis("off")
    plt.matshow(label_map[0, :, :])
    plt.axis("off")
    plt.show()


def draw_box_plt(boxes_dec, ax, color=None, linewidth_scale=1.0):
    """
    draw boxes in a given plt ax
    :param boxes_dec: (N, 5) or (N, 7) in metric
    :param ax:
    :return: ax with drawn boxes
    """
    if not len(boxes_dec)>0:
        return ax
    boxes_np= boxes_dec
    if not isinstance(boxes_np, np.ndarray):
        boxes_np = boxes_np.cpu().detach().numpy()
    if boxes_np.shape[-1]>5:
        boxes_np = boxes_np[:, [0, 1, 3, 4, 6]]
    x = boxes_np[:, 0]
    y = boxes_np[:, 1]
    dx = boxes_np[:, 2]
    dy = boxes_np[:, 3]

    x1 = x - dx / 2
    y1 = y - dy / 2
    x2 = x + dx / 2
    y2 = y + dy / 2
    theta = boxes_np[:, 4:5]
    # bl, fl, fr, br
    corners = np.array([[x1, y1],[x1,y2], [x2,y2], [x2, y1]]).transpose(2, 0, 1)
    new_x = (corners[:, :, 0] - x[:, None]) * np.cos(theta) + (corners[:, :, 1]
              - y[:, None]) * (-np.sin(theta)) + x[:, None]
    new_y = (corners[:, :, 0] - x[:, None]) * np.sin(theta) + (corners[:, :, 1]
              - y[:, None]) * (np.cos(theta)) + y[:, None]
    corners = np.stack([new_x, new_y], axis=2)
    for corner in corners:
        ax.plot(corner[[0,1,2,3,0], 0], corner[[0,1,2,3,0], 1], color=color, linewidth=0.5*linewidth_scale)
        # draw front line (
        ax.plot(corner[[2, 3], 0], corner[[2, 3], 1], color=color, linewidth=2*linewidth_scale)
    return ax


def draw_points_boxes_plt(pc_range, points=None, boxes_pred=None, boxes_gt=None, save_path=None,
                          points_c='y.', bbox_gt_c='green', bbox_pred_c='red', return_ax=False, ax=None):
    if ax is None:
        ax = plt.figure(figsize=(15, 6)).add_subplot(1, 1, 1)
        ax.set_aspect('equal', 'box')
        ax.set(xlim=(pc_range[0], pc_range[3]),
               ylim=(pc_range[1], pc_range[4]))
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], points_c, markersize=0.1)
    if (boxes_gt is not None) and len(boxes_gt)>0:
        ax = draw_box_plt(boxes_gt, ax, color=bbox_gt_c)
    if (boxes_pred is not None) and len(boxes_pred)>0:
        ax = draw_box_plt(boxes_pred, ax, color=bbox_pred_c)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig(save_path)
    if return_ax:
        return ax
    plt.close()

import open3d as o3d
import numpy as np
import yaml
import re
import yaml
import os
import math


def load_yaml(file, opt=None):
    """
    Load yaml file and return a dictionary.

    Parameters
    ----------
    file : string
        yaml file path.

    opt : argparser
         Argparser.
    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 'config.yaml')

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param


def load_pcd(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    # pcd.points is o3d.utility.Vector3dVector
    points_xyz = np.asarray(pcd.points)  # shape (N, 3)

    # If there's intensity, sometimes it's in pcd.colors or separate channels,
    # depends on how the PCD is formatted. For simplicity, we assume we only have xyz.
    # If you do have intensity, you might store it in a separate array.

    return points_xyz


def parse_vehicle_bbox(vehicle_info):
    """
    Parse vehicle info into bounding box representation [x, y, z, h, w, l, yaw].
    vehicle_info: dict with keys 'location', 'extent', 'angle'
    """
    location = vehicle_info['location']  # [x_world, y_world, z_world]
    extent = vehicle_info['extent']     # [l, w, h] (half-size or full-size depends on dataset)
    angle = vehicle_info['angle']       # [roll, yaw, pitch]
    center = vehicle_info['center']     # [x, y, z]

    # Extract values
    x, y, z = location
    cx, cy, cz = center
    l, w, h = extent
    roll, yaw, pitch = angle  # YAML order: roll, yaw, pitch

    # in carla the bbx center is not ground
    # z = z + cz / 2
    # Convert yaw to radians
    yaw = np.deg2rad(yaw)

    # Double the dimensions (assumes half-extent is provided)
    l *= 2
    w *= 2
    h *= 2

    return np.array([x, y, z, h, w, l, yaw], dtype=np.float32)

def euler_to_rot(roll, yaw, pitch):
    """Build a 3x3 rotation matrix given roll, yaw, pitch in radians."""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    # Roll -> Yaw -> Pitch (Z-Y-X order)
    return Rz @ Ry @ Rx


def invert_pose(lidar_pose):
    """
    lidar_pose: [x, y, z, roll, yaw, pitch] in (left-handed) world frame (degrees)
    return: 4x4 transform from WORLD -> LIDAR in a right-handed convention
    """
    x, y, z, roll_deg, yaw_deg, pitch_deg = lidar_pose

    # Convert degrees to radians
    roll = np.deg2rad(roll_deg)
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    # Build forward transform T (world -> lidar)
    T = np.eye(4)
    R = euler_to_rot(roll, yaw, pitch)  # Use the updated euler_to_rot
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    # Invert to get T_lidar_world
    T_inv = np.linalg.inv(T)
    return T_inv

def transform_bbox_world_to_lidar(bbox_world, T_inv):
    """
    Transform a 3D bounding box from the world frame to the LiDAR frame 
    using a full 4x4 matrix approach.

    bbox_world: np.array [x, y, z, h, w, l, yaw]
       (h, w, l) are the box dimensions, yaw is about +Z in world frame.
    T_inv: np.array (4,4)
       The transform from WORLD to LIDAR coordinates.

    Returns
    -------
    bbox_lidar : np.array [x_l, y_l, z_l, h, w, l, yaw_l]
       The bounding box in LiDAR coordinates (same dimensions).
    """
    x, y, z, h, w, l, yaw = bbox_world

    # 1) Build 4x4 pose of the box in the WORLD frame
    T_box_world = np.eye(4, dtype=np.float32)

    # Translation: place the box center at (x, y, z)
    T_box_world[0, 3] = x
    T_box_world[1, 3] = y
    T_box_world[2, 3] = z

    # Rotation: rotate about Z by 'yaw'
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    Rz = np.array([
        [ cos_y, -sin_y, 0],
        [ sin_y,  cos_y, 0],
        [     0,      0, 1]
    ], dtype=np.float32)
    T_box_world[0:3, 0:3] = Rz

    # 2) Transform box pose into the LIDAR frame
    T_box_lidar = T_inv @ T_box_world

    # 3) Extract the new center from T_box_lidar
    x_l = T_box_lidar[0, 3]
    y_l = T_box_lidar[1, 3]
    z_l = T_box_lidar[2, 3]

    # 4) Extract the new yaw from T_box_lidar's rotation portion
    #    (assuming no pitch/roll for the bounding box, so yaw is arctan2 of first column)
    R_lidar = T_box_lidar[0:3, 0:3]
    # yaw_l = arctan2 of (row=1,col=0) over (row=0,col=0)
    yaw_l = np.arctan2(R_lidar[1, 0], R_lidar[0, 0])

    # Keep dimensions h, w, l the same
    bbox_lidar = np.array([x_l, y_l, z_l, h, w, l, yaw_l], dtype=np.float32)
    return bbox_lidar

def main(yaml_file, pcd_file, sensor_type, sensor_id):
    # (1) Load data
    data = load_yaml(yaml_file)
    if "lidar_pose0" in data.keys():
        lidar_pose_key = "lidar_pose0"
    elif "lidar_pose" in data.keys():
        lidar_pose_key = "lidar_pose"
    else:
        raise ValueError("No lidar pose found in yaml file")
    lidar_pose = data[lidar_pose_key]  # LiDAR pose in the world frame [x, y, z, roll, pitch, yaw]

    if "cars" in data.keys():
        vehicles = data.get('cars', {})
    elif "vehicles" in data.keys():
        vehicles = data.get('vehicles', {})
    else:
        raise ValueError("No vehicles found in yaml file")

    # Load LiDAR points (already in LiDAR frame)
    pcd_xyz = load_pcd(pcd_file)  # Shape: (N, 3)

    # (2) Parse bounding boxes in world frame
    unit_types = {'cars','cyclists','pedestrians','trucks'}
    units = {}
    for unit_type in unit_types:
        units.update(data.get(unit_type, {}))
    
    bboxes = []
    for vid, unit_info in units.items():
        # parse_vehicle_bbox could return [x, y, z, h, w, l, yaw]
        bbx = parse_vehicle_bbox(unit_info)
        bboxes.append(bbx)
    bboxes = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 7), dtype=np.float32)

    # (3) Build transform from WORLD to LIDAR frame
    T_inv = invert_pose(lidar_pose)  # shape (4,4)

    # (4) Transform each bounding box via matrix multiplication
    for i in range(bboxes.shape[0]):
        bboxes[i, :] = transform_bbox_world_to_lidar(bboxes[i, :], T_inv)

    # (5) Prepare data for visualization
    object_bbx_mask = np.ones((bboxes.shape[0],), dtype=np.bool_)
    lidar_points = np.hstack([pcd_xyz, np.ones((pcd_xyz.shape[0], 1))])  # dummy intensity=1

    sample_batched = {
        'ego': {
            'origin_lidar': lidar_points,     # (N, 4)
            'object_bbx_center': bboxes,      # (M, 7)
            'object_bbx_mask': object_bbx_mask
        }
    }

    # (6) Visualize bounding boxes and LiDAR points in Open3D
    vis_pcd = o3d.geometry.PointCloud()
    visualize_single_sample_dataloader(
        sample_batched['ego'],
        o3d_pcd=vis_pcd,
        order='hwl',   # box is [x, y, z, h, w, l, yaw]
        visualize=True,
        mode='constant'
    )



def main_with_args(root_dir, agent, frame, sensor_type, sensor_id):
    """
    Build the file paths from root_dir, agent, frame,
    then call the original 'main' function.
    """
    agent_str = str(agent)
    frame_str = f"{int(frame):06}"  # e.g., 72 -> "000072"

    yaml_file = f"{root_dir}/{agent_str}/{frame_str}.yaml"
    pcd_file  = f"{root_dir}/{agent_str}/{frame_str}_" + sensor_type + str(sensor_id) + ".pcd"

    main(yaml_file, pcd_file, sensor_type, sensor_id)

def test1():
    # root_dir = '/data1/sensor_config_data/testset/v2xset/2021_08_20_21_48_35'
    # root_dir = '/home/handsomeyun/Yun/Multi-Mod_Sensor_Config_Lib/data_dumping/town10/'
    root_dir = '/media/carma/aebdc025-05c3-40fe-a0e9-f424cfe2ae03/home/mobility/data_dumping/radar_dataset/train/test_town04'
    agent    = -125
    frame    = 41
    sensor_type = "lidar"
    sensor_id = 0

    main_with_args(root_dir, agent, frame)

def test2():
    yaml_file = "data_examples/test_town04_05/-125/000031.yaml"
    pcd_file = "data_examples/test_town04_05/-125/000031_lidar0.pcd"
    main(yaml_file, pcd_file)

if __name__ == "__main__":
    test2()