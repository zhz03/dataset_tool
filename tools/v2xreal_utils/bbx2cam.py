import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

from tools.carla_dataset_utils.bbx_projection import decode_yaml
from tools.carla_dataset_utils.box_utils import boxes_to_corners_3d
from tools.carla_dataset_utils.project_bbox_lidar import parse_vehicle_bbox
from tools.v2xreal_utils.bbx2lidar import invert_pose, transform_bbox_world_to_lidar


BOX_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _draw_projected_box(
    image: np.ndarray,
    pixels: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw a projected bounding box on the image."""
    if not np.any(mask):
        return

    for start, end in BOX_EDGES:
        if not (mask[start] and mask[end]):
            continue
        pt1 = pixels[start]
        pt2 = pixels[end]
        if not (np.all(np.isfinite(pt1)) and np.all(np.isfinite(pt2))):
            continue
        p1 = int(round(pt1[0])), int(round(pt1[1]))
        p2 = int(round(pt2[0])), int(round(pt2[1]))
        cv2.line(image, p1, p2, color, thickness, lineType=cv2.LINE_AA)


def _project_lidar_corners_to_image(
    corners_lidar: np.ndarray,
    camera_extrinsic: np.ndarray,
    camera_intrinsic: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project lidar-frame corners into the image plane.

    Args:
        corners_lidar: Array of shape (N, 8, 3) in the lidar coordinate system.
        camera_extrinsic: 4x4 matrix from YAML.
        camera_intrinsic: 3x3 intrinsic matrix.
    Returns:
        pixels: (N, 8, 2) projected pixel coordinates (NaN for invalid).
        mask: (N, 8) boolean mask indicating visible corners.
    """
    if corners_lidar.size == 0:
        return np.zeros((0, 8, 2), dtype=np.float32), np.zeros((0, 8), dtype=bool)

    num_boxes = corners_lidar.shape[0]
    corners_flat = corners_lidar.reshape(-1, 3)
    corners_hom = np.hstack([corners_flat, np.ones((corners_flat.shape[0], 1), dtype=np.float32)])

    lidar_to_camera = np.linalg.inv(camera_extrinsic)
    corners_cam = (lidar_to_camera @ corners_hom.T).T[:, :3]

    depths = corners_cam[:, 2]
    depth_mask = depths > 0.0

    projected = (camera_intrinsic @ corners_cam.T).T
    projected[:, 0] = np.divide(projected[:, 0], projected[:, 2], out=np.full_like(projected[:, 0], np.nan), where=depth_mask)
    projected[:, 1] = np.divide(projected[:, 1], projected[:, 2], out=np.full_like(projected[:, 1], np.nan), where=depth_mask)

    pixels = projected[:, :2].reshape(num_boxes, 8, 2)
    mask = depth_mask.reshape(num_boxes, 8)
    return pixels, mask


def project_bounding_boxes(
    image_path: Union[str, Path],
    yaml_path: Union[str, Path],
    cam_index: int = 0,
    lidar_index: int = 0,
    output_dir: Union[str, Path, None] = None,
    line_color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    visualize: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Project 3D bounding boxes by first converting them to the lidar frame,
    then projecting to the camera image.
    """
    image_path = Path(image_path)
    yaml_path = Path(yaml_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    lidar_pose_list, camera_list, vehicle_dict, pedestrian_dict = decode_yaml(str(yaml_path))
    if not lidar_pose_list:
        raise ValueError(f"No lidar poses available in YAML: {yaml_path}")
    if lidar_index >= len(lidar_pose_list):
        raise IndexError(f"lidar_index {lidar_index} out of range (found {len(lidar_pose_list)})")

    if not camera_list:
        raise ValueError(f"No camera entries available in YAML: {yaml_path}")
    if cam_index >= len(camera_list):
        raise IndexError(f"cam_index {cam_index} out of range (found {len(camera_list)})")

    camera_params = camera_list[cam_index]
    camera_extrinsic = np.asarray(camera_params["extrinsic"], dtype=np.float32)
    camera_intrinsic = np.asarray(camera_params["intrinsic"], dtype=np.float32)

    vehicles = vehicle_dict if vehicle_dict is not None else {}

    T_world_to_lidar = invert_pose(lidar_pose_list[lidar_index])

    bboxes_world = []
    vehicle_ids: List[str] = []
    for vid, vehicle in vehicles.items():
        try:
            bbox_world = parse_vehicle_bbox(vehicle)
        except KeyError as exc:
            print(f"Skipping vehicle {vid}: missing field {exc}")
            continue
        bboxes_world.append(bbox_world)
        vehicle_ids.append(str(vid))

    if bboxes_world:
        bboxes_world = np.stack(bboxes_world, axis=0)
    else:
        bboxes_world = np.zeros((0, 7), dtype=np.float32)

    bboxes_lidar = bboxes_world.copy()
    for i in range(bboxes_world.shape[0]):
        bboxes_lidar[i, :] = transform_bbox_world_to_lidar(bboxes_world[i, :], T_world_to_lidar)

    corners_lidar = boxes_to_corners_3d(bboxes_lidar, order="hwl") if bboxes_lidar.size else np.zeros((0, 8, 3), dtype=np.float32)
    pixels, depth_mask = _project_lidar_corners_to_image(corners_lidar, camera_extrinsic, camera_intrinsic)

    image_h, image_w = image.shape[:2]
    in_image_mask = np.zeros_like(depth_mask)
    if pixels.size:
        in_bounds = (
            (pixels[:, :, 0] >= 0) & (pixels[:, :, 0] < image_w) &
            (pixels[:, :, 1] >= 0) & (pixels[:, :, 1] < image_h)
        )
        in_image_mask = depth_mask & in_bounds

    for box_pixels, mask in zip(pixels, in_image_mask):
        _draw_projected_box(image, box_pixels, mask, color=line_color, thickness=thickness)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{image_path.stem}_bbx_{timestamp}.jpg"
        cv2.imwrite(str(output_file), image)
    else:
        output_file = None

    if visualize:
        try:
            cv2.imshow("Projected Bounding Boxes", image)
            cv2.waitKey(0)
        finally:
            cv2.destroyAllWindows()

    result = {
        "image": image,
        "bbox_world": bboxes_world,
        "bbox_lidar": bboxes_lidar,
        "bbox_pixels": pixels,
        "bbox_visible_mask": in_image_mask,
        "lidar_pose_list": lidar_pose_list,
        "camera_params": camera_params,
        "vehicle_ids": vehicle_ids,
        "source_image": str(image_path),
        "source_yaml": str(yaml_path),
        "saved_path": str(output_file) if output_file else None,
    }
    return result


def test1() -> None:
    image_file = "/Users/zhaoliang/Documents/zhz03/github/dataset_tool/test_examples/v2x-real/vehicle_1/000030_cam1.jpeg"
    yaml_file = "/Users/zhaoliang/Documents/zhz03/github/dataset_tool/test_examples/v2x-real/vehicle_1/000030.yaml"
    output_dir = "/Users/zhaoliang/Documents/zhz03/github/dataset_tool/test_examples/v2x-real/vehicle_1/output"

    result = project_bounding_boxes(
        image_path=image_file,
        yaml_path=yaml_file,
        cam_index=0,
        lidar_index=0,
        output_dir=output_dir,
        visualize=False,
    )
    print(f"Projected {len(result['vehicle_ids'])} vehicles to {result['source_image']}.")


if __name__ == "__main__":
    test1()


