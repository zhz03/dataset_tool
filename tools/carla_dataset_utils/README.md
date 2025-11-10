# CARLA Dataset Utilities

Utilities in this folder help visualize and debug multi-sensor data collected in the CARLA simulator. They provide:

- Transform helpers to move between LiDAR, camera, and world frames.
- Projection routines to overlay LiDAR points or 3D bounding boxes on camera images.
- Open3D viewers for point clouds and bounding boxes.
- YAML parsing helpers for CARLA log files.

All paths below are relative to `tools/carla_dataset_utils`.

## Prerequisites

- Python 3.8+ (tested with CARLA 0.9.x)
- Dependencies:
  - `numpy`, `scipy`
  - `opencv-python`, `Pillow`
  - `matplotlib`
  - `open3d`
  - `pyyaml`
  - `carla` (the Python API module)
  - `opencood` (only required by `project_bbox_lidar.py`)

Install with:

```bash
pip install -r ../../requirements.txt
```

Ensure the CARLA Python egg is on `PYTHONPATH` and that your data directory contains synchronized `camera*.png`, `lidar*.pcd`, and `*.yaml` files.

## Quick Start

1. Collect data with CARLA using the logging pipeline that produces:

   ```
   <capture_root>/<scene_id>/<frame_id>/
     camera0_000045.png
     lidar0_000045.pcd
     000045.yaml
   ```

   The YAML file must include `lidar_pose*`, `camera*`, `vehicles`, and `pedestrians` entries.

2. Activate your virtual environment and `cd` into the repository root:

   ```bash
   cd /Users/zhaoliang/Documents/zhz03/github/dataset_tool
   ```

3. Use the scripts below to visualize your data (examples use the sample dataset under `data_dumping/example/...`).

## Script Overview

| Script | Purpose |
| --- | --- |
| `project_lidar2cam.py` | Project LiDAR point clouds onto camera images. |
| `project_3dbbx2img.py` | Overlay 3D bounding boxes from YAML annotations onto images. |
| `bbx_projection.py` | YAML parsing and global→local coordinate conversion for bounding boxes. |
| `project_bbox_lidar.py` | Open3D visualizations of point clouds with predicted/GT boxes (adapted from OpenCOOD). |
| `box_utils.py` | Shared geometry utilities (intrinsics, box generation, projection helpers). |
| `vis_pcd.py` | Minimal Open3D viewer for `.pcd` files. |
| `vis_utils.py` | Combine point cloud and box visualization for a single frame. |

Each script includes test or demo functions at the bottom that show canonical usage; you can adapt these for your dataset.

## Detailed Usage

### `project_lidar2cam.py`

Projects LiDAR point clouds into camera space and saves colorized overlays.

- **Key functions**
  - `create_transformation(...)`: Build CARLA transforms (4×4 matrices).
  - `process_jpeg_to_array(...)` / `process_pcd_to_array(...)`: Load RGB images and `.pcd` point clouds.
  - `project_lidar_to_camera*`: Several visualization variants that color points by intensity, height, or lateral displacement.
  - `project_save_single_frame(...)`: End-to-end helper that reads an image, point cloud, and YAML metadata, then writes the overlay.

- **Run the demo**

  ```bash
  python project_lidar2cam.py
  ```

  `test5()` uses hard-coded paths; edit `img_file_path`, `pcd_file_path`, `yaml_file`, and `output_dir` to match your capture. Select camera/LiDAR indices with `cam_index` and `lidar_index` (0-based).

- **Typical workflow**
  1. Call `decode_yaml(yaml_file)` to get sensor arrays.
  2. Use `cam_index` to pick a camera, extract intrinsics (`camera_intrinsic`) and pose (`cords`).
  3. Use `lidar_index` to pick a LiDAR sensor pose.
  4. Load the paired `camera*.png` and `lidar*.pcd`.
  5. Invoke `project_lidar_to_camera2(...)` to render and save the overlay.

Ensure your YAML intrinsics follow CARLA’s format (3×3 matrix with principal point stored at `[0,2]` and `[1,2]`).

### `project_3dbbx2img.py`

Draws annotated 3D bounding boxes onto a camera image.

- **Dependencies:** `decode_yaml` for annotations and `box_utils` for geometry.
- **Important:** The import `from tools.carla_dataset_utils.proj_lidar2cam import ...` assumes this module is renamed to `proj_lidar2cam`. If the file name is `project_lidar2cam.py`, update the import accordingly before running.
- **Usage pattern:**
  1. Parse the YAML to obtain `vehicle_dict`.
  2. Convert each vehicle’s extent/pose to box coordinates.
  3. Transform points from world to camera space (`create_transformation` + intrinsics).
  4. Use OpenCV to draw image-space edges and save the result.

- **Run the sample:**

  ```bash
  python project_3dbbx2img.py
  ```

  `test_bbx()` demonstrates loading a frame, building boxes, and writing annotated images. Edit the hard-coded paths before executing.

### `bbx_projection.py`

Utility functions shared across the visualization scripts.

- `decode_yaml(yaml_file)`: Reads CARLA metadata via `tools.utils.yaml_utils.load_yaml`. Returns:
  - `lidar_pose_list`: List of `[x, y, z, roll, pitch, yaw]`.
  - `camera_list`: Each entry contains `cords`, `intrinsic`, and `extrinsic`.
  - `vehicle_dict`, `pedestrian_dict`: Object annotations.
- `global_to_local(global_pos, lidar_pose)`: Converts world coordinates into a LiDAR-centric frame given the sensor pose.

Use this module if you need sensor poses or per-object annotations outside the provided demos.

### `project_bbox_lidar.py`

Rich visualization suite adapted from OpenCOOD for inspecting predictions vs. ground truth in LiDAR space.

- Provides conversions from bounding box tensors to Open3D `LineSet`/`OrientedBoundingBox` objects.
- Supports several coloring modes (`intensity`, `z-value`, `constant`).
- Includes BEV rendering (`visualize_single_sample_output_bev`) that projects point clouds into bird’s-eye-view rasters.
- Requires tensor inputs from cooperative perception datasets (`torch.Tensor` or `numpy.ndarray`). Many helper functions expect OpenCOOD dataset wrappers (e.g., `dataset.project_points_to_bev_map`, `dataset.params[...]`).

This script is primarily for research workflows; adapt it if you want to visualize your own detection outputs.

### `box_utils.py`

Shared geometry helpers:

- `get_K(image_w, image_h, fov)` and `decode_wh(K)`: Build or parse camera intrinsics.
- `convert_to_box(...)` / `convert_carla_data_to_box(...)`: Translate CARLA annotations into center/extent/rotation dictionaries compatible with Open3D.
- `create_rotated_box` and `create_rotated_box_points`: Generate Open3D boxes or vertex arrays for further projection.
- `proj_points_2_img(...)`: Project 3D box corners to image space and draw edges via Pillow.
- `save_img(...)`: Convenience wrapper around `PIL.Image.save`.

These utilities are used by both the LiDAR→image and 3D box overlay pipelines; you can import them directly in custom scripts.

### `vis_pcd.py`

Lightweight point cloud viewer.

- `load_pcd2op3d(file_path)`: Reads `.pcd` files using `open3d.t`.
- `visualize_pcd(file_path)`: Opens an Open3D window and displays the point cloud with configurable background/point size.

Example:

```bash
python -c "from tools.carla_dataset_utils.vis_pcd import visualize_pcd; visualize_pcd('path/to/lidar0_000045.pcd')"
```

### `vis_utils.py`

Combines point cloud and bounding boxes in a single Open3D scene.

- `visualize_single_sample_data(yaml_file, pcd_file)` loads a frame, converts global vehicle boxes into LiDAR coordinates, and renders both the point cloud and boxes.
- Imports point cloud and geometry helpers from the same package. If the relative imports still point to `data_post_process.*`, update them to `tools.carla_dataset_utils.*`.

This is useful for quick qualitative checks of calibration and annotation quality.

## Data Expectations

The YAML file produced by the logging pipeline should follow the structure:

```yaml
lidar_pose_0: [x, y, z, roll, pitch, yaw]
camera0:
  cords: [x, y, z, roll, yaw, pitch]
  intrinsic: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  extrinsic: [...]
vehicles:
  "<id>":
    location: [x, y, z]
    angle: [roll, pitch, yaw]
    extent: [half_length, half_width, half_height]
pedestrians: {...}
```

Angles in YAML are in degrees. Intrinsics/extrinsics are expected to be CARLA’s native outputs.

## Tips & Troubleshooting

- **Coordinate conventions:** CARLA uses left-handed UE4 coordinates. The projection utilities swap axes (`x, y, z → y, -z, x`) before applying camera intrinsics.
- **Missing intensity channel:** `process_pcd_to_array` will synthesize zero intensities if your `.pcd` does not include them.
- **Large point clouds:** Set `dot_extent` in `project_lidar_to_camera*` to control point size and limit loops for faster drawing.
- **Module imports:** Several scripts reference absolute paths used during development. Adjust hard-coded directories and import paths before running in your environment.
- **Headless environments:** Open3D visualization functions require a display. Use `export PYOPENGL_PLATFORM=osmesa` with OSMesa builds for headless servers, or skip visualization by saving to disk.

## Extending the Utilities

- Wrap `project_save_single_frame` in a loop to process entire sequences.
- Use `create_rotated_box_points` and `proj_points_2_img` to draw custom annotations (e.g., pedestrians) on images.
- Adapt `project_bbox_lidar` for your detector outputs by feeding predicted/ground-truth tensors through `bbx2oabb` or `bbx2linset`.

For questions or improvements, contact the author listed in each script header or open an issue in this repository.


