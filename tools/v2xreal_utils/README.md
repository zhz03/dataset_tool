## V2X-Real Utility Scripts

This folder contains helper scripts for inspecting V2X-Real samples, converting 3D bounding boxes between coordinate frames, and projecting LiDAR data into camera images.

### Prerequisites
- Python 3.8+
- NumPy, OpenCV (`cv2`), Open3D, Matplotlib
- Access to the V2X-Real sample assets:
  - `*.yaml` calibration files decoded with `decode_yaml`
  - LiDAR frames in `*.bin` format (`XYZI` or `XYZIR` float layout)
  - JPEG camera frames

Install dependencies (adjust as needed):
```
pip install numpy opencv-python open3d matplotlib pyyaml
```

### Repository Layout
- `bbx2cam.py` – Projects vehicle bounding boxes into a camera image.
- `bbx2lidar.py` – Loads a LiDAR scan, converts vehicle bounding boxes into the LiDAR frame, and optionally visualizes them in Open3D.
- `lidar2cam.py` – Projects dense LiDAR points onto a camera image with depth-coloured overlays.

### Common Data Requirements
All scripts expect the directory structure used in `test_examples/v2x-real/...`, where every timestamp holds:
- `TIMESTAMP.yaml` – sensor poses and calibration
- `TIMESTAMP.bin` – synchronized LiDAR frame
- `TIMESTAMP_cam{N}.jpeg` – camera capture for camera index `N`

Update the hard-coded paths in each script or wrap the core functions in your own tooling if your dataset paths differ.

### Running the Scripts

#### Project bounding boxes to an image (`bbx2cam.py`)
```
python tools/v2xreal_utils/bbx2cam.py
```
The script runs `test1()`, which:
- Loads `000030.yaml` and the matching camera image.
- Converts vehicle boxes from world → LiDAR → camera.
- Draws the projected 3D boxes using `_draw_projected_box`.
- Saves a copy in `test_examples/v2x-real/.../output/`.

You can call the main utility directly:
```python
from tools.v2xreal_utils.bbx2cam import project_bounding_boxes

result = project_bounding_boxes(
    image_path=".../000030_cam1.jpeg",
    yaml_path=".../000030.yaml",
    cam_index=0,
    lidar_index=0,
    output_dir=".../output",   # Optional
    visualize=False            # True opens an OpenCV window
)
```
Returned fields include bounding boxes in world and LiDAR frames, the projected pixel coordinates, visibility masks, and the source paths.

#### Transform boxes to the LiDAR frame (`bbx2lidar.py`)
```
python tools/v2xreal_utils/bbx2lidar.py
```
`test1()` loads the LiDAR binary and YAML, converts all vehicle boxes into LiDAR coordinates, and visualizes them alongside the point cloud with Open3D (toggle via `visualize` argument).

Programmatic use:
```python
from tools.v2xreal_utils.bbx2lidar import main

sample = main(
    bin_file=".../000030.bin",
    yaml_file=".../000030.yaml",
    lidar_index=0,
    visualize=False  # True to launch Open3D viewer
)
ego_sample = sample["ego"]
```

#### Project LiDAR points onto an image (`lidar2cam.py`)
```
python tools/v2xreal_utils/lidar2cam.py
```
`lidar2cam_single_z()`:
- Loads LiDAR points and the selected camera calibration (`cam_index`).
- Transforms the LiDAR coordinates into the camera frame.
- Colours points according to depth (`Z`) with a Jet colormap.
- Blends the overlay on the camera image and writes the result to disk.

Adjust the file paths and `cam_index` to match your dataset.

### Notes
- `invert_pose` in `bbx2lidar.py` converts CARLA’s world pose into a homogeneous transform. Reuse it when building your own conversions.
- Both box utilities depend on `tools.carla_dataset_utils` for YAML parsing and box geometry. Ensure that package is importable (e.g., run from the repository root or add it to `PYTHONPATH`).
- Set `visualize=False` when running on headless machines to avoid Open3D/GUI requirements.

### Troubleshooting
- **FileNotFoundError**: Verify the absolute paths passed to the functions; the sample code uses hard-coded absolute paths.
- **ValueError (LiDAR format)**: Ensure `.bin` files are floats laid out as `XYZI` or `XYZIR`.
- **OpenCV/Open3D windows not showing**: GUI calls require a local display. Disable visualization for remote/headless runs.


