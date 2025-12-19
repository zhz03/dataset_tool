# Radar Tooling

Utilities and sample data for inspecting cooperative V2X radar point
clouds stored as binary blobs of consecutive `float32` values in the
`[x, y, z, value]` format.

## Repository Contents

- `visualzie_4d_radar.py` – loads a `.bin` file, reshapes it to `(N, 4)`
  and plots the 3D positions colored by the fourth attribute (e.g.,
  intensity or radial velocity) using Matplotlib.
- `read_radar_bin.py` – lightweight CLI helper that simply reports how
  many `float32` values (data points) are present in a `.bin` file.
- `*.bin` – short radar captures you can use for testing the scripts.
  They already follow the `[x, y, z, value]` layout expected by the
  scripts.

## Requirements

- Python 3.8+
- NumPy
- Matplotlib (only needed for `visualzie_4d_radar.py`)

Set up a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or pip install numpy matplotlib
```

## Usage

Change into the radar tools directory:

```bash
cd /home/zhaoliang/zzl/dataset_tool/tools/radar
```

### Inspect a radar binary

```bash
python read_radar_bin.py v2x-radar-000001.bin
```

This prints the total number of `float32` entries so you can sanity-check
file sizes or verify conversions.

### Visualize radar points in 3D

```bash
python visualzie_4d_radar.py v2x-radar-000001.bin
```

The script loads the binary, reports the `(N, 4)` shape and opens an
interactive Matplotlib window. The first three columns are mapped to the
X/Y/Z axes, while the fourth column controls the color scale (Viridis by
default) so you can quickly spot intensity or Doppler variations.

## Tips

- Files whose length is not a multiple of four floats will trigger a
  warning; the visualizer discards any trailing remainder so the reshape
  still succeeds.
- Replace the `.bin` argument with your own radar captures as long as
  they share the same layout.
- Use Matplotlib's built-in tools (rotate, zoom, save) to inspect the
  point cloud from different viewpoints.


