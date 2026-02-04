## Online Point Tracking Without Ground Truth

This guide explains how to run DynOMo online to track points through a sequence without ground truth data.

## Overview

The online tracking system:
1. **Optimizes the scene online** - Runs DynOMo frame-by-frame to build a 4D Gaussian representation
2. **Tracks query points** - Tracks user-specified points through the sequence
3. **Visualizes results** - Creates videos showing tracked point trajectories

**No ground truth required** - Works with just RGB images and optional depth/poses.

---

## Quick Start

### 1. Prepare Your Data

Organize your data following the structure from `TEST_WITHOUT_GT_README.md`:

```
your_data/
└── sequence_name/
    ├── rgb/              # Required: RGB images
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    ├── depth/            # Optional: precomputed depth
    ├── intrinsics.txt    # Optional: camera intrinsics
    └── poses.txt         # Optional: camera poses
```

### 2. Configure Tracking

Edit `configs/custom/track_custom.py`:

```python
config = {
    'data': {
        'basedir': '/path/to/your/data',  # UPDATE THIS
        'desired_height': 360,
        'desired_width': 640,
        'online_depth': 'DepthAnythingV2',  # Use online depth
        # ...
    },
    'tracking': {
        'n_query_points': 1024,  # Grid of query points
        'traj_len': 10,  # Trajectory trail length
        # ...
    },
}
```

### 3. Run Online Tracking

```bash
# Full pipeline: optimize + track + visualize
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence
```

This will:
- Optimize the scene frame-by-frame (takes ~30 sec per frame)
- Extract trajectories for a grid of query points
- Create visualization videos

---

## Usage Modes

### Mode 1: Grid Tracking (Default)

Track a uniform grid of points across the frame:

```bash
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence
```

Number of points controlled by `config['tracking']['n_query_points']`.

### Mode 2: Custom Query Points

Track specific points of interest:

**Step 1:** Create query points interactively

```bash
python create_query_points.py \
    --sequence_dir /path/to/data/my_sequence \
    --output my_queries.json
```

This opens an interactive window:
- **Left click**: Add point
- **Right click**: Remove point
- **'g'**: Generate grid
- **'q'**: Save and quit

**Step 2:** Run tracking with custom points

```bash
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence \
    --query_points my_queries.json
```

### Mode 3: Auto-Generated Grid

Generate a grid programmatically:

```bash
python create_query_points.py \
    --sequence_dir /path/to/data/my_sequence \
    --grid 32 \
    --output grid_32x32.json
```

Creates a 32×32 grid of points (1024 total).

---

## Execution Modes

The tracking script supports different execution modes:

### Full Pipeline (Default)

Run optimization, tracking, and visualization:

```bash
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence \
    --mode full
```

### Optimization Only

Just optimize the scene (for later analysis):

```bash
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence \
    --mode optimize
```

### Tracking Only

Extract trajectories from previously optimized scene:

```bash
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence \
    --mode track
```

### Grid Visualization

Create dense grid visualization (flow-like):

```bash
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence \
    --mode grid
```

### Optical Flow

Compute and visualize optical flow:

```bash
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence \
    --mode flow
```

---

## Configuration Options

### Basic Settings

```python
config = {
    'data': {
        'basedir': '/path/to/data',
        'desired_height': 360,        # Lower for faster processing
        'desired_width': 640,
        'every_x_frame': 1,           # Process every N frames
        'online_depth': 'DepthAnythingV2',
        'load_embeddings': True,      # Enable for better tracking
        'embedding_dim': 64,          # Lower for faster processing
    },
}
```

### Tracking Settings

```python
config = {
    'tracking': {
        'n_query_points': 1024,       # Number of grid points
        'best_x': 1,                  # Track best of X Gaussians
        'vis_thresh': 0.5,            # Visibility threshold
        'traj_len': 10,               # Trajectory trail length
    },
}
```

### Optimization Settings

```python
config = {
    'tracking_obj': {
        'num_iters_init': 500,        # Iterations for first frame
        'num_iters': 60,              # Iterations per frame
    },
    'tracking_cam': {
        'num_iters': 40,              # Camera optimization iterations
    },
}
```

### Performance Presets

**Fast (lower quality, faster):**
```python
from configs.custom.track_custom import get_fast_config
config = get_fast_config()
```

**High Quality (slower, more accurate):**
```python
from configs.custom.track_custom import get_high_quality_config
config = get_high_quality_config()
```

---

## Output Files

After tracking completes, you'll find:

```
results/tracking/my_sequence/
├── config.json                          # Configuration used
├── query_points.json                    # Query points metadata
├── query_points_frame0.png              # Visualization of query points
├── my_sequence/
│   ├── params.npz                       # Optimized scene parameters
│   ├── tracked_trajectories.npz         # Tracked trajectories
│   ├── trajectory_visualization/        # Visualization videos
│   │   ├── tracked_points.mp4
│   │   ├── tracked_points_trails.mp4
│   │   └── frames/
│   ├── grid_points_vis/                 # Dense grid visualization
│   │   └── video.mp4
│   └── flow/                            # Optical flow (if enabled)
│       └── video.mp4
└── eval/                                # Additional visualizations
```

### Key Files

**tracked_trajectories.npz:**
```python
import numpy as np
data = np.load('tracked_trajectories.npz')
trajectories_2D = data['trajectories_2D']  # Shape: (N, T, 2)
trajectories_3D = data['trajectories_3D']  # Shape: (N, T, 3)
visibility = data['visibility']            # Shape: (N, T)
```

**query_points.json:**
```json
{
  "points": [[x1, y1], [x2, y2], ...],
  "normalized_points": [[nx1, ny1], ...],
  "metadata": {...}
}
```

---

## Performance Tips

### Speed vs Quality Tradeoffs

**For faster processing:**
- Reduce resolution: `desired_height=270, desired_width=480`
- Process fewer frames: `every_x_frame=2`
- Disable embeddings: `load_embeddings=False`
- Reduce iterations: `num_iters=30`
- Fewer query points: `n_query_points=256`

**For better quality:**
- Higher resolution: `desired_height=480, desired_width=640`
- Enable embeddings: `load_embeddings=True, embedding_dim=128`
- More iterations: `num_iters_init=1000, num_iters=100`
- More query points: `n_query_points=4096`
- Use best-of-K: `best_x=3`

### Memory Management

If you run out of memory:
- Reduce image size
- Disable embeddings
- Set `every_x_frame=2` or higher
- Close other applications
- Use a GPU with more VRAM

### Estimated Runtimes

On a RTX 3090 with default settings:
- **Optimization**: ~30-60 seconds per frame
- **Trajectory extraction**: ~5-10 seconds total
- **Visualization**: ~1 second per frame

For a 100-frame sequence:
- Total time: ~50-100 minutes

---

## Visualization Outputs

### 1. Trajectory Visualization

Shows tracked points with trails:
- Green points: Currently visible
- Colored trails: Recent trajectory (length = `traj_len`)
- Faded trails: Points that became occluded

### 2. Grid Visualization

Dense grid of tracked points across entire frame:
- Useful for understanding scene motion
- Similar to optical flow but with long-range tracking

### 3. Optical Flow

Frame-to-frame motion field:
- Color indicates direction
- Brightness indicates magnitude
- Saved as colored flow visualizations

---

## Advanced Usage

### Resume from Checkpoint

If tracking was interrupted:

```bash
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence \
    --mode track  # Skip optimization, use existing params
```

### Extract Custom Trajectories

Load optimized scene and query specific points:

```python
import torch
import numpy as np
from src.evaluate.trajectory_evaluator import TrajEvaluator

# Load evaluator
evaluator = TrajEvaluator(
    config=config,
    results_dir='results/tracking/my_sequence',
    primary_device='cuda:0'
)

# Query specific points
points = torch.tensor([[100, 200], [300, 400]], dtype=torch.float32)
start_time = torch.zeros(len(points), dtype=torch.long)

# Track
traj_2D, traj_3D, visibility, _ = evaluator.get_gs_traj_pts(
    start_pixels=points,
    start_time=start_time,
    start_pixels_normalized=False
)

# Save
np.save('custom_trajectories.npy', traj_2D.cpu().numpy())
```

### Batch Processing Multiple Sequences

```bash
for seq in sequence1 sequence2 sequence3; do
    python track_online_without_gt.py \
        --config configs/custom/track_custom.py \
        --sequence $seq \
        --output_dir results/batch_tracking/$seq
done
```

---

## Troubleshooting

### Issue: Optimization is very slow
**Solutions:**
- Use preset: `config = get_fast_config()`
- Reduce iterations in config
- Process every other frame: `every_x_frame=2`
- Reduce resolution

### Issue: Poor tracking quality
**Solutions:**
- Use preset: `config = get_high_quality_config()`
- Enable embeddings: `load_embeddings=True`
- Increase iterations
- Use best-of-K: `best_x=3`

### Issue: CUDA out of memory
**Solutions:**
- Reduce resolution: `desired_height=270`
- Disable embeddings: `load_embeddings=False`
- Process fewer points: `n_query_points=256`
- Set `every_x_frame=2`

### Issue: Points drift over time
**Solutions:**
- Enable embeddings for better correspondence
- Increase optimization iterations
- Check depth quality (consider using online depth)
- Verify camera intrinsics are correct

### Issue: Tracking fails at occlusions
**Solutions:**
- This is expected behavior - points become invisible when occluded
- Check visibility predictions in output
- Use multiple query points for redundancy

---

## Comparison with Ground Truth Systems

| Feature | With Ground Truth | Without Ground Truth (This System) |
|---------|-------------------|-----------------------------------|
| **Input Requirements** | RGB + GT Depth + GT Annotations | RGB only (optional depth/poses) |
| **Metrics** | PSNR, SSIM, Depth RMSE, TAP-Vid | Qualitative (visualization) |
| **Point Selection** | From GT annotations | User-specified or grid |
| **Validation** | Quantitative comparison | Visual inspection |
| **Use Cases** | Benchmarking, evaluation | Real-world data, custom videos |

---

## Examples

### Example 1: Track Object Motion

```bash
# 1. Select points on the object interactively
python create_query_points.py \
    --sequence_dir /data/my_object \
    --output object_points.json
# (Click on object in the window)

# 2. Track the object
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_object \
    --query_points object_points.json
```

### Example 2: Dense Scene Flow

```bash
# Track dense grid for scene understanding
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_scene \
    --mode full

# Then visualize dense grid
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_scene \
    --mode grid
```

### Example 3: Fast Preview

```bash
# Quick test with low quality
python create_query_points.py \
    --sequence_dir /data/test \
    --grid 16 \
    --output test_grid.json

# Edit config to use fast preset, then:
python track_online_without_gt.py \
    --config configs/custom/track_custom_fast.py \
    --sequence test \
    --query_points test_grid.json
```

---

## Integration with Main DynOMo Pipeline

The custom dataset can also be used with the main DynOMo pipeline:

### 1. Register Dataset

Add to `src/datasets/sequence_dicts.py`:

```python
SEQUENCE_DICT['custom_test'] = {
    'my_sequence': {
        'basedir': '/path/to/data',
        'online_depth': 'DepthAnythingV2',
        # ... other settings
    }
}
```

### 2. Run Main Pipeline

```bash
python scripts/run_dynomo.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence
```

---

## Citation

If you use this code for tracking, please cite the original DynOMo paper and acknowledge this extension.

---

## Support

For issues specific to tracking without ground truth:
1. Check console output for warnings
2. Verify data structure matches `TEST_WITHOUT_GT_README.md`
3. Try fast preset first to validate setup
4. Check visualization outputs for quality assessment

For general DynOMo questions, refer to the main README.md.
