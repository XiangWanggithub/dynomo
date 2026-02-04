# Custom Dataset Testing & Tracking - Complete Guide

This document provides a complete overview of testing and tracking capabilities for custom datasets without ground truth in DynOMo.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Quick Start](#quick-start)
4. [File Structure](#file-structure)
5. [Workflows](#workflows)
6. [Configuration](#configuration)
7. [Outputs](#outputs)

---

## Overview

Two main capabilities for custom datasets:

### 1. **Dataset Testing** (No Training)
- Verify data loading
- Visualize RGB and depth
- Check dataset structure
- **Fast**: ~1 second per frame

### 2. **Online Point Tracking** (With Training)
- Run DynOMo optimization online
- Track points through sequence
- Generate trajectory visualizations
- **Slower**: ~30-60 seconds per frame

---

## Features

### ✅ What You Can Do

**Dataset Testing:**
- ✅ Test dataset loading without training
- ✅ Visualize RGB frames
- ✅ Visualize depth maps (online or precomputed)
- ✅ Verify camera parameters
- ✅ Check data consistency

**Online Tracking:**
- ✅ Run DynOMo optimization online
- ✅ Track grid of points automatically
- ✅ Track custom user-selected points
- ✅ Visualize point trajectories
- ✅ Compute optical flow
- ✅ Dense grid visualization
- ✅ Extract 2D and 3D trajectories

### ⚠️ What You Cannot Do (No Ground Truth)

- ❌ Compute quantitative metrics (PSNR, SSIM, etc.)
- ❌ Evaluate against ground truth depth
- ❌ Compare with ground truth trajectories
- ❌ Compute TAP-Vid or other benchmark metrics

**Solution**: Use visual inspection and qualitative assessment instead.

---

## Quick Start

### Option A: Just Test Dataset (Fast)

```bash
# 1. Organize data
/path/to/data/
└── my_sequence/
    └── rgb/
        ├── 0000.png
        ├── 0001.png
        └── ...

# 2. Update config
# Edit configs/custom/test_custom.py: set basedir

# 3. Test
python test_without_gt.py \
    --config configs/custom/test_custom.py \
    --sequence my_sequence \
    --visualize_only
```

### Option B: Full Online Tracking (Slow)

```bash
# 1. Update config
# Edit configs/custom/track_custom.py: set basedir

# 2. Run tracking
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence
```

---

## File Structure

### Created Files

```
DynOMo/
├── src/datasets/datasets/
│   └── custom_test.py                  # Custom dataset class
├── configs/custom/
│   ├── test_custom.py                  # Testing config
│   └── track_custom.py                 # Tracking config
├── test_without_gt.py                  # Dataset testing script
├── track_online_without_gt.py          # Online tracking script
├── create_query_points.py              # Query point selection tool
├── example_test_dataset.py             # Minimal test example
├── TEST_WITHOUT_GT_README.md           # Testing documentation
├── TRACKING_WITHOUT_GT_README.md       # Tracking documentation
└── CUSTOM_DATASET_SUMMARY.md          # This file
```

### Data Structure

```
your_data_dir/
└── sequence_name/
    ├── rgb/                    # REQUIRED
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    ├── depth/                  # Optional (can use online)
    │   ├── 0000.npy
    │   └── ...
    ├── intrinsics.txt          # Optional (fx fy cx cy)
    ├── poses.txt               # Optional (4x4 matrices)
    ├── segmentation/           # Optional
    └── embeddings/             # Optional
```

---

## Workflows

### Workflow 1: Dataset Validation

**Purpose**: Verify data loads correctly before training

```bash
# Step 1: Quick validation
python example_test_dataset.py
# (Edit script to set your data path first)

# Step 2: Full visualization
python test_without_gt.py \
    --config configs/custom/test_custom.py \
    --sequence my_sequence \
    --visualize_only

# Step 3: Check outputs
# View: results/test_without_gt/my_sequence/
#   - input_rgb/video.mp4
#   - input_depth/video.mp4
```

**Time**: ~1-2 minutes for 100 frames

---

### Workflow 2: Point Tracking (Grid)

**Purpose**: Track scene motion automatically

```bash
# Run tracking with automatic grid
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence

# Check outputs
# View: results/tracking/my_sequence/my_sequence/
#   - trajectory_visualization/tracked_points.mp4
#   - tracked_trajectories.npz
```

**Time**: ~50-100 minutes for 100 frames

---

### Workflow 3: Point Tracking (Custom Points)

**Purpose**: Track specific objects or regions

```bash
# Step 1: Select points interactively
python create_query_points.py \
    --sequence_dir /path/to/data/my_sequence \
    --output my_queries.json
# (Click on object to track)

# Step 2: Run tracking
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence \
    --query_points my_queries.json

# Step 3: View results
# results/tracking/my_sequence/my_sequence/trajectory_visualization/
```

**Time**: ~50-100 minutes for 100 frames

---

### Workflow 4: Dense Flow Visualization

**Purpose**: Understand scene motion like optical flow

```bash
# Run optimization
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence \
    --mode optimize

# Visualize dense grid
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence \
    --mode grid

# View: results/tracking/my_sequence/my_sequence/grid_points_vis/video.mp4
```

---

### Workflow 5: Fast Preview

**Purpose**: Quick test before full run

```bash
# Use fast config (edit track_custom.py):
config = get_fast_config()

# Run tracking
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_sequence
```

**Time**: ~15-30 minutes for 100 frames (faster but lower quality)

---

## Configuration

### Choosing the Right Config

| Use Case | Resolution | Embeddings | Iterations | Speed | Quality |
|----------|-----------|------------|------------|-------|---------|
| **Preview** | 270×480 | No | 200/30 | Fast | Low |
| **Default** | 360×640 | Yes (64D) | 500/60 | Medium | Medium |
| **High Quality** | 480×640 | Yes (128D) | 1000/100 | Slow | High |

### Key Parameters

**For speed:**
```python
config['data']['desired_height'] = 270
config['data']['desired_width'] = 480
config['data']['load_embeddings'] = False
config['data']['every_x_frame'] = 2  # Skip every other frame
config['tracking_obj']['num_iters'] = 30
```

**For quality:**
```python
config['data']['desired_height'] = 480
config['data']['desired_width'] = 640
config['data']['load_embeddings'] = True
config['data']['embedding_dim'] = 128
config['tracking_obj']['num_iters'] = 100
config['tracking']['best_x'] = 3  # Use best of 3
```

**For online depth:**
```python
config['data']['online_depth'] = 'DepthAnythingV2'  # Recommended
# or
config['data']['online_depth'] = 'DepthAnything'
# or
config['data']['online_depth'] = None  # Use precomputed depth
```

---

## Outputs

### Dataset Testing Output

```
results/test_without_gt/my_sequence/
├── input_rgb/
│   ├── 0000.png ... 0099.png
│   └── video.mp4
└── input_depth/
    ├── 0000.png ... 0099.png
    └── video.mp4
```

### Online Tracking Output

```
results/tracking/my_sequence/
├── config.json
├── query_points.json
├── query_points_frame0.png
└── my_sequence/
    ├── params.npz                      # Optimized Gaussians
    ├── tracked_trajectories.npz        # Trajectory data
    ├── trajectory_visualization/
    │   ├── tracked_points.mp4
    │   ├── tracked_points_trails.mp4
    │   └── frames/*.png
    ├── grid_points_vis/               # Dense grid
    │   └── video.mp4
    ├── flow/                          # Optical flow
    │   └── video.mp4
    └── eval/                          # Additional outputs
        ├── render_rgb/
        └── render_depth/
```

### Trajectory Data Format

```python
import numpy as np

# Load trajectories
data = np.load('tracked_trajectories.npz')

trajectories_2D = data['trajectories_2D']  # Shape: (N_points, N_frames, 2)
trajectories_3D = data['trajectories_3D']  # Shape: (N_points, N_frames, 3)
visibility = data['visibility']            # Shape: (N_points, N_frames)
query_points = data['query_points']        # Shape: (N_points, 2)

# Example: Plot trajectory of first point
import matplotlib.pyplot as plt
traj = trajectories_2D[0]  # (N_frames, 2)
plt.plot(traj[:, 0], traj[:, 1])
plt.title('Trajectory of Point 0')
plt.show()
```

---

## Comparison: Testing vs Tracking

| Feature | Dataset Testing | Online Tracking |
|---------|----------------|-----------------|
| **Speed** | Fast (~1 sec/frame) | Slow (~30-60 sec/frame) |
| **Purpose** | Validate data | Track points |
| **Training** | No | Yes |
| **Outputs** | RGB/depth videos | Trajectories + visualizations |
| **GPU Required** | Optional | Recommended |
| **Use Case** | Data validation | Motion analysis |

---

## Common Use Cases

### 1. New Video from Phone

```bash
# Extract frames from video first
ffmpeg -i myvideo.mp4 /data/myvideo/rgb/%04d.png

# Test dataset
python test_without_gt.py \
    --config configs/custom/test_custom.py \
    --sequence myvideo \
    --visualize_only

# Track if looks good
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence myvideo
```

### 2. Pre-existing Image Sequence

```bash
# Just point to directory
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_images
```

### 3. Depth from Sensor

```bash
# Save depth as NPY files in depth/ folder
# Disable online depth in config:
config['data']['online_depth'] = None

# Run as normal
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence sensor_data
```

### 4. Known Camera Intrinsics

```bash
# Create intrinsics.txt:
echo "fx fy cx cy" > /data/my_seq/intrinsics.txt

# Run (will use these instead of config)
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence my_seq
```

---

## Troubleshooting

### Issue: "No images found"
→ Check RGB folder structure: `sequence_dir/rgb/*.png`

### Issue: "CUDA out of memory"
→ Reduce resolution, disable embeddings, set `every_x_frame=2`

### Issue: "Tracking quality poor"
→ Enable embeddings, increase iterations, check depth quality

### Issue: "Very slow"
→ Use fast preset, reduce resolution, process fewer frames

---

## Next Steps

After testing/tracking:

1. **Analyze trajectories** - Load NPZ files and analyze motion
2. **Adjust parameters** - Tune config for better quality
3. **Process more sequences** - Batch processing
4. **Export results** - Convert to other formats for analysis

---

## Documentation Index

- **TEST_WITHOUT_GT_README.md** - Dataset testing details
- **TRACKING_WITHOUT_GT_README.md** - Online tracking details
- **This file** - Quick reference and overview

---

## Getting Help

1. Try `example_test_dataset.py` first for minimal test
2. Check console output for specific errors
3. Verify data structure matches examples
4. Start with fast preset before full quality
5. Use visualization outputs for debugging

---

## Key Commands Cheat Sheet

```bash
# Validate dataset only
python example_test_dataset.py

# Visualize RGB + depth
python test_without_gt.py --config configs/custom/test_custom.py --sequence SEQ --visualize_only

# Track with grid (automatic)
python track_online_without_gt.py --config configs/custom/track_custom.py --sequence SEQ

# Create custom points
python create_query_points.py --sequence_dir /data/SEQ --output queries.json

# Track custom points
python track_online_without_gt.py --config configs/custom/track_custom.py --sequence SEQ --query_points queries.json

# Only optimize (no tracking)
python track_online_without_gt.py --config configs/custom/track_custom.py --sequence SEQ --mode optimize

# Only track (reuse optimization)
python track_online_without_gt.py --config configs/custom/track_custom.py --sequence SEQ --mode track

# Dense grid visualization
python track_online_without_gt.py --config configs/custom/track_custom.py --sequence SEQ --mode grid

# Optical flow
python track_online_without_gt.py --config configs/custom/track_custom.py --sequence SEQ --mode flow
```

---

**That's it! You now have complete testing and tracking capabilities for custom datasets without ground truth.**
