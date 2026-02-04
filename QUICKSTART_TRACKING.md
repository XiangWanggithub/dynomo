# Quick Start: Tracking Points on Your Video

**Goal**: Track points through a video sequence without ground truth.

---

## ⚡ 5-Minute Setup

### 1. Prepare Your Video

Extract frames from your video:

```bash
# Create directory
mkdir -p /path/to/data/myvideo/rgb

# Extract frames
ffmpeg -i myvideo.mp4 /path/to/data/myvideo/rgb/%04d.png
```

Or if you already have images, just organize them:

```
/path/to/data/
└── myvideo/
    └── rgb/
        ├── 0000.png
        ├── 0001.png
        ├── 0002.png
        └── ...
```

### 2. Configure

Edit `configs/custom/track_custom.py` - **only change this line:**

```python
'basedir': '/path/to/data',  # UPDATE THIS PATH
```

### 3. Run Tracking

```bash
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence myvideo
```

### 4. Wait & View Results

- **Time**: ~1 minute per frame (50-100 mins for 100 frames)
- **Progress**: Watch console for frame-by-frame updates
- **Results**: `results/tracking/myvideo/myvideo/trajectory_visualization/tracked_points.mp4`

---

## 📖 What Just Happened?

1. **DynOMo optimized your scene online** - Built a 4D Gaussian representation
2. **Tracked 1024 points** - Grid of points across the frame
3. **Generated visualizations** - Videos showing point trajectories

---

## 🎯 Track Specific Points

Want to track a specific object instead of a grid?

### 1. Select Points

```bash
python create_query_points.py \
    --sequence_dir /path/to/data/myvideo \
    --output my_points.json
```

A window opens → click on points you want to track → press 'q' to save.

### 2. Track Those Points

```bash
python track_online_without_gt.py \
    --config configs/custom/track_custom.py \
    --sequence myvideo \
    --query_points my_points.json
```

---

## 🚀 Speed It Up

Too slow? Use the fast preset:

Edit `configs/custom/track_custom.py`:

```python
# Add this at the bottom:
config = get_fast_config()
```

This reduces quality but runs ~2-3x faster.

---

## 📊 What You Get

After tracking completes, find outputs in `results/tracking/myvideo/`:

```
myvideo/
├── params.npz                          # Optimized 4D scene
├── tracked_trajectories.npz            # Your trajectory data
└── trajectory_visualization/
    ├── tracked_points.mp4              # Points overlay
    ├── tracked_points_trails.mp4       # With motion trails
    └── frames/                         # Individual frames
```

**Load trajectories in Python:**

```python
import numpy as np

data = np.load('results/tracking/myvideo/myvideo/tracked_trajectories.npz')
trajectories = data['trajectories_2D']  # (N_points, N_frames, 2)
visibility = data['visibility']          # (N_points, N_frames)

# Plot first point
import matplotlib.pyplot as plt
plt.plot(trajectories[0, :, 0], trajectories[0, :, 1])
plt.show()
```

---

## ❓ Common Issues

**"No images found"**
→ Check folder structure: `your_data/myvideo/rgb/*.png`

**"CUDA out of memory"**
→ Add to config: `'desired_height': 270, 'desired_width': 480`

**"Too slow"**
→ Use `config = get_fast_config()` or set `'every_x_frame': 2`

**"Poor tracking"**
→ Make sure `'load_embeddings': True` in config

---

## 🎓 Next Steps

Once you have basic tracking working:

1. **Adjust quality** - See `TRACKING_WITHOUT_GT_README.md` for config options
2. **Analyze motion** - Load NPZ files and analyze trajectories
3. **Dense flow** - Run with `--mode grid` for optical flow-like viz
4. **Custom queries** - Select specific points to track

---

## 📚 Full Documentation

- **CUSTOM_DATASET_SUMMARY.md** - Overview of all features
- **TRACKING_WITHOUT_GT_README.md** - Complete tracking guide
- **TEST_WITHOUT_GT_README.md** - Dataset testing guide

---

## ✅ Checklist

Before running:
- [ ] Video extracted to frames in `rgb/` folder
- [ ] Updated `basedir` in config
- [ ] Enough disk space (~100MB per 100 frames)
- [ ] GPU available (or prepared to wait longer on CPU)

After running:
- [ ] Check `tracked_points.mp4` looks correct
- [ ] Load `tracked_trajectories.npz` successfully
- [ ] Satisfied with quality (or adjust config and re-run)

---

**That's it! You're now tracking points through your video with DynOMo.**

For questions or issues, check the full documentation files or console output for error messages.
