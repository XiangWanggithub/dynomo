# Custom Dataset Online Tracking with DynOMo

This guide explains how to use DynOMo for online tracking on custom datasets with RGB images, depth maps, and SAM segmentation masks.

## 📁 Dataset Structure

Your dataset should follow this directory structure:

```
data/custom/
├── your_sequence_name/
│   ├── rgb/              # RGB images
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── depth/            # Depth maps (optional if using online depth)
│   │   ├── 000000.npy    # or .png, .exr
│   │   ├── 000001.npy
│   │   └── ...
│   ├── sam/              # SAM segmentation masks
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   ├── embeddings/       # Optional: precomputed embeddings
│   │   ├── 000000.npy
│   │   ├── 000001.npy
│   │   └── ...
│   └── poses.npy         # Optional: camera poses (Nx4x4 array)
```

### File Format Requirements

- **RGB images**: `.png`, `.jpg`, or `.jpeg`
- **Depth maps**: `.npy` (numpy array), `.png`, or `.exr`
- **SAM masks**: `.png` or `.jpg`
  - Background should be 0 (black)
  - Different objects should have different colors/IDs
- **Embeddings** (optional): `.npy` files with shape (H, W, C)
- **Poses** (optional): `.npy` file with shape (N, 4, 4) or `.txt` file

## ⚙️ Configuration

### 1. Update Camera Parameters

Edit `configs/data/custom.yaml` with your camera intrinsics:

```yaml
dataset_name: 'custom'
camera_params:
  image_height: 480        # Your image height
  image_width: 640         # Your image width
  fx: 525.0               # Focal length x
  fy: 525.0               # Focal length y
  cx: 320.0               # Principal point x
  cy: 240.0               # Principal point y
  png_depth_scale: 1000.0  # Depth scale (1000.0 for mm→m, 1.0 if already in meters)
  crop_edge: 0
```

### 2. Update Training Configuration

Edit `configs/custom/dynomo_custom.py`:

```python
# Set your sequence name
scene_name = "your_sequence_name"

# Update data directory
config = dict(
    data=dict(
        basedir="data/custom",  # Path to your data directory
        sequence=scene_name,

        # Image scaling (0.5 = half size, or use absolute pixels)
        desired_image_height=0.5,
        desired_image_width=0.5,

        # Embedding settings
        load_embeddings=True,      # Set False if no precomputed embeddings
        online_emb='dinov2_vits14',  # Use online extraction (or None for precomputed)

        # Depth settings
        online_depth=None,  # Set to 'DepthAnythingV2-vitl' for online depth

        # Camera pose settings
        use_gt_poses=False,  # Set True if you have ground truth poses
        pose_file=None,      # e.g., "poses.npy"
    ),
    # ... other settings
)
```

## 🚀 Usage

### Training / Fine-tuning

```bash
# Basic training
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence your_sequence_name \
    --gpus 0

# Training with online depth estimation
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence your_sequence_name \
    --gpus 0 \
    --online_depth DepthAnythingV2-vitl

# Training with online embedding extraction
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence your_sequence_name \
    --gpus 0 \
    --online_emb dinov2_vits14

# Resume from checkpoint
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence your_sequence_name \
    --gpus 0 \
    --checkpoint experiments/custom/200_200_200/your_sequence/checkpoint_100.npz
```

### Inference

```bash
# Inference using results directory
python scripts/inference_custom.py \
    --results_dir experiments/custom/200_200_200/your_sequence \
    --gpu 0

# Inference with trajectory visualization
python scripts/inference_custom.py \
    --results_dir experiments/custom/200_200_200/your_sequence \
    --gpu 0 \
    --vis_trajs \
    --vis_grid

# Inference with novel view synthesis
python scripts/inference_custom.py \
    --results_dir experiments/custom/200_200_200/your_sequence \
    --gpu 0 \
    --novel_view_mode circle

# Inference using checkpoint file
python scripts/inference_custom.py \
    --checkpoint experiments/custom/200_200_200/your_sequence/params.npz \
    --config configs/custom/dynomo_custom.py \
    --sequence your_sequence \
    --gpu 0
```

## 📊 Output

Training outputs will be saved to:
```
experiments/custom/{iters}_{init_iters}_{cam_iters}/{sequence_name}/
├── config.json           # Configuration used
├── params.npz           # Final trained parameters
├── checkpoint_*.npz     # Checkpoints during training
└── logs/                # Training logs
```

Inference outputs will be saved to:
```
experiments/custom/{iters}_{init_iters}_{cam_iters}/{sequence_name}/eval/
├── renderings/          # Rendered images
├── trajectories/        # Predicted trajectories
├── visualizations/      # Visualization results
└── metrics.json         # Evaluation metrics
```

## 🎯 Tips

### Depth Maps

- **Precomputed depth**: Save as `.npy` files in meters
- **Online depth**: Use `--online_depth DepthAnythingV2-vitl` (no depth folder needed)
- **Depth scale**: Adjust `png_depth_scale` in config if using PNG depth (e.g., 1000.0 for millimeters)

### SAM Segmentation

- Background should be black (RGB: 0, 0, 0)
- Each object instance should have a unique color
- If you have binary masks (foreground/background only), that's fine too

### Embeddings

- **Precomputed**: Save as `.npy` files with shape (H, W, embedding_dim)
- **Online**: Use `--online_emb dinov2_vits14` (no embeddings folder needed)
- DynOMo will automatically apply PCA if embedding dimension doesn't match

### Camera Poses

- **No poses**: Use identity poses (default) - suitable for static camera
- **Ground truth poses**: Set `use_gt_poses=True` and provide `pose_file`
- **Optimize poses**: Keep `gt_w2c=False` to optimize camera poses during tracking

### Performance Tuning

Adjust these parameters in the config for better results:

```python
# Reduce for faster training
tracking_iters = 100
tracking_iters_init = 100
tracking_iters_cam = 100

# Image resolution (smaller = faster)
desired_image_height = 0.25  # Quarter resolution
desired_image_width = 0.25

# Frame sampling (process every N frames)
every_x_frame = 2  # Process every 2nd frame
```

## 🐛 Troubleshooting

### "RGB directory not found"
- Check that `basedir` in config points to the correct directory
- Ensure RGB images are in `{basedir}/{sequence}/rgb/`

### "Number of SAM masks doesn't match RGB images"
- Ensure you have one SAM mask for each RGB image
- Check that filenames are sorted correctly (use zero-padded numbers)

### Out of memory
- Reduce `desired_image_height` and `desired_image_width`
- Reduce `tracking_iters`, `tracking_iters_init`, and `tracking_iters_cam`
- Process fewer frames using `every_x_frame`

### Poor tracking quality
- Increase number of iterations
- Adjust loss weights in config
- Try different `mov_init_by` methods ('kNN', 'seg', 'per_point')
- Ensure depth maps are accurate
- Check SAM segmentation quality

## 📚 Advanced Usage

### Custom Loss Weights

Edit `configs/custom/dynomo_custom.py`:

```python
tracking_obj=dict(
    loss_weights=dict(
        im=1.0,          # RGB loss
        depth=0.1,       # Depth loss
        rot=16.0,        # Rotation regularization
        rigid=128.0,     # Rigidity loss
        iso=16,          # Isometry loss
        embeddings=16.0, # Embedding loss
        # ... adjust as needed
    ),
)
```

### Custom Learning Rates

```python
tracking_obj=dict(
    lrs=dict(
        means3D=0.016,          # 3D position learning rate
        rgb_colors=0.0025,      # RGB color learning rate
        unnorm_rotations=0.1,   # Rotation learning rate
        # ... adjust as needed
    ),
)
```

### Batch Processing Multiple Sequences

```bash
# Create a script to process multiple sequences
for seq in sequence1 sequence2 sequence3; do
    python scripts/train_custom.py \
        --config configs/custom/dynomo_custom.py \
        --sequence $seq \
        --gpus 0
done
```

## 📞 Support

For issues or questions, please refer to the main DynOMo repository or create an issue.
