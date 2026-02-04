# Testing DynOMo on Datasets Without Ground Truth

This guide explains how to test DynOMo on custom datasets that don't include ground truth depth or annotations. The test code focuses on visualization and qualitative assessment rather than quantitative metrics.

## Overview

The test code includes:
- **Custom Dataset Class** (`src/datasets/datasets/custom_test.py`): Handles datasets without ground truth
- **Test Script** (`test_without_gt.py`): Runs inference and visualization
- **Configuration** (`configs/custom/test_custom.py`): Example configuration file

## Dataset Structure

Your dataset should follow this directory structure:

```
your_data_dir/
└── sequence_name/
    ├── rgb/                    # RGB images (REQUIRED)
    │   ├── 0000.png
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    ├── depth/                  # Depth maps (OPTIONAL - can use online prediction)
    │   ├── 0000.npy            # NumPy arrays with depth values
    │   ├── 0001.npy
    │   └── ...
    ├── segmentation/           # Segmentation masks (OPTIONAL)
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    ├── embeddings/             # Feature embeddings (OPTIONAL)
    │   ├── 0000.npy
    │   ├── 0001.npy
    │   └── ...
    ├── poses.txt               # Camera poses (OPTIONAL - defaults to identity)
    └── intrinsics.txt          # Camera intrinsics (OPTIONAL - uses config defaults)
```

### Required Files

- **rgb/**: Folder containing RGB images (PNG or JPG format)
  - Images should be named sequentially (e.g., `0000.png`, `0001.png`, ...)

### Optional Files

- **depth/**: Precomputed depth maps (NPY format)
  - If not provided, use `online_depth='DepthAnythingV2'` in config for automatic depth prediction
  - Format: NumPy arrays with shape (H, W) containing depth values in meters

- **segmentation/**: Segmentation masks (PNG format)
  - If not provided, dummy masks will be created automatically
  - Format: PNG images where 0 = background, non-zero = foreground

- **embeddings/**: Feature embeddings (NPY format)
  - If not provided and needed, use `online_emb='dino'` in config
  - Format: NumPy arrays with shape (H, W, D) where D is embedding dimension

- **poses.txt**: Camera poses (one per line)
  - Format: Each line contains 16 values (4x4 matrix in row-major order)
  - If not provided, identity poses are used (assumes static camera, dynamic scene)
  - Example line: `1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0`

- **intrinsics.txt**: Camera intrinsics
  - Format: Single line with `fx fy cx cy`
  - If not provided, uses values from config file
  - Example: `525.0 525.0 320.0 240.0`

## Usage

### 1. Prepare Your Data

Organize your data following the directory structure above. At minimum, you need the `rgb/` folder with images.

```bash
# Example directory structure
/path/to/data/
└── my_sequence/
    └── rgb/
        ├── 0000.png
        ├── 0001.png
        └── ...
```

### 2. Update Configuration

Edit `configs/custom/test_custom.py` to match your dataset:

```python
config = {
    'data': {
        'basedir': '/path/to/data',  # UPDATE THIS
        'sequence': 'my_sequence',

        # Update camera parameters if you know them
        'camera_params': {
            'image_height': 480,
            'image_width': 640,
            'fx': 525.0,  # Focal length x
            'fy': 525.0,  # Focal length y
            'cx': 320.0,  # Principal point x
            'cy': 240.0,  # Principal point y
        },

        # Use online depth prediction if no depth/ folder
        'online_depth': 'DepthAnythingV2',

        # Other settings...
    },
}
```

### 3. Run Tests

#### Option A: Visualize Dataset Only (No Model)

This verifies your dataset loads correctly and visualizes the RGB and depth:

```bash
python test_without_gt.py \
    --config configs/custom/test_custom.py \
    --sequence my_sequence \
    --visualize_only
```

Output:
- `results/test_without_gt/my_sequence/input_rgb/`: RGB frames and video
- `results/test_without_gt/my_sequence/input_depth/`: Depth frames and video (colorized)

#### Option B: Run Inference with Trained Model

If you have a trained DynOMo model checkpoint:

```bash
python test_without_gt.py \
    --config configs/custom/test_custom.py \
    --sequence my_sequence \
    --checkpoint /path/to/checkpoint.pth \
    --output_dir results/my_test
```

Output:
- Input visualizations (RGB, depth)
- Rendered outputs (RGB, depth) from the model

**Note**: You may need to modify the `run_inference()` method in `test_without_gt.py` to match your specific model's rendering interface.

### 4. View Results

Results are saved to the output directory (default: `results/test_without_gt/<sequence>/`):

```
results/test_without_gt/my_sequence/
├── input_rgb/
│   ├── 0000.png
│   ├── ...
│   └── video.mp4          # Video compilation
├── input_depth/
│   ├── 0000.png
│   ├── ...
│   └── video.mp4
├── rendered_rgb/          # If running inference
│   ├── 0000.png
│   ├── ...
│   └── video.mp4
└── rendered_depth/        # If running inference
    ├── 0000.png
    ├── ...
    └── video.mp4
```

## Configuration Options

### Depth Prediction

```python
# Option 1: Use online depth prediction (recommended if no depth/ folder)
'online_depth': 'DepthAnythingV2'  # or 'DepthAnything'

# Option 2: Use precomputed depth from depth/ folder
'online_depth': None
```

### Frame Sampling

```python
'every_x_frame': 1,  # Sample every x frames (1=all, 2=every other, etc.)
'start': 0,          # Start frame index
'end': -1,           # End frame (-1 = all frames from start)
```

### Feature Embeddings

```python
# Option 1: Use online embedding computation
'load_embeddings': True,
'online_emb': 'dino'

# Option 2: Use precomputed embeddings from embeddings/ folder
'load_embeddings': True,
'online_emb': None

# Option 3: Disable embeddings (fastest)
'load_embeddings': False
```

## Camera Parameters

If you don't know your camera intrinsics, you can:

1. **Use default values**: The config has reasonable defaults for common cameras
2. **Estimate from image**: For unknown cameras, use:
   ```python
   fx = fy = max(width, height)  # Approximation
   cx = width / 2
   cy = height / 2
   ```
3. **Calibrate**: Use camera calibration tools (OpenCV, COLMAP, etc.)

## Common Issues

### Issue: "No images found in rgb folder"
**Solution**: Ensure your RGB images are named correctly (e.g., `0000.png`, `0001.png`) and are in PNG or JPG format.

### Issue: "Warning: Found X depth files but Y RGB images"
**Solution**: Either:
- Fix the depth folder to have matching number of files, or
- Use online depth prediction: set `online_depth='DepthAnythingV2'`

### Issue: Model rendering fails
**Solution**: The `run_inference()` method in `test_without_gt.py` may need to be adapted to your specific DynOMo model interface. Check your model's rendering API and update accordingly.

### Issue: Out of memory
**Solutions**:
- Reduce resolution: decrease `desired_height` and `desired_width`
- Sample fewer frames: increase `every_x_frame` (e.g., set to 2 or 5)
- Disable embeddings: set `load_embeddings=False`

## Differences from Standard Dataset

The `CustomTestDataset` differs from standard datasets (DAVIS, iPhone, etc.) in these ways:

| Feature | Standard Datasets | CustomTestDataset |
|---------|------------------|-------------------|
| **Ground Truth** | Required | Not required |
| **Depth** | Precomputed | Online or precomputed |
| **Segmentation** | Required | Dummy masks if not provided |
| **Poses** | From COLMAP/files | Identity if not provided |
| **Metrics** | PSNR, SSIM, depth RMSE | Visualization only |

## Example: Complete Workflow

```bash
# 1. Organize your data
mkdir -p /data/my_video/rgb
# Copy your images to /data/my_video/rgb/

# 2. (Optional) Create intrinsics file if you know camera parameters
echo "525.0 525.0 320.0 240.0" > /data/my_video/intrinsics.txt

# 3. Test dataset loading
python test_without_gt.py \
    --config configs/custom/test_custom.py \
    --sequence my_video \
    --visualize_only

# 4. (Optional) Run inference with trained model
python test_without_gt.py \
    --config configs/custom/test_custom.py \
    --sequence my_video \
    --checkpoint checkpoints/trained_model.pth
```

## Advanced: Adding to Main Codebase

To integrate `CustomTestDataset` into the main DynOMo pipeline:

1. **Register the dataset** in `src/datasets/datasets/__init__.py`:
   ```python
   from .custom_test import CustomTestDataset
   ```

2. **Add sequence config** in `src/datasets/sequence_dicts.py`:
   ```python
   SEQUENCE_DICT['custom_test']['my_sequence'] = {
       'basedir': '/path/to/data',
       # ... other config
   }
   ```

3. **Use in main script** `scripts/run_dynomo.py`:
   ```bash
   python scripts/run_dynomo.py \
       --config configs/custom/test_custom.py \
       --sequence my_sequence
   ```

## Questions or Issues?

If you encounter problems:
1. Check the dataset structure matches the required format
2. Verify file naming is sequential (0000.png, 0001.png, ...)
3. Try with `--visualize_only` first to verify data loading
4. Check console output for warnings about missing files

For the main DynOMo project, please refer to the main README.md.
