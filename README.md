<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">DynOMo: Online Point Tracking by Dynamic Online Monocular Gaussian Reconstruction</h1>
  <p align="center">
    <a href="https://jennyseidenschwarz.github.io/"><strong>Jenny Seidenschwarz</strong></a>
    ·
    <a href="https://research.nvidia.com/labs/dvl/author/qunjie-zhou/"><strong>Qunjie Zhou</strong></a>
    ·
    <a href="https://www.bart-ai.com/"><strong>Bardenius Duisterhof</strong></a>
    ·
    <a href="https://www.cs.cmu.edu/~deva/"><strong>Deva Ramanan</strong></a>
    ·
    <a href="https://research.nvidia.com/labs/dvl/author/laura-leal-taixe/"><strong>Laura Leal-Taixe</strong></a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2409.02104">Paper</a> | <a href="https://jennyseidenschwarz.github.io/DynOMo.github.io/">Project Page</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./assets/combo_tracks.gif" alt="Logo" width="100%">
  </a>
</p>


<br>

This repository contains the official code of the 3DV 2025 paper "DynOMo: Online Point Tracking by Dynamic Online Monocular Gaussian Reconstruction".

**This version has been refactored to support custom datasets with RGB, Depth, and SAM segmentation data.**

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 0px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#custom-dataset-support">Custom Dataset Support</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#training">Training / Fine-tuning</a></li>
        <li><a href="#inference">Inference</a></li>
      </ul>
    </li>
    <li>
      <a href="#documentation">Documentation</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## Installation

We provide a conda environment file to create our environment. Please run the following to install all necessary dependencies including the rasterizer.

```bash
# create conda environment and install rasterizer
bash scripts/create_env.sh
```

## Custom Dataset Support

This repository has been refactored to support custom datasets with:
- **RGB images** (required)
- **Depth maps** (optional - can use online depth estimation)
- **SAM segmentation masks** (required)

### Dataset Structure

Your data should be organized as follows:

```
data/custom/
└── your_sequence_name/
    ├── rgb/          # RGB images (required)
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── depth/        # Depth maps (optional)
    │   ├── 000000.npy
    │   ├── 000001.npy
    │   └── ...
    └── sam/          # SAM segmentation masks (required)
        ├── 000000.png
        ├── 000001.png
        └── ...
```

### Quick Start

1. **Validate your data:**
```bash
python examples/prepare_custom_data.py \
    --basedir data/custom \
    --sequence your_sequence_name
```

2. **Configure camera parameters** in `configs/data/custom.yaml`

3. **Update config file** `configs/custom/dynomo_custom.py` with your sequence name

## Usage

### Training

Train DynOMo on your custom dataset:

```bash
# Basic training (with precomputed depth)
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

# Training with online depth and embeddings
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence your_sequence_name \
    --gpus 0 \
    --online_depth DepthAnythingV2-vitl \
    --online_emb dinov2_vits14
```

**Training Arguments:**
- `--config`: Path to configuration file
- `--sequence`: Sequence name (folder in basedir)
- `--gpus`: GPU device IDs
- `--online_depth`: Online depth estimation method (`DepthAnything`, `DepthAnythingV2-vitl`)
- `--online_emb`: Online embedding extraction (`dinov2_vits14`, `dinov2_vits14_reg`)
- `--checkpoint`: Path to checkpoint file to resume training

**Output:** Results saved to `experiments/custom/{iters}_{init_iters}_{cam_iters}/{sequence}/`

### Inference

Run inference on trained models:

```bash
# Basic inference
python scripts/inference_custom.py \
    --results_dir experiments/custom/200_200_200/your_sequence \
    --gpu 0

# Inference with visualization
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
```

**Inference Arguments:**
- `--results_dir`: Path to results directory (contains config.json and params.npz)
- `--gpu`: GPU device ID
- `--vis_trajs`: Visualize trajectories
- `--vis_grid`: Visualize evaluation grids
- `--novel_view_mode`: Novel view synthesis mode (`circle`, `zoom_out`)
- `--no_eval_renderings`: Disable rendering evaluation
- `--no_eval_trajs`: Disable trajectory evaluation

**Output:** Evaluation results saved to `experiments/custom/{...}/{sequence}/eval/`

## Documentation

Comprehensive documentation is available:

- **Quick Start Guide** (中文): `QUICKSTART_CUSTOM.md` or `重构完成说明.md`
- **Complete Documentation** (English): `CUSTOM_DATASET_README.md`
- **Project Structure**: `PROJECT_STRUCTURE.md`
- **Refactoring Summary**: `REFACTOR_SUMMARY.md`
- **Files Checklist**: `FILES_CHECKLIST.md`

## Configuration

### Camera Parameters

Edit `configs/data/custom.yaml`:

```yaml
camera_params:
  image_height: 480
  image_width: 640
  fx: 525.0
  fy: 525.0
  cx: 320.0
  cy: 240.0
  png_depth_scale: 1000.0
```

### Training Parameters

Edit `configs/custom/dynomo_custom.py`:

```python
scene_name = "your_sequence_name"

config = dict(
    data=dict(
        basedir="data/custom",
        sequence=scene_name,
        desired_image_height=0.5,
        desired_image_width=0.5,
        online_depth=None,
        online_emb='dinov2_vits14',
    ),
    tracking_obj=dict(
        num_iters=200,
        loss_weights={...},
        lrs={...},
    ),
)
```

## Features

- ✅ Support for RGB + Depth + SAM segmentation data
- ✅ Online depth estimation (DepthAnything, DepthAnythingV2)
- ✅ Online embedding extraction (DINOv2)
- ✅ Separate training and inference entry points
- ✅ Flexible configuration system
- ✅ Checkpoint management
- ✅ Comprehensive documentation

## Troubleshooting

### Common Issues

1. **No RGB directory found**: Check `basedir` and `sequence` in config
2. **File count mismatch**: Ensure RGB, depth, SAM have same number of files
3. **Out of memory**: Reduce image resolution in config
4. **Poor tracking quality**: Adjust loss weights or increase iterations

### Tips

- If you don't have depth maps, use `--online_depth DepthAnythingV2-vitl`
- If you don't have embeddings, use `--online_emb dinov2_vits14`
- For faster training, reduce `tracking_iters` in config
- For lower memory usage, reduce `desired_image_height/width`

## Acknowledgement

This work builds on [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [SplaTAM](https://github.com/spla-tam/SplaTAM).

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{seidenschwarz2025dynomo,
  title={DynOMo: Online Point Tracking by Dynamic Online Monocular Gaussian Reconstruction},
  author={Seidenschwarz, Jenny and Zhou, Qunjie and Duisterhof, Bardienus and Ramanan, Deva and Leal-Taixe, Laura},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Refactored Version**: Custom Dataset Support
**Date**: 2026-02-06
**Status**: Production Ready ✅
