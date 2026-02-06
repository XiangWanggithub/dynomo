# DynOMo Conda环境安装指南

## 📋 环境要求

- **操作系统**: Linux (Ubuntu 18.04+推荐)
- **CUDA版本**: 11.6
- **Python版本**: 3.10.13
- **GPU**: 支持CUDA的NVIDIA GPU

## 🚀 安装步骤

### 方法一：自动安装（推荐）

在DynOMo根目录下运行：

```bash
# 1. 创建conda环境并安装依赖
conda env create -f environment.yml

# 2. 激活环境
conda activate dynomo

# 3. 安装额外的pip包
pip install imageio-ffmpeg

# 4. 安装Gaussian Rasterizer
cd diff-gaussian-rasterization-w-depth-vis-weights
python setup.py install
pip install .
cd ..

# 完成！
echo "✅ DynOMo环境安装完成！"
```

### 方法二：手动安装（如果自动安装失败）

#### 步骤1: 创建基础conda环境

```bash
# 创建Python 3.10环境
conda create -n dynomo python=3.10.13 -y
conda activate dynomo
```

#### 步骤2: 安装CUDA Toolkit

```bash
# 安装CUDA 11.6
conda install -c nvidia/label/cuda-11.6.0 cuda-toolkit=11.6 -y
conda install cudatoolkit=11.6.2 -c conda-forge -y
```

#### 步骤3: 安装PyTorch及相关库

```bash
# 安装PyTorch 1.12.1 with CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 \
    cudatoolkit=11.6 -c pytorch -c conda-forge -y
```

#### 步骤4: 安装PyTorch3D和相关库

```bash
# 安装PyTorch3D
conda install pytorch3d=0.7.5 -c pytorch3d -y

# 安装PyG相关库
conda install pytorch-scatter pytorch-sparse -c pyg -y
```

#### 步骤5: 安装科学计算库

```bash
# 安装numpy, scipy等
conda install numpy=1.26.4 scipy=1.12.0 -y
conda install mkl mkl-devel mkl-include -y
```

#### 步骤6: 安装其他conda依赖

```bash
# 安装图像处理和可视化库
conda install pillow opencv ffmpeg -c conda-forge -y

# 安装其他工具
conda install pyyaml yacs termcolor tabulate -c conda-forge -y
conda install ipython jupyter -c conda-forge -y
```

#### 步骤7: 安装pip依赖

```bash
# 基础工具
pip install imageio==2.34.0 imageio-ffmpeg==0.4.9
pip install opencv-python==4.9.0.80
pip install matplotlib==3.8.3
pip install pandas==2.2.0
pip install natsort==8.4.0
pip install tqdm==4.65.0

# 深度学习相关
pip install kornia==0.7.1
pip install lpips==0.1.4
pip install pytorch-msssim==1.0.0
pip install torchmetrics==1.3.1

# 3D处理
pip install open3d==0.16.0
pip install plyfile==0.8.1
pip install pyquaternion==0.9.9
pip install roma==1.5.0

# 跟踪和可视化
pip install wandb==0.16.3
pip install plotly==5.19.0
pip install dash==2.15.0
pip install mediapy==1.2.0
pip install flow-vis==0.1

# 工具库
pip install gdown==5.2.0
pip install configargparse==1.7
pip install scikit-learn==1.4.1.post1
pip install h5py
pip install rich==13.7.0
pip install click==8.1.7
```

#### 步骤8: 安装Gaussian Rasterizer

```bash
cd diff-gaussian-rasterization-w-depth-vis-weights
python setup.py install
pip install .
cd ..
```

## ✅ 验证安装

运行以下命令验证环境配置：

```bash
# 激活环境
conda activate dynomo

# 验证Python版本
python --version
# 应输出: Python 3.10.13

# 验证PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
# 应输出:
# PyTorch: 1.12.1+cu116
# CUDA: True

# 验证PyTorch3D
python -c "import pytorch3d; print(f'PyTorch3D: {pytorch3d.__version__}')"

# 验证其他关键库
python -c "import cv2, numpy, scipy, matplotlib, open3d, kornia; print('✅ 所有关键库导入成功')"

# 运行测试脚本
python test_custom_setup.py
```

## 🐛 常见问题

### 问题1: CUDA版本不匹配

**症状**: `RuntimeError: CUDA error: no kernel image is available`

**解决方案**:
```bash
# 检查CUDA版本
nvidia-smi
# 确保显示CUDA 11.x

# 重新安装PyTorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 \
    cudatoolkit=11.6 -c pytorch -c conda-forge --force-reinstall -y
```

### 问题2: Rasterizer安装失败

**症状**: 编译错误或找不到CUDA

**解决方案**:
```bash
# 设置环境变量
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 重新安装
cd diff-gaussian-rasterization-w-depth-vis-weights
rm -rf build dist *.egg-info
python setup.py clean --all
python setup.py install
pip install .
cd ..
```

### 问题3: 内存不足

**症状**: `MemoryError` 或 `Killed`

**解决方案**:
```bash
# 分批安装依赖，避免同时下载太多包
# 或增加swap空间
```

### 问题4: conda solve环境很慢

**症状**: conda一直卡在"Solving environment"

**解决方案**:
```bash
# 使用mamba加速（推荐）
conda install mamba -n base -c conda-forge -y
mamba env create -f environment.yml

# 或者使用libmamba solver
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba
```

## 📦 完整依赖列表

### Conda包（主要）
- Python 3.10.13
- PyTorch 1.12.1 (CUDA 11.6)
- PyTorch3D 0.7.5
- CUDA Toolkit 11.6
- NumPy 1.26.4
- SciPy 1.12.0

### Pip包（主要）
- opencv-python 4.9.0.80
- open3d 0.16.0
- kornia 0.7.1
- wandb 0.16.3
- imageio 2.34.0
- matplotlib 3.8.3

完整列表请参考 `environment.yml`

## 🎯 下一步

环境安装完成后，你可以：

```bash
# 1. 验证数据
python examples/prepare_custom_data.py \
    --basedir data/custom \
    --sequence your_sequence

# 2. 开始训练
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence your_sequence \
    --gpus 0
```

## 💡 提示

- 首次运行可能需要下载预训练模型（DepthAnything, DINOv2）
- 确保有足够的磁盘空间（至少50GB）
- 建议使用SSD以提高数据加载速度
- 如果使用远程服务器，建议使用tmux或screen避免连接断开

---

**环境名称**: `dynomo`
**Python版本**: 3.10.13
**CUDA版本**: 11.6
**PyTorch版本**: 1.12.1
