#!/bin/bash
#
# DynOMo CustomDataset环境一键安装脚本
# 用法: bash install_env_custom.sh
#

set -e  # 遇到错误立即退出

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         DynOMo CustomDataset环境安装脚本                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未检测到conda，请先安装Anaconda或Miniconda"
    echo "   下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ 检测到conda: $(conda --version)"
echo ""

# 检查NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  警告: 未检测到nvidia-smi，请确保已安装NVIDIA驱动"
    read -p "是否继续安装? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ 检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    echo ""
fi

# 步骤1: 创建conda环境
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 1/5: 创建conda环境"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查环境是否已存在
if conda env list | grep -q "^dynomo "; then
    echo "⚠️  环境 'dynomo' 已存在"
    read -p "是否删除并重新创建? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在删除旧环境..."
        conda env remove -n dynomo -y
    else
        echo "使用现有环境..."
    fi
fi

if ! conda env list | grep -q "^dynomo "; then
    echo "正在创建conda环境 (这可能需要几分钟)..."

    # 检查是否有environment.yml
    if [ -f "environment.yml" ]; then
        echo "使用environment.yml创建环境..."
        conda env create -f environment.yml
    else
        echo "environment.yml不存在，手动创建环境..."
        conda create -n dynomo python=3.10.13 -y
    fi
fi

echo "✓ 环境创建完成"
echo ""

# 步骤2: 激活环境并安装PyTorch
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 2/5: 激活环境"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dynomo

echo "✓ 环境已激活: $(which python)"
echo "  Python版本: $(python --version)"
echo ""

# 步骤3: 安装额外的pip包
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 3/5: 安装额外的pip包"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "正在安装imageio-ffmpeg..."
pip install imageio-ffmpeg

echo "✓ 额外包安装完成"
echo ""

# 步骤4: 安装Gaussian Rasterizer
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 4/5: 安装Gaussian Rasterizer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d "diff-gaussian-rasterization-w-depth-vis-weights" ]; then
    echo "正在编译Gaussian Rasterizer (这可能需要几分钟)..."

    # 设置CUDA环境变量
    export CUDA_HOME=$CONDA_PREFIX
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    cd diff-gaussian-rasterization-w-depth-vis-weights

    # 清理之前的编译
    rm -rf build dist *.egg-info

    # 编译安装
    python setup.py install
    pip install .

    cd ..

    echo "✓ Gaussian Rasterizer安装完成"
else
    echo "⚠️  警告: 未找到diff-gaussian-rasterization-w-depth-vis-weights目录"
    echo "   Rasterizer未安装，可能导致运行错误"
fi
echo ""

# 步骤5: 验证安装
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "步骤 5/5: 验证安装"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "检查Python版本..."
python --version

echo ""
echo "检查PyTorch和CUDA..."
python -c "import torch; print(f'✓ PyTorch版本: {torch.__version__}'); print(f'✓ CUDA可用: {torch.cuda.is_available()}'); print(f'✓ CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "检查关键库..."
python -c "
try:
    import cv2
    print('✓ OpenCV已安装')
except ImportError:
    print('✗ OpenCV未安装')

try:
    import numpy
    print('✓ NumPy已安装')
except ImportError:
    print('✗ NumPy未安装')

try:
    import scipy
    print('✓ SciPy已安装')
except ImportError:
    print('✗ SciPy未安装')

try:
    import matplotlib
    print('✓ Matplotlib已安装')
except ImportError:
    print('✗ Matplotlib未安装')

try:
    import pytorch3d
    print('✓ PyTorch3D已安装')
except ImportError:
    print('✗ PyTorch3D未安装')
"

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                  🎉 安装完成！                                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "下一步:"
echo ""
echo "1. 激活环境:"
echo "   conda activate dynomo"
echo ""
echo "2. 验证数据:"
echo "   python examples/prepare_custom_data.py \\"
echo "       --basedir data/custom \\"
echo "       --sequence your_sequence"
echo ""
echo "3. 开始训练:"
echo "   python scripts/train_custom.py \\"
echo "       --config configs/custom/dynomo_custom.py \\"
echo "       --sequence your_sequence \\"
echo "       --gpus 0"
echo ""
echo "详细文档请查看: 安装conda环境指南.md"
echo ""
