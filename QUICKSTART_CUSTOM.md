# Quick Start Guide - Custom Dataset

快速开始使用自定义数据集进行在线追踪。

## 1️⃣ 准备数据

### 数据目录结构

```
data/custom/
└── my_sequence/
    ├── rgb/          # RGB图像
    ├── depth/        # 深度图（可选）
    └── sam/          # SAM分割掩码
```

### 验证数据结构

```bash
python examples/prepare_custom_data.py \
    --basedir data/custom \
    --sequence my_sequence
```

## 2️⃣ 配置相机参数

编辑 `configs/data/custom.yaml`:

```yaml
camera_params:
  image_height: 480
  image_width: 640
  fx: 525.0
  fy: 525.0
  cx: 320.0
  cy: 240.0
  png_depth_scale: 1000.0  # 如果depth是PNG格式，单位为毫米
```

## 3️⃣ 修改配置文件

编辑 `configs/custom/dynomo_custom.py`:

```python
# 设置序列名称
scene_name = "my_sequence"

config = dict(
    data=dict(
        basedir="data/custom",
        sequence=scene_name,
        # ... 其他配置
    ),
)
```

## 4️⃣ 训练模型

### 基础训练（有预计算深度）

```bash
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence my_sequence \
    --gpus 0
```

### 使用在线深度估计（没有深度图）

```bash
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence my_sequence \
    --gpus 0 \
    --online_depth DepthAnythingV2-vitl
```

### 使用在线特征提取（没有预计算embeddings）

```bash
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence my_sequence \
    --gpus 0 \
    --online_emb dinov2_vits14
```

### 完整命令（在线深度+在线特征）

```bash
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence my_sequence \
    --gpus 0 \
    --online_depth DepthAnythingV2-vitl \
    --online_emb dinov2_vits14
```

## 5️⃣ 推理

### 基础推理

```bash
python scripts/inference_custom.py \
    --results_dir experiments/custom/200_200_200/my_sequence \
    --gpu 0
```

### 带可视化的推理

```bash
python scripts/inference_custom.py \
    --results_dir experiments/custom/200_200_200/my_sequence \
    --gpu 0 \
    --vis_trajs \
    --vis_grid
```

### 新视角合成

```bash
python scripts/inference_custom.py \
    --results_dir experiments/custom/200_200_200/my_sequence \
    --gpu 0 \
    --novel_view_mode circle
```

## 📊 输出结果

训练结果保存在：
```
experiments/custom/200_200_200/my_sequence/
├── config.json      # 使用的配置
├── params.npz       # 训练的参数
└── logs/            # 训练日志
```

推理结果保存在：
```
experiments/custom/200_200_200/my_sequence/eval/
├── renderings/      # 渲染图像
├── trajectories/    # 预测轨迹
└── metrics.json     # 评估指标
```

## 💡 常见问题

### Q: 没有深度图怎么办？
A: 使用 `--online_depth DepthAnythingV2-vitl` 进行在线深度估计

### Q: 没有预计算的embeddings怎么办？
A: 使用 `--online_emb dinov2_vits14` 进行在线特征提取

### Q: 内存不足怎么办？
A: 在配置文件中降低图像分辨率：
```python
desired_image_height=0.25,  # 降低到1/4
desired_image_width=0.25,
```

### Q: SAM分割掩码格式要求？
A:
- 背景应为黑色 (0, 0, 0)
- 不同物体应有不同的颜色/ID
- 支持PNG或JPG格式

### Q: 如何加速训练？
A:
1. 减少迭代次数：`tracking_iters = 100`
2. 降低图像分辨率：`desired_image_height = 0.25`
3. 处理部分帧：`every_x_frame = 2`

## 🔧 参数调优

### 调整损失权重

在 `configs/custom/dynomo_custom.py` 中：

```python
tracking_obj=dict(
    loss_weights=dict(
        im=1.0,          # RGB损失
        depth=0.1,       # 深度损失
        embeddings=16.0, # 特征损失
        rigid=128.0,     # 刚性约束
        # ...
    ),
)
```

### 调整学习率

```python
tracking_obj=dict(
    lrs=dict(
        means3D=0.016,          # 3D位置学习率
        rgb_colors=0.0025,      # 颜色学习率
        unnorm_rotations=0.1,   # 旋转学习率
        # ...
    ),
)
```

## 📝 完整示例

```bash
# 1. 验证数据
python examples/prepare_custom_data.py \
    --basedir data/custom \
    --sequence my_sequence

# 2. 训练
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence my_sequence \
    --gpus 0 \
    --online_depth DepthAnythingV2-vitl \
    --online_emb dinov2_vits14

# 3. 推理
python scripts/inference_custom.py \
    --results_dir experiments/custom/200_200_200/my_sequence \
    --gpu 0 \
    --vis_trajs \
    --vis_grid
```

## 🎯 下一步

- 查看完整文档：`CUSTOM_DATASET_README.md`
- 调整配置参数以获得更好的结果
- 尝试不同的损失权重和学习率
- 在多个序列上进行批处理

祝你使用愉快！🚀
