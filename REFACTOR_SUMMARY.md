# 代码重构总结

## 📋 重构概述

本次重构为DynOMo项目添加了对自定义数据集的完整支持，包括RGB图像、深度图和SAM分割掩码。实现了两个独立的入口：**参数微调（训练）** 和 **推理**。

## 🆕 新增文件

### 1. 核心数据集类
- **`src/datasets/datasets/custom_dataset.py`**
  - 自定义数据集加载类 `CustomDataset`
  - 支持RGB、Depth、SAM分割
  - 支持在线深度估计和在线特征提取
  - 支持可选的相机位姿加载

### 2. 配置文件
- **`configs/custom/dynomo_custom.py`**
  - 自定义数据集的主配置文件
  - 包含训练、追踪、可视化等所有参数

- **`configs/data/custom.yaml`**
  - 相机内参配置文件
  - 包含图像尺寸、焦距、主点等参数

### 3. 训练/微调入口
- **`scripts/train_custom.py`**
  - 参数微调脚本
  - 支持从头训练或从checkpoint恢复
  - 支持在线深度估计和特征提取
  - 自动保存配置和checkpoint

### 4. 推理入口
- **`scripts/inference_custom.py`**
  - 推理脚本
  - 支持轨迹可视化
  - 支持新视角合成
  - 支持渲染质量评估

### 5. 工具脚本
- **`examples/prepare_custom_data.py`**
  - 数据集结构验证工具
  - 检查文件完整性
  - 生成使用建议

### 6. 文档
- **`CUSTOM_DATASET_README.md`**
  - 完整使用文档
  - 包含详细的配置说明和故障排除

- **`QUICKSTART_CUSTOM.md`**
  - 快速开始指南
  - 包含常见问题解答
  - 提供完整示例命令

- **`REFACTOR_SUMMARY.md`**
  - 本文档，重构总结

## 🔧 修改的文件

### 1. `src/datasets/datasets/__init__.py`
```python
# 添加了CustomDataset的导入
from .custom_dataset import CustomDataset
```

### 2. `src/utils/get_data.py`
- 添加了 `CustomDataset` 导入
- 更新了 `get_dataset()` 函数，添加对 'custom' 数据集的支持
- 添加了 `use_gt_poses` 和 `pose_file` 参数传递

## 🎯 主要功能

### 1. 数据集加载
- ✅ RGB图像加载（支持 .jpg, .png, .jpeg）
- ✅ 深度图加载（支持 .npy, .png, .exr）
- ✅ SAM分割掩码加载
- ✅ 可选的预计算特征加载
- ✅ 可选的相机位姿加载

### 2. 在线处理
- ✅ 在线深度估计（DepthAnything, DepthAnythingV2）
- ✅ 在线特征提取（DINOv2）
- ✅ 自动PCA降维

### 3. 训练功能
- ✅ 从头训练
- ✅ 从checkpoint恢复
- ✅ 自动保存配置
- ✅ 自动保存checkpoint
- ✅ GPU选择

### 4. 推理功能
- ✅ 轨迹预测和可视化
- ✅ 渲染质量评估
- ✅ 新视角合成（circle, zoom_out）
- ✅ 网格可视化

## 📁 数据集结构要求

```
data/custom/
└── your_sequence/
    ├── rgb/              # 必需：RGB图像
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── depth/            # 可选：深度图（或使用在线估计）
    │   ├── 000000.npy
    │   ├── 000001.npy
    │   └── ...
    ├── sam/              # 必需：SAM分割掩码
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    ├── embeddings/       # 可选：预计算特征（或使用在线提取）
    │   ├── 000000.npy
    │   ├── 000001.npy
    │   └── ...
    └── poses.npy         # 可选：相机位姿
```

## 🚀 使用示例

### 训练
```bash
# 基础训练
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence my_sequence \
    --gpus 0

# 使用在线处理
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence my_sequence \
    --gpus 0 \
    --online_depth DepthAnythingV2-vitl \
    --online_emb dinov2_vits14
```

### 推理
```bash
# 基础推理
python scripts/inference_custom.py \
    --results_dir experiments/custom/200_200_200/my_sequence \
    --gpu 0

# 带可视化
python scripts/inference_custom.py \
    --results_dir experiments/custom/200_200_200/my_sequence \
    --gpu 0 \
    --vis_trajs \
    --vis_grid
```

## 🔍 关键设计决策

### 1. 继承基类设计
`CustomDataset` 继承自 `GradSLAMDataset`，复用了大部分数据加载和预处理逻辑，只需实现特定方法：
- `get_filepaths()`: 获取文件路径
- `load_poses()`: 加载相机位姿
- `_load_bg()`: 加载背景掩码
- `_load_instseg()`: 加载实例分割

### 2. 配置灵活性
- 支持在线和离线两种模式
- 参数可通过配置文件或命令行传递
- 自动保存配置便于复现

### 3. 错误处理
- 数据集验证脚本检查文件完整性
- 清晰的错误信息和警告
- 自动处理可选组件

### 4. 兼容性
- 与现有DAVIS、iPhone、Panoptic Sports数据集保持一致
- 最小化对原有代码的修改
- 向后兼容

## 📊 输出结构

### 训练输出
```
experiments/custom/{iters}_{init_iters}_{cam_iters}/{sequence}/
├── config.json           # 配置文件
├── params.npz           # 最终参数
├── checkpoint_*.npz     # 训练checkpoint
└── logs/                # 日志文件
```

### 推理输出
```
experiments/custom/{iters}_{init_iters}_{cam_iters}/{sequence}/eval/
├── renderings/          # 渲染结果
├── trajectories/        # 轨迹预测
├── visualizations/      # 可视化结果
└── metrics.json         # 评估指标
```

## ⚙️ 配置参数说明

### 核心参数
- `basedir`: 数据集根目录
- `sequence`: 序列名称
- `desired_image_height/width`: 目标图像尺寸（比例或像素）
- `online_depth`: 在线深度估计方法
- `online_emb`: 在线特征提取方法
- `use_gt_poses`: 是否使用真值位姿
- `pose_file`: 位姿文件路径

### 训练参数
- `tracking_iters`: 对象追踪迭代次数
- `tracking_iters_init`: 初始化迭代次数
- `tracking_iters_cam`: 相机追踪迭代次数
- `loss_weights`: 各损失项权重
- `lrs`: 各参数学习率

## 🎓 最佳实践

### 1. 数据准备
- 确保文件命名使用零填充（如 `000000.png`）
- SAM掩码背景应为黑色
- 深度图单位统一（米或毫米）

### 2. 性能优化
- 降低图像分辨率可加速训练
- 使用 `every_x_frame` 跳帧处理
- 适当减少迭代次数

### 3. 质量优化
- 调整损失权重平衡各项约束
- 根据场景特点选择初始化方法
- 使用checkpoint进行参数搜索

## 🐛 故障排除

### 常见问题
1. **找不到RGB目录**: 检查 `basedir` 和 `sequence` 配置
2. **文件数量不匹配**: 确保RGB、depth、SAM文件数量一致
3. **内存不足**: 降低图像分辨率或减少batch size
4. **追踪质量差**: 调整损失权重或增加迭代次数

### 调试技巧
1. 使用 `prepare_custom_data.py` 验证数据集
2. 查看保存的配置文件确认参数
3. 检查训练日志定位问题
4. 从小规模实验开始

## 📚 扩展方向

### 潜在改进
1. 支持更多深度估计方法
2. 支持更多特征提取方法
3. 添加数据增强
4. 支持多GPU并行训练
5. 添加实时可视化

### 集成建议
1. 与其他追踪方法比较
2. 支持更多数据集格式
3. 添加自动参数调优
4. 集成到更大的系统中

## ✅ 测试清单

- [ ] 数据集验证脚本运行正常
- [ ] 训练脚本可以从头开始训练
- [ ] 训练脚本可以从checkpoint恢复
- [ ] 推理脚本正常生成结果
- [ ] 在线深度估计工作正常
- [ ] 在线特征提取工作正常
- [ ] 可视化功能正常
- [ ] 配置文件正确保存和加载

## 📞 支持与反馈

如遇到问题：
1. 查看 `CUSTOM_DATASET_README.md` 完整文档
2. 查看 `QUICKSTART_CUSTOM.md` 快速指南
3. 使用 `prepare_custom_data.py` 验证数据
4. 检查配置文件和日志

## 🎉 总结

本次重构成功实现了：
- ✅ 完整的自定义数据集支持
- ✅ 独立的训练和推理入口
- ✅ 灵活的配置系统
- ✅ 详细的文档和示例
- ✅ 良好的错误处理
- ✅ 向后兼容性

现在您可以轻松地在自己的数据集上使用DynOMo进行在线追踪！
