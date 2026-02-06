# 项目结构说明

## 📁 新增和修改的文件结构

```
DynOMo/
│
├── 📄 重构完成说明.md              # 重构完成总结（中文）
├── 📄 CUSTOM_DATASET_README.md     # 完整使用文档（英文）
├── 📄 QUICKSTART_CUSTOM.md         # 快速开始指南（中文）
├── 📄 REFACTOR_SUMMARY.md          # 详细重构总结
├── 📄 PROJECT_STRUCTURE.md         # 本文件
├── 📄 test_custom_setup.py         # 设置验证脚本
│
├── 📂 src/
│   ├── 📂 datasets/
│   │   └── 📂 datasets/
│   │       ├── 📄 __init__.py      # ✏️ 已修改：添加CustomDataset导入
│   │       └── 📄 custom_dataset.py # ⭐ 新增：自定义数据集类
│   │
│   └── 📂 utils/
│       └── 📄 get_data.py          # ✏️ 已修改：支持custom数据集
│
├── 📂 configs/
│   ├── 📂 custom/
│   │   └── 📄 dynomo_custom.py     # ⭐ 新增：自定义数据集配置
│   │
│   └── 📂 data/
│       └── 📄 custom.yaml          # ⭐ 新增：相机参数配置
│
├── 📂 scripts/
│   ├── 📄 train_custom.py          # ⭐ 新增：训练/微调入口（入口1）
│   └── 📄 inference_custom.py      # ⭐ 新增：推理入口（入口2）
│
└── 📂 examples/
    └── 📄 prepare_custom_data.py   # ⭐ 新增：数据验证工具
```

## 📝 文件说明

### 核心代码文件

#### 1. `src/datasets/datasets/custom_dataset.py`
**类型**: 核心数据集类  
**功能**:
- 加载RGB、Depth、SAM分割数据
- 支持在线深度估计和特征提取
- 处理相机位姿（可选）
- 数据预处理和归一化

**关键类和方法**:
```python
class CustomDataset(GradSLAMDataset):
    def get_filepaths()         # 获取所有文件路径
    def load_poses()            # 加载相机位姿
    def _load_bg()              # 加载背景掩码
    def _load_instseg()         # 加载实例分割
    def read_embedding_from_file()  # 读取特征
```

#### 2. `scripts/train_custom.py`
**类型**: 训练入口（入口1）  
**功能**:
- 参数微调/训练
- 从checkpoint恢复
- 配置管理和保存
- GPU调度

**主要函数**:
```python
def train_custom_sequence()  # 训练单个序列
def main()                   # 命令行入口
```

**使用示例**:
```bash
python scripts/train_custom.py \
    --config configs/custom/dynomo_custom.py \
    --sequence my_seq \
    --gpus 0
```

#### 3. `scripts/inference_custom.py`
**类型**: 推理入口（入口2）  
**功能**:
- 模型推理
- 轨迹预测
- 可视化生成
- 性能评估

**主要函数**:
```python
def inference_custom_sequence()  # 推理单个序列
def main()                       # 命令行入口
```

**使用示例**:
```bash
python scripts/inference_custom.py \
    --results_dir experiments/custom/200_200_200/my_seq \
    --gpu 0
```

### 配置文件

#### 4. `configs/custom/dynomo_custom.py`
**类型**: Python配置文件  
**内容**:
- 数据集配置（路径、分辨率等）
- 训练超参数（学习率、迭代次数等）
- 损失权重
- 可视化选项

**关键配置项**:
```python
config = dict(
    data=dict(
        basedir="data/custom",
        sequence="my_sequence",
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

#### 5. `configs/data/custom.yaml`
**类型**: YAML配置文件  
**内容**: 相机内参

**配置项**:
```yaml
dataset_name: 'custom'
camera_params:
  image_height: 480
  image_width: 640
  fx: 525.0
  fy: 525.0
  cx: 320.0
  cy: 240.0
  png_depth_scale: 1000.0
```

### 工具脚本

#### 6. `examples/prepare_custom_data.py`
**类型**: 数据验证工具  
**功能**:
- 检查目录结构
- 验证文件完整性
- 生成数据集报告
- 提供使用建议

**使用示例**:
```bash
python examples/prepare_custom_data.py \
    --basedir data/custom \
    --sequence my_sequence
```

#### 7. `test_custom_setup.py`
**类型**: 安装验证脚本  
**功能**:
- 测试模块导入
- 验证配置文件
- 检查脚本完整性
- 测试数据集类

**使用示例**:
```bash
python test_custom_setup.py
```

### 文档文件

#### 8. `重构完成说明.md`
**语言**: 中文  
**内容**: 
- 重构目标和完成情况
- 快速开始指南
- 常用命令
- 问题排查

#### 9. `CUSTOM_DATASET_README.md`
**语言**: 英文  
**内容**:
- 完整的使用文档
- 数据集要求详解
- 配置参数说明
- 高级用法
- 故障排除

#### 10. `QUICKSTART_CUSTOM.md`
**语言**: 中文  
**内容**:
- 5步快速开始
- 常见问题解答
- 参数调优建议
- 完整示例

#### 11. `REFACTOR_SUMMARY.md`
**语言**: 中文  
**内容**:
- 详细的重构总结
- 设计决策说明
- 文件清单
- 扩展方向

## 🔄 数据流程

### 训练流程
```
用户数据
  ↓
CustomDataset.get_filepaths()      # 获取文件路径
  ↓
CustomDataset.__getitem__()        # 加载和预处理
  ↓
DynOMo.track()                     # 训练
  ↓
保存到 experiments/
```

### 推理流程
```
加载 params.npz + config.json
  ↓
DynOMo.eval()                      # 推理
  ↓
生成可视化和评估
  ↓
保存到 experiments/.../eval/
```

## 📦 依赖关系

```
scripts/train_custom.py
  ├── configs/custom/dynomo_custom.py
  ├── configs/data/custom.yaml
  ├── src/datasets/datasets/custom_dataset.py
  ├── src/model/dynomo.py
  └── src/utils/get_data.py

scripts/inference_custom.py
  ├── experiments/.../config.json
  ├── experiments/.../params.npz
  └── src/model/dynomo.py
```

## 🎯 使用流程

### 完整工作流
```
1. 准备数据
   └── data/custom/my_seq/{rgb,depth,sam}/

2. 验证数据
   └── python examples/prepare_custom_data.py

3. 配置参数
   ├── configs/data/custom.yaml
   └── configs/custom/dynomo_custom.py

4. 训练（入口1）
   └── python scripts/train_custom.py
       └── 输出: experiments/.../params.npz

5. 推理（入口2）
   └── python scripts/inference_custom.py
       └── 输出: experiments/.../eval/
```

## 💾 输出目录结构

```
experiments/custom/200_200_200/my_sequence/
│
├── 📄 config.json              # 训练配置
├── 📄 params.npz              # 最终参数
├── 📄 checkpoint_50.npz       # 训练checkpoint
├── 📄 checkpoint_100.npz
│
├── 📂 logs/                   # 训练日志
│
└── 📂 eval/                   # 推理输出
    ├── 📂 renderings/         # 渲染结果
    ├── 📂 trajectories/       # 轨迹预测
    ├── 📂 visualizations/     # 可视化
    └── 📄 metrics.json        # 评估指标
```

## 🔑 关键配置参数映射

| 配置文件 | 参数 | 作用 | 默认值 |
|---------|------|------|--------|
| custom.yaml | image_height/width | 图像尺寸 | 480x640 |
| custom.yaml | fx/fy/cx/cy | 相机内参 | 525/525/320/240 |
| dynomo_custom.py | basedir | 数据目录 | data/custom |
| dynomo_custom.py | online_depth | 在线深度 | None |
| dynomo_custom.py | online_emb | 在线特征 | dinov2_vits14 |
| dynomo_custom.py | tracking_iters | 迭代次数 | 200 |
| dynomo_custom.py | desired_image_* | 分辨率缩放 | 0.5 |

## 🎨 可选组件

| 组件 | 必需性 | 替代方案 |
|-----|--------|---------|
| RGB | ✅ 必需 | 无 |
| Depth | ❌ 可选 | 使用 --online_depth |
| SAM | ✅ 必需 | 无 |
| Embeddings | ❌ 可选 | 使用 --online_emb |
| Poses | ❌ 可选 | 使用单位矩阵 |

## 📊 性能优化选项

| 优化目标 | 配置参数 | 建议值 |
|---------|---------|--------|
| 加速训练 | tracking_iters | 100 |
| 降低内存 | desired_image_* | 0.25 |
| 跳帧处理 | every_x_frame | 2 |
| GPU使用 | gpus | [0] 或 [0,1] |

---

## 📚 阅读顺序建议

1. 📄 **重构完成说明.md** - 了解重构内容
2. 📄 **QUICKSTART_CUSTOM.md** - 快速上手
3. 📄 **本文档** - 理解项目结构
4. 📄 **CUSTOM_DATASET_README.md** - 深入学习
5. 📄 **REFACTOR_SUMMARY.md** - 了解设计细节

Happy coding! 🚀
