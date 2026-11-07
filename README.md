
# MDRF-Net

MDRF-Net: A Multi-Dimensional Dynamic Difference Progressive Fusion Architecture Based on UAV Imagery for Identifying Melon Drought Resistance

本项目基于 PyTorch 实现 MDRF-Net，利用干旱胁迫（DS）和正常灌溉（WW）条件下的多时相无人机 RGB 影像，实现甜瓜品种抗旱性二分类识别。

## 1. 项目结构

```text
MDRF-Net/
├── main.py       # 主程序，训练与测试入口
├── model.py      # MDRF-Net 网络结构
├── dataload.py   # 数据读取与预处理
├── train.py      # 训练、验证和测试
└── README.md     # 使用说明
```

## 2. 环境配置

实验环境：

- Python 3.9
- PyTorch
- NVIDIA A40 GPU

安装主要依赖：

```bash
pip install torch torchvision numpy pandas pillow tensorboard torchinfo
```

## 3. 数据准备

每个样本包含 11 个时相的 DS 和 WW 配对 RGB 影像，共 22 幅图像。

数据目录格式：

```text
dataset/
├── 5.9experimental/
│   ├── 001.png
│   └── 002.png
├── 5.9control/
│   ├── 001.png
│   └── 002.png
├── 5.13experimental/
├── 5.13control/
├── ...
├── 6.9experimental/
└── 6.9control/
```

其中：

- `experimental`：干旱胁迫组（DS）
- `control`：正常灌溉组（WW）
- 同一品种在所有时相下的图像文件名必须一致。
- 每个品种必须具有全部 11 个时相的 DS/WW 图像。

标签文件采用 CSV 格式：

```csv
id,label
001,0
002,1
```

`id` 对应图像文件名（不含扩展名），`label` 为二分类标签（0/1），具体类别含义以实际标签文件为准。

## 4. 模型训练

### 4.1 修改数据路径

在 `main.py` 中修改以下参数：

```python
base_dir = "/path/to/dataset"

label_file = "/path/to/label.csv"

days = [
    "5.9", "5.13", "5.16", "5.19",
    "5.22", "5.25", "5.28", "5.29",
    "6.3", "6.6", "6.9"
]
```

同时修改 `train.py` 中的模型保存路径：

```python
self.bestmodel_dir = "/path/to/bestmodel"
```

### 4.2 训练参数

| 参数 | 设置 |
|------|------|
| 图像尺寸 | 256 × 256 |
| 时相数量 | 11 |
| 数据集比例 | 训练集:验证集:测试集 = 3:1:1 |
| Batch Size | 4 |
| 优化器 | Adam |
| 学习率 | 0.0005 |
| 损失函数 | CrossEntropyLoss |
| 最大 Epoch | 1000 |
| Early Stopping | 50 |
| Random Seed | 42 |

图像采用 ImageNet 均值和标准差进行归一化。

### 4.3 开始训练

```bash
python main.py
```

程序自动完成：

1. 加载并预处理多时相 RGB 图像。
2. 按类别分层划分训练集、验证集和测试集。
3. 训练 MDRF-Net 并监控验证集损失。
4. 保存验证损失最优的模型权重。
5. 加载候选权重并计算测试集指标。

## 5. 模型输出

模型输入为两组图像张量：

```python
DS.shape = [B, 3, 11, 256, 256]
WW.shape = [B, 3, 11, 256, 256]
```

模型输出：

```python
output.shape = [B, 2]
```

输出为二分类 Logits，可使用 Softmax 获得类别概率。

## 6. 模型保存与评估

模型采用早停策略，根据验证集损失保存最优权重，文件格式为 `.pth`。

训练过程中的损失及分类指标记录在 `runs/` 目录，可通过 TensorBoard 查看：

```bash
tensorboard --logdir=runs
```

测试指标包括 Accuracy、Precision、Recall 和 F1-score。

论文报告的独立测试集结果：

| 指标 | 结果 |
|------|------|
| Accuracy | 93.02% |
| Precision | 84.85% |
| Recall | 96.55% |
| F1-score | 90.32% |

## 7. 注意事项

- 训练前需要修改 `main.py` 和 `train.py` 中的绝对路径。
- 图像文件名必须与标签 ID 对应，缺失标签需提前检查。
- 模型默认使用 11 个时相，调整时相数量需同步修改输入配置。
- 当前 `train.py` 中的单图 `inference()` 函数不适用于完整的多时相输入，正式推理需按 `[B, 3, 11, H, W]` 组织两组数据。
- 论文结果来自指定数据集和实验设置，重新训练结果可能存在差异。

## 8. 论文引用

Gao, X., Hao, X., Li, X., Song, C., Hu, Y., Ma, Y., Liu, B., & Qian, Y. (2026).

MDRF-Net: A multi-dimensional dynamic difference progressive fusion architecture based on UAV imagery for identifying melon drought resistance.

*Smart Agricultural Technology*, 15, 102534.

https://doi.org/10.1016/j.atech.2026.102534
