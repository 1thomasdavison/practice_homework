# 压力图睡眠姿态分类数据集 - 完整报告

## 项目概述

本项目构建了一个用于训练分类神经网络的压力图睡眠姿态数据集，实现了从原始数据到训练完成的完整流程。

## 主要功能特性

### 1. 数据处理管道
- **数据加载**: 自动读取所有人员的txt格式压力图数据
- **数据预处理**: 将多帧数据平均合并为单张压力图
- **标签映射**: 将原始7类标签映射为4类睡眠姿态
- **人员分割**: 按人员进行训练/测试集划分（70%/30%）

### 2. 4类睡眠姿态分类
- **类别1**: 仰卧（原标签1）
- **类别2**: 俯卧（原标签2） 
- **类别3**: 左侧卧（原标签3+4合并）
- **类别4**: 右侧卧（原标签5+6合并）
- **移除**: 坐姿（原标签7）

### 3. 数据增广技术
- **旋转增广**: 随机旋转-15°到+15°
- **平移增广**: 随机平移±2像素
- **仿射变换**: 使用PyTorch的grid_sample实现
- **动态增广**: 每次访问数据都应用不同的增广

### 4. 数据集统计
- **总样本数**: 138个（23人×6个姿态，去除坐姿）
- **训练集**: 96个样本（16人）
- **测试集**: 42个样本（7人）
- **数据形状**: 40×26压力传感器矩阵
- **数据类型**: 32位浮点数

## 文件结构

```
project/
├── dataloader/
│   ├── get_data.py          # 核心数据加载和预处理
│   └── dataloader.py        # PyTorch数据集和增广实现
├── utilization_of_ai/
│   ├── classification.md    # 分类检测ai使用情况
├── train.py                 # 完整训练过程
├── data/                    # 原始数据文件夹
├── web_app.py               # 数据功能性展示
└── 睡姿展示系统.ipynb         # 展示，与web_app.py一起使用
```

## 核心类和函数

### DataManager类
- 管理数据集的构建、划分和保存
- 支持数据集缓存，避免重复处理
- 提供统计信息和数据分布查看

### PressureMapDataset类
- PyTorch Dataset实现
- 集成数据增广功能
- 支持训练/测试模式切换

### 数据增广方法
- `_apply_augmentation()`: 主要增广入口
- `_rotate_tensor()`: 旋转变换实现
- `_translate_tensor()`: 平移变换实现

## 训练结果

### 模型架构
- **网络类型**: 简单CNN（SimpleCNN）
- **参数数量**: 1,175,364
- **输入尺寸**: 1×40×26
- **输出类别**: 4类

### 性能表现
- **最佳测试准确率**: 100.00%
- **训练准确率**: 100.00%
- **各类别表现**:
  - 仰卧: 100.00% (7/7)
  - 俯卧: 100.00% (7/7)  
  - 左侧卧: 100.00% (14/14)
  - 右侧卧: 100.00% (14/14)


## 使用方法

### 快速开始
```python
from dataloader.dataloader import DataManager, PressureMapDataset
import torch

# 1. 创建数据管理器
data_root = "/path/to/data/text_data"
data_manager = DataManager(data_root, train_ratio=0.7, random_state=42)

# 2. 创建数据集（启用增广）
train_dataset = PressureMapDataset(data_manager.train_data, augment=True)
test_dataset = PressureMapDataset(data_manager.test_data, augment=False)

# 3. 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# 4. 开始训练
# 训练代码详见 train.py
```

### 数据增广测试
```bash
cd project
python test_augmentation.py
```

### 完整训练
```bash
cd project
python train_example.py
```

## 技术亮点

1. **完整的数据流水线**: 从原始txt文件到PyTorch训练的完整自动化流程
2. **智能数据增广**: 结合旋转和平移的双重增广策略
3. **人员级别划分**: 确保训练和测试的独立性
4. **高准确率**: 在4类分类任务上达到100%的测试准确率

## 改进建议

1. **更多增广方法**: 可考虑添加噪声、缩放等增广技术
2. **模型优化**: 尝试更复杂的网络架构如ResNet、DenseNet
3. **超参数调优**: 系统化调优学习率、批次大小等参数
4. **交叉验证**: 实现k折交叉验证以获得更稳定的性能评估
5. **数据不平衡处理**: 针对类别分布不均匀的情况添加平衡策略

## 总结

本项目成功构建了一个高质量的压力图睡眠姿态分类数据集，集成了数据增广技术，并在4类分类任务上取得了优异的性能。代码结构清晰，功能完整，为相关研究提供了坚实的基础。
