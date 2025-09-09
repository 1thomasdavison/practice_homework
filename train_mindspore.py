"""
使用MindSpore实现的压力图睡姿分类训练脚本
适配华为Ascend AI处理器
"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.train import Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.nn import Adam, CrossEntropyLoss
from mindspore.nn.metrics import Accuracy
import numpy as np
import sys
import os
import pickle
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(__file__))
from dataloader.dataloader_mindspore import DataManager

def save_dataset(data_manager, save_path):
    """
    保存处理好的数据集
    
    Args:
        data_manager: DataManager对象
        save_path: 保存路径
    """
    dataset_info = {
        'train_data': data_manager.train_data,
        'test_data': data_manager.test_data,
        'person_list': data_manager.person_list,
        'all_data': data_manager.all_data
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(dataset_info, f)
    
    print(f"数据集已保存到: {save_path}")

def load_dataset(load_path):
    """
    加载已保存的数据集
    
    Args:
        load_path: 数据集文件路径
        
    Returns:
        dataset_info: 数据集信息字典
    """
    with open(load_path, 'rb') as f:
        dataset_info = pickle.load(f)
    
    print(f"数据集已从 {load_path} 加载")
    return dataset_info

# MindSpore版本的SimpleCNN模型
class SimpleCNN(nn.Cell):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv_layers = nn.SequentialCell([
            # 第一层卷积 (1, 40, 26) -> (32, 38, 24)
            nn.Conv2d(1, 32, kernel_size=3, padding=0, has_bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 第二层卷积 (32, 38, 24) -> (64, 36, 22)
            nn.Conv2d(32, 64, kernel_size=3, padding=0, has_bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 最大池化 (64, 36, 22) -> (64, 18, 11)
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层卷积 (64, 18, 11) -> (128, 16, 9)
            nn.Conv2d(64, 128, kernel_size=3, padding=0, has_bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # 最大池化 (128, 16, 9) -> (128, 8, 4)
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        
        # 全连接层
        self.fc_layers = nn.SequentialCell([
            nn.Dropout(0.5),
            nn.Dense(128 * 8 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Dense(256, 128),
            nn.ReLU(),
            nn.Dense(128, num_classes)
        ])
        
        # 用于展平的操作
        self.flatten = nn.Flatten()
        
    def construct(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

class TrainOneStepCell(nn.Cell):
    """训练单步封装"""
    def __init__(self, network, optimizer, loss_fn):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.grad = ops.GradOperation(get_by_list=True)
        self.weights = self.optimizer.parameters
        
    def construct(self, data, label):
        def forward_fn(data, label):
            logits = self.network(data)
            loss = self.loss_fn(logits, label)
            return loss, logits
        
        grad_fn = ops.value_and_grad(forward_fn, None, self.weights, has_aux=True)
        (loss, logits), grads = grad_fn(data, label)
        self.optimizer(grads)
        return loss, logits

def create_dataset(data_list, batch_size=16, training=True):
    """创建MindSpore数据集"""
    def generator():
        for pressure_map, label, person_name in data_list:
            # 确保数据类型正确
            pressure_map = np.array(pressure_map, dtype=np.float32)
            label = np.array(label - 1, dtype=np.int32)  # 转换为0-3的标签
            
            # 添加通道维度 (40, 26) -> (1, 40, 26)
            if len(pressure_map.shape) == 2:
                pressure_map = np.expand_dims(pressure_map, axis=0)
            
            yield pressure_map, label
    
    dataset = ds.GeneratorDataset(generator, ["data", "label"])
    
    if training:
        dataset = dataset.shuffle(buffer_size=len(data_list))
    
    dataset = dataset.batch(batch_size)
    return dataset

def train_one_epoch(model, train_dataset, loss_fn, optimizer):
    """训练一个epoch"""
    model.set_train(True)
    total_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    
    train_one_step = TrainOneStepCell(model, optimizer, loss_fn)
    
    for batch_idx, (data, labels) in enumerate(train_dataset.create_tuple_iterator()):
        loss, outputs = train_one_step(data, labels)
        
        # 计算准确率
        predicted = ops.argmax(outputs, axis=1)
        batch_correct = ops.equal(predicted, labels).sum()
        
        total_loss += loss.asnumpy()
        correct += batch_correct.asnumpy()
        total += labels.shape[0]
        batch_count += 1
        
        if batch_idx % 5 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.asnumpy():.4f}, Acc: {100.*correct/total:.2f}%')
    
    return total_loss / batch_count, 100. * correct / total

def test(model, test_dataset, loss_fn):
    """测试模型"""
    model.set_train(False)
    total_loss = 0
    correct = 0
    total = 0
    batch_count = 0
    
    for data, labels in test_dataset.create_tuple_iterator():
        outputs = model(data)
        loss = loss_fn(outputs, labels)
        
        predicted = ops.argmax(outputs, axis=1)
        batch_correct = ops.equal(predicted, labels).sum()
        
        total_loss += loss.asnumpy()
        correct += batch_correct.asnumpy()
        total += labels.shape[0]
        batch_count += 1
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / batch_count
    
    return avg_loss, accuracy

def count_parameters(model):
    """计算模型参数数量"""
    total_params = 0
    for param in model.trainable_params():
        total_params += param.size
    return total_params

def main():
    """主函数 - 完整的训练示例"""
    
    # 设置MindSpore上下文
    mindspore.set_context(
        mode=mindspore.GRAPH_MODE,
        device_target="Ascend",  # 使用Ascend设备
        device_id=0
    )
    
    print("使用设备: Ascend")
    
    # 数据路径
    data_root = "/workspaces/codespaces-jupyter/project/data/text_data"
    save_path = "/workspaces/codespaces-jupyter/project/dataset_mindspore.pkl"
    
    # 创建或加载数据集
    if os.path.exists(save_path):
        print("加载已保存的数据集...")
        dataset_info = load_dataset(save_path)
        
        # 重新创建数据管理器（用于获取数据信息）
        data_manager = DataManager(data_root, train_ratio=0.7, random_state=42)
    else:
        print("创建新的数据集...")
        data_manager = DataManager(data_root, train_ratio=0.7, random_state=42)
        save_dataset(data_manager, save_path)
    
    # 打印数据集信息
    data_manager.print_info()
    
    # 创建MindSpore数据集
    batch_size = 16
    train_dataset = create_dataset(data_manager.train_data, batch_size=batch_size, training=True)
    test_dataset = create_dataset(data_manager.test_data, batch_size=batch_size, training=False)
    
    # 创建模型
    model = SimpleCNN(num_classes=4)
    
    print(f"\n模型参数数量: {count_parameters(model):,}")
    
    # 定义损失函数和优化器
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.trainable_params(), learning_rate=0.001)
    
    # 训练设置
    num_epochs = 15
    best_acc = 0
    
    print(f"\n开始训练 {num_epochs} 个epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_dataset, loss_fn, optimizer)
        
        # 测试
        test_loss, test_acc = test(model, test_dataset, loss_fn)
        
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            mindspore.save_checkpoint(model, '/workspaces/codespaces-jupyter/project/best_model_mindspore.ckpt')
            print(f"保存最佳模型，准确率: {best_acc:.2f}%")
    
    print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")
    
    # 详细的测试结果分析
    print("\n" + "=" * 60)
    print("详细测试结果分析")
    print("=" * 60)
    
    model.set_train(False)
    class_names = ['仰卧', '俯卧', '左侧卧', '右侧卧']
    
    # 计算每个类别的准确率
    class_correct = [0] * 4
    class_total = [0] * 4
    
    for data, labels in test_dataset.create_tuple_iterator():
        outputs = model(data)
        predicted = ops.argmax(outputs, axis=1)
        
        for i in range(labels.shape[0]):
            label = labels[i].asnumpy()
            pred = predicted[i].asnumpy()
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1
    
    print("各类别准确率:")
    for i in range(4):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f'{class_names[i]}: {class_correct[i]}/{class_total[i]} = {acc:.2f}%')
        else:
            print(f'{class_names[i]}: 0/0 = N/A')

if __name__ == "__main__":
    main()
