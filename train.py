"""
使用示例：如何使用压力图数据集进行训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
from dataloader.dataloader import DataManager
import pickle

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

# 简单的CNN模型示例
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):  # 修改为4类
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层卷积 (1, 40, 26) -> (32, 38, 24)
            nn.Conv2d(1, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # 第二层卷积 (32, 38, 24) -> (64, 36, 22)
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # 最大池化 (64, 36, 22) -> (64, 18, 11)
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第三层卷积 (64, 18, 11) -> (128, 16, 9)
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # 最大池化 (128, 16, 9) -> (128, 8, 4)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 计算卷积层输出的特征维度
        # 最终输出: (128, 8, 4) = 4096
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, labels, _) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 5 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    return total_loss / len(train_loader), 100. * correct / total

def test(model, test_loader, criterion, device):
    """测试模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels, _ in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return avg_loss, accuracy

def main():
    """主函数 - 完整的训练示例"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    data_root = "/workspaces/codespaces-jupyter/project/data/text_data"
    save_path = "/workspaces/codespaces-jupyter/project/dataset.pkl"
    
    # 创建或加载数据集
    if os.path.exists(save_path):
        print("加载已保存的数据集...")
        dataset_info = load_dataset(save_path)
        
        # 重新创建数据管理器（用于获取dataloader）
        data_manager = DataManager(data_root, train_ratio=0.7, random_state=42)
    else:
        print("创建新的数据集...")
        data_manager = DataManager(data_root, train_ratio=0.7, random_state=42)
        save_dataset(data_manager, save_path)
    
    # 打印数据集信息
    data_manager.print_info()
    
    # 创建数据加载器（启用数据增广）
    batch_size = 16
    train_loader, test_loader = data_manager.get_dataloaders(
        batch_size=batch_size, 
        num_workers=0,  # 在某些环境中设置为0避免多进程问题
        shuffle=True,
        augment_train=True  # 启用训练数据增广
    )
    
    # 创建模型
    model = SimpleCNN(num_classes=4).to(device)  # 修改为4类
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练设置
    num_epochs = 15
    best_acc = 0
    
    print(f"\n开始训练 {num_epochs} 个epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)
        
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 测试
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), '/workspaces/codespaces-jupyter/project/best_model.pth')
            print(f"保存最佳模型，准确率: {best_acc:.2f}%")
    
    print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")
    
    # 详细的测试结果分析
    print("\n" + "=" * 60)
    print("详细测试结果分析")
    print("=" * 60)
    
    model.eval()
    class_names = ['仰卧', '俯卧', '左侧卧', '右侧卧']  # 修改为4类
    
    # 计算每个类别的准确率
    class_correct = [0] * 4  # 修改为4类
    class_total = [0] * 4
    
    with torch.no_grad():
        for data, labels, persons in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    print("各类别准确率:")
    for i in range(4):  # 修改为4类
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f'{class_names[i]}: {class_correct[i]}/{class_total[i]} = {acc:.2f}%')
        else:
            print(f'{class_names[i]}: 0/0 = N/A')

if __name__ == "__main__":
    main()
