import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
from .get_data import build_dataset

class PressureMapDataset(Dataset):
    """压力图数据集"""
    
    def __init__(self, data_list, transform=None, augment=False):
        """
        Args:
            data_list: 数据列表 [(压力图, 标签, 人员名称), ...]
            transform: 数据变换
            augment: 是否启用数据增广
        """
        self.data_list = data_list
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        pressure_map, label, person_name = self.data_list[idx]
        
        # 转换为张量
        pressure_map = torch.FloatTensor(pressure_map)
        label = torch.LongTensor([label - 1])[0]  # 转换为0-3的标签（4类）
        
        # 添加通道维度 (1, 40, 26)
        pressure_map = pressure_map.unsqueeze(0)
        
        # 数据增广
        if self.augment:
            pressure_map = self._apply_augmentation(pressure_map)
        
        if self.transform:
            pressure_map = self.transform(pressure_map)
            
        return pressure_map, label, person_name
    
    def _apply_augmentation(self, pressure_map):
        """
        应用数据增广
        
        Args:
            pressure_map: 输入压力图张量 (1, H, W)
            
        Returns:
            增广后的压力图张量
        """
        # 随机旋转（-15到15度）
        if torch.rand(1) > 0.5:
            angle = torch.rand(1) * 30 - 15  # -15到15度
            pressure_map = self._rotate_tensor(pressure_map, angle.item())
        
        # 随机平移（小范围）
        if torch.rand(1) > 0.5:
            # 在高度和宽度方向分别平移最多2个像素
            shift_h = torch.randint(-2, 3, (1,)).item()
            shift_w = torch.randint(-2, 3, (1,)).item()
            pressure_map = self._translate_tensor(pressure_map, shift_h, shift_w)
        
        return pressure_map
    
    def _rotate_tensor(self, tensor, angle):
        """
        旋转张量
        
        Args:
            tensor: 输入张量 (1, H, W)
            angle: 旋转角度（度）
            
        Returns:
            旋转后的张量
        """
        # 转换角度为弧度
        angle_rad = torch.tensor(angle * np.pi / 180.0)
        
        # 创建旋转矩阵
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)
        
        # 2x3仿射变换矩阵
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32).unsqueeze(0)
        
        # 创建网格并应用变换
        grid = F.affine_grid(rotation_matrix, tensor.unsqueeze(0).size(), align_corners=False)
        rotated = F.grid_sample(tensor.unsqueeze(0), grid, mode='bilinear', 
                               padding_mode='zeros', align_corners=False)
        
        return rotated.squeeze(0)
    
    def _translate_tensor(self, tensor, shift_h, shift_w):
        """
        平移张量
        
        Args:
            tensor: 输入张量 (1, H, W)
            shift_h: 垂直方向平移像素数
            shift_w: 水平方向平移像素数
            
        Returns:
            平移后的张量
        """
        # 标准化平移量
        H, W = tensor.shape[1], tensor.shape[2]
        shift_h_norm = 2.0 * shift_h / H
        shift_w_norm = 2.0 * shift_w / W
        
        # 2x3仿射变换矩阵（平移）
        translation_matrix = torch.tensor([
            [1, 0, shift_w_norm],
            [0, 1, shift_h_norm]
        ], dtype=torch.float32).unsqueeze(0)
        
        # 创建网格并应用变换
        grid = F.affine_grid(translation_matrix, tensor.unsqueeze(0).size(), align_corners=False)
        translated = F.grid_sample(tensor.unsqueeze(0), grid, mode='bilinear', 
                                  padding_mode='zeros', align_corners=False)
        
        return translated.squeeze(0)

def split_dataset_by_person(all_data, person_list, train_ratio=0.7, random_state=42):
    """
    按人员划分数据集
    
    Args:
        all_data: 所有数据 [(压力图, 标签, 人员名称), ...]
        person_list: 人员列表
        train_ratio: 训练集比例
        random_state: 随机种子
        
    Returns:
        train_data, test_data: 训练和测试数据
    """
    # 按人员划分
    train_persons, test_persons = train_test_split(
        person_list, 
        train_size=train_ratio, 
        random_state=random_state
    )
    
    print(f"训练集人员 ({len(train_persons)} 人): {train_persons}")
    print(f"测试集人员 ({len(test_persons)} 人): {test_persons}")
    
    # 根据人员划分数据
    train_data = []
    test_data = []
    
    for pressure_map, label, person_name in all_data:
        if person_name in train_persons:
            train_data.append((pressure_map, label, person_name))
        else:
            test_data.append((pressure_map, label, person_name))
    
    print(f"训练集样本数: {len(train_data)}")
    print(f"测试集样本数: {len(test_data)}")
    
    # 统计训练集和测试集的类别分布
    def count_labels(data):
        label_counts = {}
        for _, label, _ in data:
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts
    
    train_label_counts = count_labels(train_data)
    test_label_counts = count_labels(test_data)
    
    label_names = {1: "仰卧", 2: "俯卧", 3: "左侧卧", 4: "右侧卧"}
    
    print("\n训练集类别分布:")
    for label, count in sorted(train_label_counts.items()):
        print(f"  {label} ({label_names[label]}): {count} 个样本")
    
    print("\n测试集类别分布:")
    for label, count in sorted(test_label_counts.items()):
        print(f"  {label} ({label_names[label]}): {count} 个样本")
    
    return train_data, test_data

def create_dataloaders(train_data, test_data, batch_size=32, num_workers=4, shuffle=True, augment_train=True):
    """
    创建数据加载器
    
    Args:
        train_data: 训练数据
        test_data: 测试数据
        batch_size: 批次大小
        num_workers: 工作进程数
        shuffle: 是否打乱训练数据
        augment_train: 是否对训练数据启用数据增广
        
    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """
    # 训练集启用数据增广，测试集不启用
    train_dataset = PressureMapDataset(train_data, augment=augment_train)
    test_dataset = PressureMapDataset(test_data, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_dataset_info(dataset):
    """获取数据集信息"""
    if len(dataset) == 0:
        return
    
    sample_data, sample_label, sample_person = dataset[0]
    print(f"样本形状: {sample_data.shape}")
    print(f"标签范围: 0-3 (对应原始标签 1-4)")
    print(f"数据类型: {sample_data.dtype}")
    print(f"标签类型: {sample_label.dtype}")
    print(f"数据增广: {'启用' if hasattr(dataset, 'augment') and dataset.augment else '未启用'}")

class DataManager:
    """数据管理类"""
    
    def __init__(self, data_root, train_ratio=0.7, random_state=42):
        """
        Args:
            data_root: 数据根目录
            train_ratio: 训练集比例
            random_state: 随机种子
        """
        self.data_root = data_root
        self.train_ratio = train_ratio
        self.random_state = random_state
        
        # 构建数据集
        print("开始构建数据集...")
        self.all_data, self.person_list = build_dataset(data_root)
        
        # 划分数据集
        print("\n开始划分数据集...")
        self.train_data, self.test_data = split_dataset_by_person(
            self.all_data, self.person_list, train_ratio, random_state
        )
        
    def get_dataloaders(self, batch_size=32, num_workers=4, shuffle=True, augment_train=True):
        """获取数据加载器"""
        return create_dataloaders(
            self.train_data, self.test_data, 
            batch_size, num_workers, shuffle, augment_train
        )
    
    def get_datasets(self, augment_train=True):
        """获取数据集对象"""
        train_dataset = PressureMapDataset(self.train_data, augment=augment_train)
        test_dataset = PressureMapDataset(self.test_data, augment=False)
        return train_dataset, test_dataset
    
    def print_info(self):
        """打印数据集信息"""
        print(f"\n=== 数据集信息 ===")
        print(f"总样本数: {len(self.all_data)}")
        print(f"总人数: {len(self.person_list)}")
        print(f"训练集人数: {len(set([person for _, _, person in self.train_data]))}")
        print(f"测试集人数: {len(set([person for _, _, person in self.test_data]))}")
        print(f"训练集样本数: {len(self.train_data)}")
        print(f"测试集样本数: {len(self.test_data)}")
        
        if len(self.train_data) > 0:
            train_dataset = PressureMapDataset(self.train_data, augment=True)
            print(f"\n训练集样本信息:")
            get_dataset_info(train_dataset)

if __name__ == "__main__":
    # 测试数据管理器
    data_root = "/workspaces/codespaces-jupyter/project/data/text_data"
    
    # 创建数据管理器
    data_manager = DataManager(data_root, train_ratio=0.7, random_state=42)
    data_manager.print_info()
    
    # 创建数据加载器
    train_loader, test_loader = data_manager.get_dataloaders(batch_size=8)
    
    print(f"\n=== 数据加载器测试 ===")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 测试一个批次
    for batch_idx, (data, labels, persons) in enumerate(train_loader):
        print(f"\n第 {batch_idx + 1} 个批次:")
        print(f"数据形状: {data.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签值: {labels.tolist()}")
        print(f"人员: {persons}")
        
        if batch_idx == 0:  # 只测试第一个批次
            break
