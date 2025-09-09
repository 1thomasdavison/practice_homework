"""
MindSpore版本的数据加载器
适配华为Ascend AI处理器
"""

import mindspore.dataset as ds
import numpy as np
from sklearn.model_selection import train_test_split
from .get_data import build_dataset

class PressureMapDataset:
    """压力图数据集 - MindSpore版本"""
    
    def __init__(self, data_list, augment=False):
        """
        Args:
            data_list: 数据列表 [(压力图, 标签, 人员名称), ...]
            augment: 是否启用数据增广
        """
        self.data_list = data_list
        self.augment = augment
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        pressure_map, label, person_name = self.data_list[idx]
        
        # 转换为numpy数组
        pressure_map = np.array(pressure_map, dtype=np.float32)
        label = np.array(label - 1, dtype=np.int32)  # 转换为0-3的标签（4类）
        
        # 添加通道维度 (40, 26) -> (1, 40, 26)
        if len(pressure_map.shape) == 2:
            pressure_map = np.expand_dims(pressure_map, axis=0)
        
        # 数据增广
        if self.augment:
            pressure_map = self._apply_augmentation(pressure_map)
            
        return pressure_map, label, person_name
    
    def _apply_augmentation(self, pressure_map):
        """
        应用数据增广 - MindSpore版本
        
        Args:
            pressure_map: 输入压力图数组 (1, H, W)
            
        Returns:
            增广后的压力图数组
        """
        # 随机旋转（-15到15度）
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            pressure_map = self._rotate_array(pressure_map, angle)
        
        # 随机平移（小范围）
        if np.random.rand() > 0.5:
            shift_h = np.random.randint(-2, 3)
            shift_w = np.random.randint(-2, 3)
            pressure_map = self._translate_array(pressure_map, shift_h, shift_w)
        
        return pressure_map
    
    def _rotate_array(self, array, angle):
        """
        旋转数组
        
        Args:
            array: 输入数组 (1, H, W)
            angle: 旋转角度（度）
            
        Returns:
            旋转后的数组
        """
        from scipy.ndimage import rotate
        # 只旋转图像部分 (H, W)
        rotated_img = rotate(array[0], angle, reshape=False, mode='constant', cval=0.0)
        return np.expand_dims(rotated_img, axis=0)
    
    def _translate_array(self, array, shift_h, shift_w):
        """
        平移数组
        
        Args:
            array: 输入数组 (1, H, W)
            shift_h: 垂直方向平移像素数
            shift_w: 水平方向平移像素数
            
        Returns:
            平移后的数组
        """
        from scipy.ndimage import shift
        # 只平移图像部分 (H, W)
        translated_img = shift(array[0], [shift_h, shift_w], mode='constant', cval=0.0)
        return np.expand_dims(translated_img, axis=0)

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

def create_mindspore_dataset(data_list, batch_size=16, shuffle=True, augment=False):
    """
    创建MindSpore数据集
    
    Args:
        data_list: 数据列表
        batch_size: 批次大小
        shuffle: 是否打乱数据
        augment: 是否启用数据增广
        
    Returns:
        MindSpore数据集对象
    """
    def generator():
        dataset = PressureMapDataset(data_list, augment=augment)
        for i in range(len(dataset)):
            pressure_map, label, person_name = dataset[i]
            yield pressure_map, label
    
    dataset = ds.GeneratorDataset(generator, ["data", "label"])
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data_list))
    
    dataset = dataset.batch(batch_size)
    return dataset

def get_dataset_info(data_list):
    """获取数据集信息"""
    if len(data_list) == 0:
        return
    
    sample_dataset = PressureMapDataset(data_list, augment=True)
    sample_data, sample_label, sample_person = sample_dataset[0]
    
    print(f"样本形状: {sample_data.shape}")
    print(f"标签范围: 0-3 (对应原始标签 1-4)")
    print(f"数据类型: {sample_data.dtype}")
    print(f"标签类型: {sample_label.dtype}")

class DataManager:
    """数据管理类 - MindSpore版本"""
    
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
        
    def get_mindspore_datasets(self, batch_size=16, shuffle=True, augment_train=True):
        """获取MindSpore数据集"""
        train_dataset = create_mindspore_dataset(
            self.train_data, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            augment=augment_train
        )
        
        test_dataset = create_mindspore_dataset(
            self.test_data, 
            batch_size=batch_size, 
            shuffle=False, 
            augment=False
        )
        
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
            print(f"\n训练集样本信息:")
            get_dataset_info(self.train_data)

if __name__ == "__main__":
    # 测试MindSpore数据管理器
    data_root = "/workspaces/codespaces-jupyter/project/data/text_data"
    
    # 创建数据管理器
    data_manager = DataManager(data_root, train_ratio=0.7, random_state=42)
    data_manager.print_info()
    
    # 创建MindSpore数据集
    train_dataset, test_dataset = data_manager.get_mindspore_datasets(batch_size=8)
    
    print(f"\n=== MindSpore数据集测试 ===")
    
    # 测试训练集
    train_size = train_dataset.get_dataset_size()
    test_size = test_dataset.get_dataset_size()
    
    print(f"训练集批次数: {train_size}")
    print(f"测试集批次数: {test_size}")
    
    # 测试一个批次
    for batch_idx, (data, labels) in enumerate(train_dataset.create_tuple_iterator()):
        print(f"\n第 {batch_idx + 1} 个批次:")
        print(f"数据形状: {data.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签值: {labels.asnumpy().tolist()}")
        
        if batch_idx == 0:  # 只测试第一个批次
            break
