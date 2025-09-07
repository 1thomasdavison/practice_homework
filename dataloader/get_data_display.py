import os
import numpy as np
import glob

def load_pressure_data(txt_file_path):
    """
    加载单个txt文件的压力图数据
    
    Args:
        txt_file_path: txt文件路径
        
    Returns:
        numpy array: 所有帧的压力图数据 (n_frames, 40, 26)
    """
    try:
        # 首先尝试用空格分隔符加载
        try:
            data = np.loadtxt(txt_file_path)
        except ValueError:
            # 如果失败，尝试用逗号分隔符加载
            data = np.loadtxt(txt_file_path, delimiter=',')
        
        # 检查数据维度
        if len(data.shape) == 1:
            # 如果是一维数组，需要重塑
            if data.shape[0] % 26 != 0:
                print(f"Warning: {txt_file_path} 数据长度不是26的倍数")
                truncate_cols = (data.shape[0] // 26) * 26
                data = data[:truncate_cols]
            data = data.reshape(-1, 26)
        
        # 检查数据行数是否为40的倍数
        if data.shape[0] % 40 != 0:
            print(f"Warning: {txt_file_path} 数据行数不是40的倍数，行数: {data.shape[0]}")
            # 截取到最接近40倍数的行数
            truncate_rows = (data.shape[0] // 40) * 40
            data = data[:truncate_rows]
        
        # 检查列数是否为26
        if data.shape[1] != 26:
            print(f"Warning: {txt_file_path} 数据列数不是26，列数: {data.shape[1]}")
            return None
        
        # 重塑为 (n_frames, 40, 26) 的形状
        n_frames = data.shape[0] // 40
        if n_frames == 0:
            print(f"Warning: {txt_file_path} 数据不足40行")
            return None
            
        frames = data.reshape(n_frames, 40, 26)
        
        return frames
    except Exception as e:
        print(f"Error loading {txt_file_path}: {e}")
        return None

def compute_average_pressure_map(frames):
    """
    计算多帧压力图的平均值
    
    Args:
        frames: numpy array of shape (n_frames, 40, 26)
        
    Returns:
        numpy array: 平均压力图 (40, 26)
    """
    return np.mean(frames, axis=0)

def get_label_mapping():
    """
    获取标签映射关系
    原始标签 1-7: 标准仰卧，标准俯卧，左侧卧（弯腿），左侧卧（伸腿），右侧卧（弯腿），右侧卧（伸腿），（床边）坐姿
    新标签 1-5: 仰卧，俯卧，左侧卧，右侧卧，坐姿
    """
    return {
        1: 1,  # 标准仰卧 -> 仰卧
        2: 2,  # 标准俯卧 -> 俯卧
        3: 3,  # 左侧卧（弯腿) -> 左侧卧
        4: 3,  # 左侧卧（伸腿) -> 左侧卧
        5: 4,  # 右侧卧（弯腿) -> 右侧卧
        6: 4,  # 右侧卧（伸腿) -> 右侧卧
        7: 5,  # 坐姿 -> 坐姿
    }
    

def get_person_folders(data_root):
    """
    获取符合条件的人员文件夹
    排除以0308、0307结尾的文件夹和sit命名的文件夹
    """
    person_folders = []
    for folder_name in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder_name)
        if os.path.isdir(folder_path):
            # 排除特定文件夹
            if (folder_name.endswith('0308') or 
                folder_name.endswith('0307') or 
                folder_name.endswith('_6') or
                folder_name == 'sit' or
                folder_name == 'ReadMe'):
                continue
            person_folders.append(folder_name)
    
    return sorted(person_folders)

def process_person_data(person_folder_path, person_name, label_mapping):
    """
    处理单个人的数据
    
    Args:
        person_folder_path: 人员文件夹路径
        person_name: 人员名称
        label_mapping: 标签映射字典
        
    Returns:
        list: [(压力图数据, 标签, 人员名称), ...]
    """
    person_data = []
    
    # 查找该人员的所有txt文件
    txt_files = glob.glob(os.path.join(person_folder_path, f"{person_name}_*.txt"))
    
    for txt_file in txt_files:
        # 从文件名提取原始标签
        filename = os.path.basename(txt_file)
        try:
            original_label = int(filename.split('_')[-1].split('.')[0])
        except ValueError:
            print(f"无法从文件名 {filename} 提取标签")
            continue
        
        # 检查原始标签是否在映射范围内
        if original_label not in label_mapping:
            print(f"标签 {original_label} 不在映射范围内，跳过文件 {filename}")
            continue
        
        # 加载压力图数据
        frames = load_pressure_data(txt_file)
        if frames is None:
            continue
        
        # 计算平均压力图
        avg_pressure_map = compute_average_pressure_map(frames)
        
        # 获取新标签
        new_label = label_mapping[original_label]
        
        person_data.append((avg_pressure_map, new_label, person_name))
        print(f"处理完成: {filename}, 原标签: {original_label} -> 新标签: {new_label}, 形状: {avg_pressure_map.shape}")
    
    return person_data

def build_dataset(data_root):
    """
    构建完整数据集
    
    Args:
        data_root: 数据根目录
        
    Returns:
        list: 所有数据样本 [(压力图, 标签, 人员名称), ...]
        list: 人员名称列表
    """
    label_mapping = get_label_mapping()
    person_folders = get_person_folders(data_root)
    
    print(f"找到 {len(person_folders)} 个符合条件的人员文件夹:")
    for folder in person_folders:
        print(f"  - {folder}")
    
    all_data = []
    
    for person_name in person_folders:
        person_folder_path = os.path.join(data_root, person_name)
        print(f"\n处理人员: {person_name}")
        
        person_data = process_person_data(person_folder_path, person_name, label_mapping)
        all_data.extend(person_data)
        
        print(f"人员 {person_name} 共处理 {len(person_data)} 个样本")
    
    print(f"\n数据集构建完成!")
    print(f"总样本数: {len(all_data)}")
    print(f"总人数: {len(person_folders)}")
    
    # 统计各类别样本数量
    label_counts = {}
    for _, label, _ in all_data:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("各类别样本数量:")
    label_names = {1: "仰卧", 2: "俯卧", 3: "左侧卧", 4: "右侧卧", 5: "坐姿"}
    for label, count in sorted(label_counts.items()):
        print(f"  {label} ({label_names[label]}): {count} 个样本")
    
    return all_data, person_folders

if __name__ == "__main__":
    data_root = "/workspaces/codespaces-jupyter/project/data/text_data"
    dataset, person_list = build_dataset(data_root)
