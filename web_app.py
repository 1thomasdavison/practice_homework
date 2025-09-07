#!/usr/bin/env python3
"""
睡姿展示系统 - Flask Web应用
支持五类睡姿（仰卧、俯卧、左侧卧、右侧卧、坐姿）的压力图热力图展示
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.font_manager as fm
import io
import base64
from flask import Flask, render_template, jsonify, request, send_from_directory
import glob

# 配置中文字体
def setup_chinese_font():
    """设置中文字体"""
    # 直接指定字体文件路径
    font_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    ]
    
    # 尝试注册字体
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                fm.fontManager.addfont(font_path)
                print(f"✅ 成功注册字体: {font_path}")
            except Exception as e:
                print(f"⚠️ 注册字体失败 {font_path}: {e}")
    
    # 设置matplotlib字体配置
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 强制重建字体缓存
    try:
        fm._rebuild()
        print("✅ 字体缓存重建完成")
    except:
        print("⚠️ 字体缓存重建失败，使用现有缓存")
    
    print(f"✅ 中文字体配置完成")
    
# 初始化中文字体配置
setup_chinese_font()

# 导入自定义模块
from dataloader.get_data_display import get_person_folders, load_pressure_data, compute_average_pressure_map

app = Flask(__name__)

# 全局变量存储数据
dataset = {}
person_list = []
label_names = {
    1: "Supine",
    2: "Prone", 
    3: "Left Side",
    4: "Right Side",
    5: "Sitting"
}

label_names_cn = {
    1: "仰卧",
    2: "俯卧", 
    3: "左侧卧",
    4: "右侧卧",
    5: "坐姿"
}

def get_extended_label_mapping():
    """获取扩展的标签映射关系，包括坐姿"""
    return {
        1: 1,  # 标准仰卧 -> 仰卧
        2: 2,  # 标准俯卧 -> 俯卧  
        3: 3,  # 左侧卧（弯腿) -> 左侧卧
        4: 3,  # 左侧卧（伸腿) -> 左侧卧
        5: 4,  # 右侧卧（弯腿) -> 右侧卧
        6: 4,  # 右侧卧（伸腿) -> 右侧卧
        7: 5,  # 坐姿 -> 坐姿
    }

def load_extended_dataset(data_root):
    """加载包含坐姿的扩展数据集"""
    label_mapping = get_extended_label_mapping()
    person_folders = get_person_folders(data_root)
    
    # 处理sit文件夹中的坐姿数据
    sit_folder = os.path.join(data_root, 'sit')
    
    all_data = {}
    
    print(f"正在加载 {len(person_folders)} 个用户的数据...")
    
    # 处理每个人员的数据
    for person_name in person_folders:
        person_folder_path = os.path.join(data_root, person_name)
        person_data = {}
        
        # 查找该人员的所有txt文件
        txt_files = glob.glob(os.path.join(person_folder_path, f"{person_name}_*.txt"))
        
        for txt_file in txt_files:
            filename = os.path.basename(txt_file)
            try:
                original_label = int(filename.split('_')[-1].split('.')[0])
            except ValueError:
                continue
            
            if original_label not in label_mapping:
                continue
            
            # 加载压力图数据
            frames = load_pressure_data(txt_file)
            if frames is None:
                continue
            
            # 计算平均压力图
            avg_pressure_map = compute_average_pressure_map(frames)
            new_label = label_mapping[original_label]
            
            person_data[new_label] = avg_pressure_map.tolist()  # 转换为列表以便JSON序列化
        
        # 检查sit文件夹中是否有该人员的坐姿数据
        if os.path.exists(sit_folder):
            sit_files = glob.glob(os.path.join(sit_folder, f"{person_name}_*.txt"))
            for sit_file in sit_files:
                frames = load_pressure_data(sit_file)
                if frames is not None:
                    avg_pressure_map = compute_average_pressure_map(frames)
                    person_data[5] = avg_pressure_map.tolist()  # 标签5为坐姿
                    break  # 只取第一个坐姿文件
        
        if person_data:  # 只有当有数据时才添加
            all_data[person_name] = person_data
    
    return all_data, list(all_data.keys())

def create_heatmap_image(pressure_map, title="压力图热力图"):
    """
    创建压力图热力图并返回base64编码的图片
    
    Args:
        pressure_map: 压力图数据 (40, 26)
        title: 图表标题
    
    Returns:
        base64编码的图片字符串
    """
    # 转换为numpy数组
    if isinstance(pressure_map, list):
        pressure_map = np.array(pressure_map)
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 创建热力图
    im = plt.imshow(pressure_map, cmap='hot', interpolation='bilinear', aspect='auto')
    
    # 创建字体属性对象
    try:
        # 尝试使用中文字体
        font_prop = fm.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
        title_font = {'fontproperties': font_prop, 'fontsize': 16, 'fontweight': 'bold'}
        label_font = {'fontproperties': font_prop, 'fontsize': 12}
    except:
        # 如果中文字体不可用，使用默认配置
        title_font = {'fontsize': 16, 'fontweight': 'bold'}
        label_font = {'fontsize': 12}
    
    # 设置标题和标签
    plt.title(title, pad=20, **title_font)
    plt.xlabel('Width Direction (Sensor Columns)', **label_font)
    plt.ylabel('Length Direction (Sensor Rows)', **label_font)
    
    # 添加颜色条
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Pressure Value', **label_font)
    
    # 设置网格
    plt.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存为base64图片
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()  # 关闭图形以释放内存
    
    return img_base64

def get_pressure_statistics(pressure_map):
    """计算压力图统计信息"""
    if isinstance(pressure_map, list):
        pressure_map = np.array(pressure_map)
    
    return {
        'max': float(np.max(pressure_map)),
        'min': float(np.min(pressure_map)),
        'mean': float(np.mean(pressure_map)),
        'std': float(np.std(pressure_map)),
        'shape': pressure_map.shape
    }

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html', 
                         person_list=person_list,
                         label_names=label_names_cn)

@app.route('/api/users')
def get_users():
    """获取所有用户列表"""
    return jsonify({
        'users': person_list,
        'total': len(person_list)
    })

@app.route('/api/user/<user_name>/postures')
def get_user_postures(user_name):
    """获取指定用户的可用睡姿"""
    if user_name not in dataset:
        return jsonify({'error': '用户不存在'}), 404
    
    postures = list(dataset[user_name].keys())
    posture_info = {
        posture: label_names_cn[posture] 
        for posture in postures
    }
    
    return jsonify({
        'user': user_name,
        'postures': posture_info,
        'total': len(postures)
    })

@app.route('/api/heatmap/<user_name>/<int:posture_label>')
def get_heatmap(user_name, posture_label):
    """获取指定用户和睡姿的热力图"""
    if user_name not in dataset:
        return jsonify({'error': '用户不存在'}), 404
    
    if posture_label not in dataset[user_name]:
        return jsonify({'error': f'用户 {user_name} 没有 {label_names_cn.get(posture_label, "未知")} 的数据'}), 404
    
    pressure_map = dataset[user_name][posture_label]
    title = f"{user_name} - {label_names[posture_label]} Pressure Map"
    
    # 生成热力图
    img_base64 = create_heatmap_image(pressure_map, title)
    
    # 获取统计信息
    stats = get_pressure_statistics(pressure_map)
    
    return jsonify({
        'user': user_name,
        'posture': label_names_cn[posture_label], 
        'posture_label': posture_label,
        'image': img_base64,
        'statistics': stats
    })

@app.route('/api/compare/<user_name>')
def compare_postures(user_name):
    """比较指定用户的所有睡姿"""
    if user_name not in dataset:
        return jsonify({'error': '用户不存在'}), 404
    
    user_data = dataset[user_name]
    available_postures = list(user_data.keys())
    n_postures = len(available_postures)
    
    if n_postures == 0:
        return jsonify({'error': '用户没有可比较的睡姿数据'}), 404
    
    # 创建比较图
    cols = min(3, n_postures)
    rows = (n_postures + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    if n_postures == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__iter__') else [axes]
    else:
        axes = axes.flatten()
    
    for i, posture_label in enumerate(available_postures):
        pressure_map = np.array(user_data[posture_label])
        
        im = axes[i].imshow(pressure_map, cmap='hot', interpolation='bilinear', aspect='auto')
        
        # 使用英文标签避免字体问题
        posture_names_en = {
            1: "Supine", 2: "Prone", 3: "Left Side", 4: "Right Side", 5: "Sitting"
        }
        axes[i].set_title(f"{posture_names_en.get(posture_label, 'Unknown')}", fontweight='bold')
        axes[i].set_xlabel('Width')
        axes[i].set_ylabel('Length')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    # 隐藏多余的子图
    for i in range(n_postures, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f"{user_name} - Posture Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存为base64图片
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return jsonify({
        'user': user_name,
        'postures': [label_names_cn[p] for p in available_postures],
        'image': img_base64,
        'total_postures': n_postures
    })

@app.route('/api/statistics')
def get_statistics():
    """获取整体统计信息"""
    posture_stats = {}
    
    for person, data in dataset.items():
        for label in data.keys():
            if label not in posture_stats:
                posture_stats[label] = 0
            posture_stats[label] += 1
    
    return jsonify({
        'total_users': len(person_list),
        'total_samples': sum(posture_stats.values()),
        'posture_stats': {
            label_names_cn[label]: count 
            for label, count in sorted(posture_stats.items())
        }
    })

def initialize_app():
    """初始化应用数据"""
    global dataset, person_list
    
    data_root = "data/text_data"
    print("正在初始化睡姿展示系统...")
    
    try:
        dataset, person_list = load_extended_dataset(data_root)
        print(f"✅ 数据加载完成！共加载 {len(person_list)} 个用户的数据")
        return True
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return False

if __name__ == '__main__':
    # 初始化数据
    if initialize_app():
        print("🚀 启动睡姿展示系统...")
        print("📡 访问地址: http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ 系统初始化失败，无法启动")
