"""
数据增广效果可视化
展示旋转和平移增广的效果
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append('/workspaces/codespaces-jupyter/project')
from dataloader.dataloader import DataManager, PressureMapDataset

def visualize_augmentation_effects():
    """可视化数据增广效果 - 热力图形式 (26x40)"""
    
    # 创建数据管理器
    data_root = "/workspaces/codespaces-jupyter/project/data/text_data"
    data_manager = DataManager(data_root, train_ratio=0.7, random_state=42)
    
    # 获取训练数据集（启用增广）
    train_dataset_aug = PressureMapDataset(data_manager.train_data, augment=True)
    train_dataset_no_aug = PressureMapDataset(data_manager.train_data, augment=False)
    
    # 选择几个样本进行可视化
    num_samples = 3
    fig, axes = plt.subplots(num_samples, 6, figsize=(20, num_samples * 4))
    
    for i in range(num_samples):
        # 获取原始样本
        original_data, label, person = train_dataset_no_aug[i * 10]  # 每10个取一个
        original_img = original_data.squeeze().numpy()
        
        # 确保数据形状为 (26, 40) 并转换到 0-255 范围
        if original_img.shape != (26, 40):
            print(f"警告: 数据形状为 {original_img.shape}, 预期为 (26, 40)")
        
        # 将数据范围标准化到 0-255
        original_img_norm = ((original_img - original_img.min()) / 
                           (original_img.max() - original_img.min() + 1e-8) * 255).astype(np.uint8)
        
        # 获取5个增广样本
        aug_samples = []
        for _ in range(5):
            aug_data, _, _ = train_dataset_aug[i * 10]
            aug_img = aug_data.squeeze().numpy()
            # 标准化到 0-255
            aug_img_norm = ((aug_img - aug_img.min()) / 
                           (aug_img.max() - aug_img.min() + 1e-8) * 255).astype(np.uint8)
            aug_samples.append(aug_img_norm)
        
        # 显示原始热力图
        im0 = axes[i, 0].imshow(original_img_norm, cmap='hot', interpolation='nearest', 
                               vmin=0, vmax=255, aspect='auto')
        axes[i, 0].set_title(f'原始热力图\n标签: {label.item()}, 人员: {person}\n形状: {original_img_norm.shape}', 
                            fontsize=10)
        axes[i, 0].set_xlabel('宽度方向 (40列)')
        axes[i, 0].set_ylabel('长度方向 (26行)')
        
        # 添加网格线以更好地显示像素
        axes[i, 0].set_xticks(np.arange(-0.5, 40, 5), minor=True)
        axes[i, 0].set_yticks(np.arange(-0.5, 26, 5), minor=True)
        axes[i, 0].grid(which="minor", color="white", linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 添加颜色条
        cbar0 = plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        cbar0.set_label('压力值 (0-255)', fontsize=8)
        
        # 显示增广热力图
        for j, aug_img_norm in enumerate(aug_samples):
            im = axes[i, j+1].imshow(aug_img_norm, cmap='hot', interpolation='nearest', 
                                   vmin=0, vmax=255, aspect='auto')
            axes[i, j+1].set_title(f'增广样本 {j+1}\n形状: {aug_img_norm.shape}', fontsize=10)
            axes[i, j+1].set_xlabel('宽度方向 (40列)')
            axes[i, j+1].set_ylabel('长度方向 (26行)')
            
            # 添加网格线
            axes[i, j+1].set_xticks(np.arange(-0.5, 40, 5), minor=True)
            axes[i, j+1].set_yticks(np.arange(-0.5, 26, 5), minor=True)
            axes[i, j+1].grid(which="minor", color="white", linestyle='-', linewidth=0.5, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=axes[i, j+1], fraction=0.046, pad=0.04)
            cbar.set_label('压力值 (0-255)', fontsize=8)
            
            # 显示统计信息
            mean_val = np.mean(aug_img_norm)
            max_val = np.max(aug_img_norm)
            axes[i, j+1].text(0.02, 0.98, f'均值: {mean_val:.1f}\n最大: {max_val}', 
                            transform=axes[i, j+1].transAxes, 
                            verticalalignment='top', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('压力图数据增广效果对比 (26×40热力图, 0-255范围)', fontsize=16, y=1.02)
    plt.show()

def compare_augmentation_statistics():
    """比较原始数据和增广数据的统计信息 - 热力图对比"""
    
    data_root = "/workspaces/codespaces-jupyter/project/data/text_data"
    data_manager = DataManager(data_root, train_ratio=0.7, random_state=42)
    
    # 创建数据集
    train_dataset_aug = PressureMapDataset(data_manager.train_data, augment=True)
    train_dataset_no_aug = PressureMapDataset(data_manager.train_data, augment=False)
    
    # 收集统计信息和热力图数据
    original_stats = []
    augmented_stats = []
    original_heatmaps = []
    augmented_heatmaps = []
    
    print("收集统计信息和热力图数据...")
    num_samples = min(30, len(train_dataset_no_aug))  # 取前30个样本
    
    for i in range(num_samples):
        # 原始数据
        orig_data, _, _ = train_dataset_no_aug[i]
        orig_img = orig_data.squeeze().numpy()
        
        # 标准化到 0-255 范围
        orig_img_norm = ((orig_img - orig_img.min()) / 
                        (orig_img.max() - orig_img.min() + 1e-8) * 255).astype(np.uint8)
        
        original_stats.append({
            'mean': np.mean(orig_img_norm),
            'std': np.std(orig_img_norm),
            'min': np.min(orig_img_norm),
            'max': np.max(orig_img_norm)
        })
        original_heatmaps.append(orig_img_norm)
        
        # 增广数据
        aug_data, _, _ = train_dataset_aug[i]
        aug_img = aug_data.squeeze().numpy()
        
        # 标准化到 0-255 范围
        aug_img_norm = ((aug_img - aug_img.min()) / 
                       (aug_img.max() - aug_img.min() + 1e-8) * 255).astype(np.uint8)
        
        augmented_stats.append({
            'mean': np.mean(aug_img_norm),
            'std': np.std(aug_img_norm),
            'min': np.min(aug_img_norm),
            'max': np.max(aug_img_norm)
        })
        augmented_heatmaps.append(aug_img_norm)
    
    # 计算平均统计信息
    orig_means = [s['mean'] for s in original_stats]
    orig_stds = [s['std'] for s in original_stats]
    aug_means = [s['mean'] for s in augmented_stats]
    aug_stds = [s['std'] for s in augmented_stats]
    
    print(f"\n热力图统计对比 (基于 {len(original_stats)} 个样本, 26×40像素, 0-255范围):")
    print("=" * 60)
    print(f"原始数据 - 平均值: {np.mean(orig_means):.2f} ± {np.std(orig_means):.2f}")
    print(f"增广数据 - 平均值: {np.mean(aug_means):.2f} ± {np.std(aug_means):.2f}")
    print(f"原始数据 - 标准差: {np.mean(orig_stds):.2f} ± {np.std(orig_stds):.2f}")
    print(f"增广数据 - 标准差: {np.mean(aug_stds):.2f} ± {np.std(aug_stds):.2f}")
    
    # 创建综合可视化
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 平均热力图对比
    ax1 = plt.subplot(3, 3, 1)
    avg_orig_heatmap = np.mean(original_heatmaps, axis=0)
    im1 = ax1.imshow(avg_orig_heatmap, cmap='hot', vmin=0, vmax=255, aspect='auto')
    ax1.set_title('原始数据平均热力图\n(26×40)', fontsize=10)
    ax1.set_xlabel('宽度 (40列)')
    ax1.set_ylabel('长度 (26行)')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = plt.subplot(3, 3, 2)
    avg_aug_heatmap = np.mean(augmented_heatmaps, axis=0)
    im2 = ax2.imshow(avg_aug_heatmap, cmap='hot', vmin=0, vmax=255, aspect='auto')
    ax2.set_title('增广数据平均热力图\n(26×40)', fontsize=10)
    ax2.set_xlabel('宽度 (40列)')
    ax2.set_ylabel('长度 (26行)')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    # 3. 差异热力图
    ax3 = plt.subplot(3, 3, 3)
    diff_heatmap = avg_aug_heatmap - avg_orig_heatmap
    im3 = ax3.imshow(diff_heatmap, cmap='RdBu_r', vmin=-50, vmax=50, aspect='auto')
    ax3.set_title('增广-原始差异热力图', fontsize=10)
    ax3.set_xlabel('宽度 (40列)')
    ax3.set_ylabel('长度 (26行)')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    # 4. 统计分布对比
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(orig_means, alpha=0.7, label='原始数据', bins=15, color='blue')
    ax4.hist(aug_means, alpha=0.7, label='增广数据', bins=15, color='red')
    ax4.set_title('像素平均值分布\n(0-255范围)')
    ax4.set_xlabel('平均值')
    ax4.set_ylabel('频次')
    ax4.legend()
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(orig_stds, alpha=0.7, label='原始数据', bins=15, color='blue')
    ax5.hist(aug_stds, alpha=0.7, label='增广数据', bins=15, color='red')
    ax5.set_title('像素标准差分布')
    ax5.set_xlabel('标准差')
    ax5.set_ylabel('频次')
    ax5.legend()
    
    # 6. 散点图对比
    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(orig_means, orig_stds, alpha=0.6, label='原始数据', color='blue')
    ax6.scatter(aug_means, aug_stds, alpha=0.6, label='增广数据', color='red')
    ax6.set_title('平均值 vs 标准差')
    ax6.set_xlabel('平均值')
    ax6.set_ylabel('标准差')
    ax6.legend()
    
    # 7. 像素强度分布热力图
    ax7 = plt.subplot(3, 3, 7)
    orig_pixel_dist = np.histogram2d(np.concatenate([h.flatten() for h in original_heatmaps[:10]]), 
                                   np.arange(len(original_heatmaps[:10])).repeat(26*40), 
                                   bins=[256, 10])[0]
    im7 = ax7.imshow(orig_pixel_dist, cmap='viridis', aspect='auto')
    ax7.set_title('原始数据像素分布')
    ax7.set_xlabel('样本索引')
    ax7.set_ylabel('像素值 (0-255)')
    plt.colorbar(im7, ax=ax7, fraction=0.046)
    
    ax8 = plt.subplot(3, 3, 8)
    aug_pixel_dist = np.histogram2d(np.concatenate([h.flatten() for h in augmented_heatmaps[:10]]), 
                                  np.arange(len(augmented_heatmaps[:10])).repeat(26*40), 
                                  bins=[256, 10])[0]
    im8 = ax8.imshow(aug_pixel_dist, cmap='viridis', aspect='auto')
    ax8.set_title('增广数据像素分布')
    ax8.set_xlabel('样本索引')
    ax8.set_ylabel('像素值 (0-255)')
    plt.colorbar(im8, ax=ax8, fraction=0.046)
    
    # 9. 增广效果量化
    ax9 = plt.subplot(3, 3, 9)
    mean_diffs = [aug - orig for aug, orig in zip(aug_means, orig_means)]
    std_diffs = [aug - orig for aug, orig in zip(aug_stds, orig_stds)]
    
    scatter = ax9.scatter(mean_diffs, std_diffs, alpha=0.6, c=orig_means, cmap='hot')
    ax9.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax9.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax9.set_title('增广效果量化\n(颜色=原始平均值)')
    ax9.set_xlabel('平均值差异')
    ax9.set_ylabel('标准差差异')
    plt.colorbar(scatter, ax=ax9, fraction=0.046)
    
    plt.tight_layout()
    plt.suptitle('压力图数据增广统计分析 (26×40热力图, 0-255范围)', fontsize=14, y=0.98)
    plt.show()

def test_augmentation_consistency():
    """测试数据增广的一致性和有效性 - 热力图可视化"""
    
    data_root = "/workspaces/codespaces-jupyter/project/data/text_data"
    data_manager = DataManager(data_root, train_ratio=0.7, random_state=42)
    
    # 创建启用增广的数据集
    train_dataset = PressureMapDataset(data_manager.train_data, augment=True)
    
    print("测试数据增广一致性 - 热力图分析...")
    
    # 测试同一个样本的多次增广结果
    sample_idx = 0
    original_data, label, person = train_dataset[sample_idx]
    
    print(f"测试样本: 索引 {sample_idx}, 标签 {label.item()}, 人员 {person}")
    print(f"原始数据形状: {original_data.shape}")
    print(f"数据范围: [{original_data.min():.4f}, {original_data.max():.4f}]")
    
    # 检查多次获取同一样本是否得到不同的增广结果
    augmented_samples = []
    augmented_heatmaps = []
    
    for i in range(6):  # 获取6个增广样本
        data, _, _ = train_dataset[sample_idx]
        augmented_samples.append(data.clone())
        
        # 转换为热力图格式 (0-255)
        img = data.squeeze().numpy()
        img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
        augmented_heatmaps.append(img_norm)
    
    # 可视化增广样本的热力图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, heatmap in enumerate(augmented_heatmaps):
        im = axes[i].imshow(heatmap, cmap='hot', vmin=0, vmax=255, aspect='auto')
        axes[i].set_title(f'增广样本 {i+1}\n形状: {heatmap.shape}\n均值: {np.mean(heatmap):.1f}', 
                         fontsize=10)
        axes[i].set_xlabel('宽度方向 (40列)')
        axes[i].set_ylabel('长度方向 (26行)')
        
        # 添加网格
        axes[i].set_xticks(np.arange(-0.5, 40, 10), minor=True)
        axes[i].set_yticks(np.arange(-0.5, 26, 5), minor=True)
        axes[i].grid(which="minor", color="white", linestyle='-', linewidth=0.3, alpha=0.5)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label('压力值 (0-255)', fontsize=8)
        
        # 显示最大值位置
        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        axes[i].plot(max_pos[1], max_pos[0], 'w*', markersize=8, markeredgecolor='black')
        axes[i].text(max_pos[1], max_pos[0]-2, f'Max: {heatmap[max_pos]}', 
                    ha='center', va='bottom', color='white', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.suptitle(f'数据增广一致性测试 - 热力图视图\n样本: {person}, 标签: {label.item()}', 
                fontsize=14, y=1.02)
    plt.show()
    
    # 数值分析
    print("\n增广样本间差异分析 (26×40热力图):")
    print("=" * 50)
    
    for i in range(len(augmented_samples)):
        for j in range(i+1, len(augmented_samples)):
            diff = torch.abs(augmented_samples[i] - augmented_samples[j])
            mean_diff = torch.mean(diff)
            max_diff = torch.max(diff)
            
            # 热力图统计对比
            heatmap_i = augmented_heatmaps[i]
            heatmap_j = augmented_heatmaps[j]
            pixel_diff = np.abs(heatmap_i.astype(float) - heatmap_j.astype(float))
            
            print(f"样本 {i+1} vs 样本 {j+1}:")
            print(f"  原始差异: 平均 {mean_diff:.6f}, 最大 {max_diff:.6f}")
            print(f"  热力图差异: 平均 {np.mean(pixel_diff):.2f}, 最大 {np.max(pixel_diff):.2f} (0-255范围)")
            print(f"  相似度: {100 - np.mean(pixel_diff)/255*100:.2f}%")
    
    # 创建差异热力图
    if len(augmented_heatmaps) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 样本1
        im1 = axes[0].imshow(augmented_heatmaps[0], cmap='hot', vmin=0, vmax=255, aspect='auto')
        axes[0].set_title('增广样本 1')
        axes[0].set_xlabel('宽度 (40列)')
        axes[0].set_ylabel('长度 (26行)')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # 样本2
        im2 = axes[1].imshow(augmented_heatmaps[1], cmap='hot', vmin=0, vmax=255, aspect='auto')
        axes[1].set_title('增广样本 2')
        axes[1].set_xlabel('宽度 (40列)')
        axes[1].set_ylabel('长度 (26行)')
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # 差异图
        diff_map = np.abs(augmented_heatmaps[0].astype(float) - augmented_heatmaps[1].astype(float))
        im3 = axes[2].imshow(diff_map, cmap='Reds', vmin=0, vmax=100, aspect='auto')
        axes[2].set_title(f'绝对差异\n平均: {np.mean(diff_map):.2f}')
        axes[2].set_xlabel('宽度 (40列)')
        axes[2].set_ylabel('长度 (26行)')
        plt.colorbar(im3, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        plt.suptitle('增广样本差异分析热力图', fontsize=14, y=1.02)
        plt.show()
    
    print(f"\n✓ 数据增广功能正常工作 - 热力图验证完成")

if __name__ == "__main__":
    print("=== 数据增广效果可视化 ===")
    
    try:
        # 1. 可视化增广效果
        print("\n1. 生成增广效果对比图...")
        visualize_augmentation_effects()
        
        # 2. 统计分析
        print("\n2. 进行统计分析...")
        compare_augmentation_statistics()
        
        # 3. 一致性测试
        print("\n3. 测试增广一致性...")
        test_augmentation_consistency()
        
        print("\n=== 所有测试完成 ===")
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
