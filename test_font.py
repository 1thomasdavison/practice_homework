#!/usr/bin/env python3
"""
测试matplotlib中文字体配置
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试数据
data = np.random.rand(10, 10)

# 创建图表
plt.figure(figsize=(10, 8))
im = plt.imshow(data, cmap='hot', interpolation='bilinear')

# 设置中文标题和标签
plt.title('测试中文字体显示 - 压力图热力图', fontsize=16, fontweight='bold')
plt.xlabel('床垫宽度方向 (传感器列)', fontsize=12)
plt.ylabel('床垫长度方向 (传感器行)', fontsize=12)

# 添加颜色条
cbar = plt.colorbar(im)
cbar.set_label('压力值', fontsize=12)

plt.tight_layout()
plt.savefig('/workspaces/codespaces-jupyter/project/font_test.png', dpi=100, bbox_inches='tight')
plt.close()

print("✅ 字体测试完成，图片保存为 font_test.png")

# 打印可用字体信息
print("\n🔍 当前matplotlib字体配置:")
print(f"默认字体族: {plt.rcParams['font.sans-serif']}")
print(f"Unicode负号处理: {plt.rcParams['axes.unicode_minus']}")

# 查找系统中的中文字体
print("\n📝 系统中可用的中文字体:")
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
chinese_fonts = []

for font_path in font_list[:50]:  # 限制检查数量
    try:
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        if 'WenQuanYi' in font_name or '文泉' in font_name:
            chinese_fonts.append(font_name)
            print(f"  - {font_name}")
    except:
        continue

if not chinese_fonts:
    print("  ⚠️ 未找到中文字体")
