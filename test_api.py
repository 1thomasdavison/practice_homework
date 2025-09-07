#!/usr/bin/env python3
"""
测试Web应用程序的API功能
"""

import requests
import json
import base64
from PIL import Image
import io

# 测试服务器地址
BASE_URL = "http://localhost:5000"

def test_api():
    """测试Web API的各个功能"""
    
    print("🧪 开始测试睡姿展示系统API...")
    
    # 1. 测试主页
    print("\n1. 测试主页访问...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ 主页访问成功")
        else:
            print(f"❌ 主页访问失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 主页访问失败: {e}")
    
    # 2. 测试用户列表
    print("\n2. 测试用户列表API...")
    try:
        response = requests.get(f"{BASE_URL}/api/users")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 用户列表获取成功，共 {data['total']} 个用户")
            print(f"前5个用户: {data['users'][:5]}")
        else:
            print(f"❌ 用户列表获取失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 用户列表获取失败: {e}")
    
    # 3. 测试统计信息
    print("\n3. 测试统计信息API...")
    try:
        response = requests.get(f"{BASE_URL}/api/statistics")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 统计信息获取成功")
            print(f"总用户数: {data['total_users']}")
            print(f"总样本数: {data['total_samples']}")
            print(f"睡姿统计: {data['posture_stats']}")
        else:
            print(f"❌ 统计信息获取失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 统计信息获取失败: {e}")
    
    # 4. 测试特定用户的睡姿
    print("\n4. 测试用户睡姿API...")
    test_user = "czy"  # 使用第一个用户
    try:
        response = requests.get(f"{BASE_URL}/api/user/{test_user}/postures")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 用户 {test_user} 睡姿获取成功")
            print(f"可用睡姿: {data['postures']}")
        else:
            print(f"❌ 用户睡姿获取失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 用户睡姿获取失败: {e}")
    
    # 5. 测试热力图生成
    print("\n5. 测试热力图生成API...")
    try:
        response = requests.get(f"{BASE_URL}/api/heatmap/{test_user}/2")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 热力图生成成功")
            print(f"用户: {data['user']}, 睡姿: {data['posture']}")
            print(f"统计信息: 最大值={data['statistics']['max']:.2f}, 平均值={data['statistics']['mean']:.2f}")
            
            # 验证图片数据
            if data['image']:
                print("✅ 图片数据生成成功")
                # 可以保存图片进行验证
                img_data = base64.b64decode(data['image'])
                with open(f"/workspaces/codespaces-jupyter/project/test_heatmap_{test_user}.png", "wb") as f:
                    f.write(img_data)
                print(f"✅ 图片已保存为 test_heatmap_{test_user}.png")
            else:
                print("❌ 图片数据为空")
        else:
            print(f"❌ 热力图生成失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 热力图生成失败: {e}")
    
    # 6. 测试睡姿对比
    print("\n6. 测试睡姿对比API...")
    try:
        response = requests.get(f"{BASE_URL}/api/compare/{test_user}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 睡姿对比生成成功")
            print(f"对比睡姿: {data['postures']}")
            print(f"睡姿数量: {data['total_postures']}")
            
            # 验证对比图片
            if data['image']:
                print("✅ 对比图片数据生成成功")
                img_data = base64.b64decode(data['image'])
                with open(f"/workspaces/codespaces-jupyter/project/test_compare_{test_user}.png", "wb") as f:
                    f.write(img_data)
                print(f"✅ 对比图片已保存为 test_compare_{test_user}.png")
            else:
                print("❌ 对比图片数据为空")
        else:
            print(f"❌ 睡姿对比失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 睡姿对比失败: {e}")
    
    print("\n🎉 API测试完成！")

if __name__ == "__main__":
    test_api()
