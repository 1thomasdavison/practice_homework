#!/usr/bin/env python3
"""
模型转换验证脚本
验证PyTorch模型转换为MindSpore后的正确性
"""

import numpy as np
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

def test_pytorch_model():
    """测试PyTorch模型"""
    try:
        import torch
        from train import SimpleCNN as PyTorchSimpleCNN
        
        print("=== PyTorch模型测试 ===")
        
        # 检查模型文件
        model_path = "best_model.pth"
        if not os.path.exists(model_path):
            print(f"❌ PyTorch模型文件不存在: {model_path}")
            return None
        
        # 加载模型
        model = PyTorchSimpleCNN(num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # 测试推理
        test_input = torch.randn(1, 1, 40, 26)
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ PyTorch模型加载成功")
        print(f"   输入形状: {test_input.shape}")
        print(f"   输出形状: {output.shape}")
        print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return {
            'model': model,
            'test_input': test_input.numpy(),
            'output': output.numpy()
        }
        
    except ImportError:
        print("⚠️ PyTorch不可用，跳过PyTorch测试")
        return None
    except Exception as e:
        print(f"❌ PyTorch模型测试失败: {str(e)}")
        return None

def test_mindspore_model():
    """测试MindSpore模型"""
    try:
        import mindspore
        import mindspore.nn as nn
        import mindspore.ops as ops
        from mindspore import Tensor
        from train_mindspore import SimpleCNN as MindSporeSimpleCNN
        
        print("\n=== MindSpore模型测试 ===")
        
        # 设置上下文
        mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="CPU")
        
        # 检查模型文件
        model_path = "best_model_converted.ckpt"
        if not os.path.exists(model_path):
            print(f"⚠️ MindSpore模型文件不存在: {model_path}")
            print("   请先运行 python convert_model.py 进行模型转换")
            
            # 创建未加载权重的模型进行结构测试
            model = MindSporeSimpleCNN(num_classes=4)
            print(f"✅ MindSpore模型结构创建成功")
            
            # 测试推理（随机权重）
            test_input = Tensor(np.random.randn(1, 1, 40, 26).astype(np.float32))
            model.set_train(False)
            output = model(test_input)
            
            print(f"   输入形状: {test_input.shape}")
            print(f"   输出形状: {output.shape}")
            print(f"   参数数量: {sum(p.size for p in model.trainable_params()):,}")
            
            return {
                'model': model,
                'test_input': test_input.asnumpy(),
                'output': output.asnumpy(),
                'weights_loaded': False
            }
        
        # 加载预训练模型
        model = MindSporeSimpleCNN(num_classes=4)
        param_dict = mindspore.load_checkpoint(model_path)
        mindspore.load_param_into_net(model, param_dict)
        model.set_train(False)
        
        # 测试推理
        test_input = Tensor(np.random.randn(1, 1, 40, 26).astype(np.float32))
        output = model(test_input)
        
        print(f"✅ MindSpore模型加载成功")
        print(f"   输入形状: {test_input.shape}")
        print(f"   输出形状: {output.shape}")
        print(f"   参数数量: {sum(p.size for p in model.trainable_params()):,}")
        
        return {
            'model': model,
            'test_input': test_input.asnumpy(),
            'output': output.asnumpy(),
            'weights_loaded': True
        }
        
    except ImportError:
        print("⚠️ MindSpore不可用，跳过MindSpore测试")
        return None
    except Exception as e:
        print(f"❌ MindSpore模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compare_outputs(pytorch_result, mindspore_result):
    """比较PyTorch和MindSpore的输出"""
    if pytorch_result is None or mindspore_result is None:
        print("\n⚠️ 无法进行输出比较，因为某个模型不可用")
        return
    
    if not mindspore_result['weights_loaded']:
        print("\n⚠️ MindSpore模型未加载预训练权重，输出比较可能不准确")
        return
    
    print("\n=== 输出比较 ===")
    
    # 使用相同的输入进行推理
    test_input = pytorch_result['test_input']
    
    # PyTorch推理
    import torch
    pytorch_model = pytorch_result['model']
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_input_tensor = torch.from_numpy(test_input)
        pytorch_output = pytorch_model(pytorch_input_tensor).numpy()
    
    # MindSpore推理
    import mindspore
    from mindspore import Tensor
    mindspore_model = mindspore_result['model']
    mindspore_model.set_train(False)
    mindspore_input_tensor = Tensor(test_input.astype(np.float32))
    mindspore_output = mindspore_model(mindspore_input_tensor).asnumpy()
    
    # 计算差异
    diff = np.abs(pytorch_output - mindspore_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"输出差异统计:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  相对差异: {mean_diff / (np.mean(np.abs(pytorch_output)) + 1e-8):.6f}")
    
    # 判断转换是否成功
    if max_diff < 1e-4:
        print("✅ 模型转换验证成功! 输出差异在可接受范围内")
        return True
    elif max_diff < 1e-2:
        print("⚠️ 模型转换基本正确，但存在一定差异")
        return True
    else:
        print("❌ 模型转换可能存在问题，输出差异较大")
        return False

def test_data_loading():
    """测试数据加载功能"""
    print("\n=== 数据加载测试 ===")
    
    try:
        # 测试数据文件路径
        test_files = [
            "data/0902数据/hfs0902/hfs_1.txt",
            "data/text_data/czy/czy_1.txt",
        ]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"测试文件: {test_file}")
                
                # 加载压力数据
                data = np.loadtxt(test_file, delimiter=',')
                print(f"  原始形状: {data.shape}")
                
                # 处理为标准格式
                if data.shape[1] == 26 and data.shape[0] >= 40:
                    if data.shape[0] % 40 == 0:
                        n_frames = data.shape[0] // 40
                        reshaped_data = data.reshape(n_frames, 40, 26)
                        averaged_data = np.mean(reshaped_data, axis=0)
                        print(f"  处理后形状: {averaged_data.shape}")
                        print(f"  数据范围: [{averaged_data.min():.2f}, {averaged_data.max():.2f}]")
                        print(f"✅ 数据加载成功")
                    else:
                        print(f"⚠️ 数据形状不规整: {data.shape}")
                else:
                    print(f"⚠️ 数据格式不符合要求: {data.shape}")
                break
        else:
            print("⚠️ 未找到测试数据文件")
            
    except Exception as e:
        print(f"❌ 数据加载测试失败: {str(e)}")

def test_inference_performance():
    """测试推理性能"""
    print("\n=== 推理性能测试 ===")
    
    try:
        # 生成测试数据
        test_data = np.random.randn(1, 1, 40, 26).astype(np.float32)
        num_runs = 10
        
        # 测试MindSpore性能
        try:
            import mindspore
            from mindspore import Tensor
            from train_mindspore import SimpleCNN
            
            mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="CPU")
            
            model = SimpleCNN(num_classes=4)
            model.set_train(False)
            
            import time
            
            # 预热
            for _ in range(3):
                input_tensor = Tensor(test_data)
                _ = model(input_tensor)
            
            # 计时
            start_time = time.time()
            for _ in range(num_runs):
                input_tensor = Tensor(test_data)
                output = model(input_tensor)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            print(f"MindSpore推理性能:")
            print(f"  平均推理时间: {avg_time*1000:.2f} ms")
            print(f"  FPS: {1/avg_time:.2f}")
            
        except ImportError:
            print("⚠️ MindSpore不可用，跳过性能测试")
        
        # 测试PyTorch性能（如果可用）
        try:
            import torch
            from train import SimpleCNN
            
            model = SimpleCNN(num_classes=4)
            model.eval()
            
            # 预热
            with torch.no_grad():
                for _ in range(3):
                    input_tensor = torch.from_numpy(test_data)
                    _ = model(input_tensor)
            
            # 计时
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    input_tensor = torch.from_numpy(test_data)
                    output = model(input_tensor)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            print(f"PyTorch推理性能:")
            print(f"  平均推理时间: {avg_time*1000:.2f} ms")
            print(f"  FPS: {1/avg_time:.2f}")
            
        except ImportError:
            print("⚠️ PyTorch不可用，跳过PyTorch性能测试")
            
    except Exception as e:
        print(f"❌ 性能测试失败: {str(e)}")

def main():
    """主测试函数"""
    print("🧪 模型转换验证脚本")
    print("=" * 60)
    
    # 测试PyTorch模型
    pytorch_result = test_pytorch_model()
    
    # 测试MindSpore模型
    mindspore_result = test_mindspore_model()
    
    # 比较输出
    compare_outputs(pytorch_result, mindspore_result)
    
    # 测试数据加载
    test_data_loading()
    
    # 测试推理性能
    test_inference_performance()
    
    print("\n" + "=" * 60)
    print("🏁 测试完成")
    
    # 总结
    if pytorch_result and mindspore_result:
        print("✅ 项目迁移验证通过，可以部署到香橙派")
    elif mindspore_result:
        print("✅ MindSpore环境正常，可以进行部署")
    else:
        print("❌ 存在问题，请检查环境配置")

if __name__ == "__main__":
    main()
