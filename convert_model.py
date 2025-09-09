"""
PyTorch模型权重转换为MindSpore格式的工具
适用于压力图睡姿分类模型的迁移
"""

import torch
import mindspore
import numpy as np
import os
import sys
from collections import OrderedDict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入模型定义
from train import SimpleCNN as PyTorchSimpleCNN
from train_mindspore import SimpleCNN as MindSporeSimpleCNN

def load_pytorch_model(pytorch_model_path, device='cpu'):
    """
    加载PyTorch模型
    
    Args:
        pytorch_model_path: PyTorch模型文件路径
        device: 设备类型
        
    Returns:
        PyTorch模型实例
    """
    print(f"正在加载PyTorch模型: {pytorch_model_path}")
    
    # 创建模型实例
    model = PyTorchSimpleCNN(num_classes=4)
    
    # 加载权重
    if os.path.exists(pytorch_model_path):
        state_dict = torch.load(pytorch_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("PyTorch模型加载成功!")
        return model
    else:
        raise FileNotFoundError(f"PyTorch模型文件不存在: {pytorch_model_path}")

def create_mindspore_model():
    """
    创建MindSpore模型实例
    
    Returns:
        MindSpore模型实例
    """
    print("正在创建MindSpore模型...")
    model = MindSporeSimpleCNN(num_classes=4)
    print("MindSpore模型创建成功!")
    return model

def convert_pytorch_to_mindspore_weights(pytorch_model, mindspore_model):
    """
    将PyTorch模型权重转换为MindSpore格式
    
    Args:
        pytorch_model: PyTorch模型实例
        mindspore_model: MindSpore模型实例
        
    Returns:
        转换后的权重字典
    """
    print("开始转换模型权重...")
    
    # 获取PyTorch模型的state_dict
    pytorch_state_dict = pytorch_model.state_dict()
    
    # 获取MindSpore模型的参数名称
    mindspore_param_dict = {param.name: param for param in mindspore_model.trainable_params()}
    
    # 权重名称映射
    name_mapping = create_weight_name_mapping(pytorch_state_dict, mindspore_param_dict)
    
    # 转换权重
    converted_weights = []
    
    for pytorch_name, mindspore_name in name_mapping.items():
        if pytorch_name in pytorch_state_dict and mindspore_name in mindspore_param_dict:
            pytorch_weight = pytorch_state_dict[pytorch_name].detach().numpy()
            
            print(f"转换: {pytorch_name} -> {mindspore_name}")
            print(f"  形状: {pytorch_weight.shape}")
            
            # 创建MindSpore参数
            mindspore_param = mindspore_param_dict[mindspore_name]
            mindspore_tensor = mindspore.Tensor(pytorch_weight, mindspore_param.dtype)
            
            converted_weights.append({
                'name': mindspore_name,
                'data': mindspore_tensor
            })
        else:
            print(f"警告: 无法找到对应的权重 - PyTorch: {pytorch_name}, MindSpore: {mindspore_name}")
    
    print(f"成功转换 {len(converted_weights)} 个权重参数")
    return converted_weights

def create_weight_name_mapping(pytorch_state_dict, mindspore_param_dict):
    """
    创建PyTorch和MindSpore权重名称的映射关系
    
    Args:
        pytorch_state_dict: PyTorch模型权重字典
        mindspore_param_dict: MindSpore模型参数字典
        
    Returns:
        权重名称映射字典
    """
    print("创建权重名称映射...")
    
    # 打印PyTorch权重名称
    print("\nPyTorch权重名称:")
    for name in pytorch_state_dict.keys():
        print(f"  {name}: {pytorch_state_dict[name].shape}")
    
    # 打印MindSpore权重名称
    print("\nMindSpore权重名称:")
    for name in mindspore_param_dict.keys():
        print(f"  {name}: {mindspore_param_dict[name].shape}")
    
    # 创建映射关系
    mapping = {}
    
    # 卷积层映射
    conv_layers = [
        ("conv_layers.0.", "conv_layers.0."),      # 第一层卷积
        ("conv_layers.2.", "conv_layers.2."),      # 第一层BN
        ("conv_layers.3.", "conv_layers.3."),      # 第二层卷积
        ("conv_layers.5.", "conv_layers.5."),      # 第二层BN
        ("conv_layers.7.", "conv_layers.7."),      # 第三层卷积
        ("conv_layers.9.", "conv_layers.9."),      # 第三层BN
    ]
    
    # 全连接层映射
    fc_layers = [
        ("fc_layers.1.", "fc_layers.1."),          # 第一个全连接层
        ("fc_layers.4.", "fc_layers.4."),          # 第二个全连接层
        ("fc_layers.6.", "fc_layers.6."),          # 输出层
    ]
    
    # 创建具体的映射
    for pytorch_prefix, mindspore_prefix in conv_layers + fc_layers:
        for pytorch_name in pytorch_state_dict.keys():
            if pytorch_name.startswith(pytorch_prefix):
                # 提取参数后缀 (weight, bias等)
                suffix = pytorch_name[len(pytorch_prefix):]
                mindspore_name = mindspore_prefix + suffix
                
                # 检查MindSpore中是否存在对应参数
                if mindspore_name in mindspore_param_dict:
                    mapping[pytorch_name] = mindspore_name
                else:
                    # 尝试其他可能的名称
                    for ms_name in mindspore_param_dict.keys():
                        if ms_name.endswith('.' + suffix) and mindspore_prefix.rstrip('.') in ms_name:
                            mapping[pytorch_name] = ms_name
                            break
    
    print(f"\n创建了 {len(mapping)} 个权重映射:")
    for pt_name, ms_name in mapping.items():
        print(f"  {pt_name} -> {ms_name}")
    
    return mapping

def save_mindspore_checkpoint(converted_weights, save_path):
    """
    保存MindSpore检查点文件
    
    Args:
        converted_weights: 转换后的权重列表
        save_path: 保存路径
    """
    print(f"保存MindSpore检查点到: {save_path}")
    
    # 创建参数列表
    param_list = []
    for weight_info in converted_weights:
        param = mindspore.Parameter(
            weight_info['data'], 
            name=weight_info['name']
        )
        param_list.append(param)
    
    # 保存检查点
    mindspore.save_checkpoint(param_list, save_path)
    print("MindSpore检查点保存成功!")

def verify_conversion(pytorch_model, mindspore_model, test_input_shape=(1, 1, 40, 26)):
    """
    验证转换后的模型是否正确
    
    Args:
        pytorch_model: PyTorch模型
        mindspore_model: MindSpore模型
        test_input_shape: 测试输入形状
    """
    print("开始验证模型转换...")
    
    # 生成测试输入
    test_input = np.random.randn(*test_input_shape).astype(np.float32)
    
    # PyTorch推理
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_input = torch.from_numpy(test_input)
        pytorch_output = pytorch_model(pytorch_input).numpy()
    
    # MindSpore推理
    mindspore_model.set_train(False)
    mindspore_input = mindspore.Tensor(test_input)
    mindspore_output = mindspore_model(mindspore_input).asnumpy()
    
    # 比较输出
    diff = np.abs(pytorch_output - mindspore_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"输出差异统计:")
    print(f"  最大差异: {max_diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  PyTorch输出形状: {pytorch_output.shape}")
    print(f"  MindSpore输出形状: {mindspore_output.shape}")
    
    if max_diff < 1e-4:
        print("✅ 模型转换验证成功! 输出差异在可接受范围内")
        return True
    else:
        print("⚠️ 模型转换可能存在问题，输出差异较大")
        return False

def convert_model(pytorch_model_path, mindspore_model_path):
    """
    完整的模型转换流程
    
    Args:
        pytorch_model_path: PyTorch模型文件路径
        mindspore_model_path: MindSpore模型保存路径
    """
    print("="*60)
    print("PyTorch to MindSpore 模型转换工具")
    print("="*60)
    
    try:
        # 设置MindSpore上下文
        mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="CPU")
        
        # 1. 加载PyTorch模型
        pytorch_model = load_pytorch_model(pytorch_model_path)
        
        # 2. 创建MindSpore模型
        mindspore_model = create_mindspore_model()
        
        # 3. 转换权重
        converted_weights = convert_pytorch_to_mindspore_weights(pytorch_model, mindspore_model)
        
        # 4. 加载转换后的权重到MindSpore模型
        for weight_info in converted_weights:
            param = mindspore_model.parameters_dict()[weight_info['name']]
            param.set_data(weight_info['data'])
        
        # 5. 验证转换
        if verify_conversion(pytorch_model, mindspore_model):
            # 6. 保存MindSpore模型
            save_mindspore_checkpoint(converted_weights, mindspore_model_path)
            print(f"\n✅ 模型转换完成！MindSpore模型已保存到: {mindspore_model_path}")
            return True
        else:
            print(f"\n❌ 模型转换失败，请检查模型结构是否一致")
            return False
            
    except Exception as e:
        print(f"\n❌ 模型转换过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 转换模型
    pytorch_model_path = "/workspaces/codespaces-jupyter/project/best_model.pth"
    mindspore_model_path = "/workspaces/codespaces-jupyter/project/best_model_converted.ckpt"
    
    if os.path.exists(pytorch_model_path):
        success = convert_model(pytorch_model_path, mindspore_model_path)
        if success:
            print("\n🎉 恭喜！模型转换成功完成！")
        else:
            print("\n💔 模型转换失败，请检查错误信息")
    else:
        print(f"❌ PyTorch模型文件不存在: {pytorch_model_path}")
        print("请先运行PyTorch训练脚本生成模型文件")
