# 香橙派 + Ascend AI 部署指南

本文档详细介绍如何将压力图睡姿分类项目部署到香橙派（Orange Pi）上，使用华为Ascend AI处理器和MindSpore框架。

## 目录

1. [项目迁移概述](#项目迁移概述)
2. [硬件要求](#硬件要求)
3. [软件依赖](#软件依赖)
4. [环境搭建](#环境搭建)
5. [模型转换](#模型转换)
6. [部署步骤](#部署步骤)
7. [性能测试](#性能测试)
8. [故障排除](#故障排除)
9. [优化建议](#优化建议)

## 项目迁移概述

### 迁移内容

- **原框架**: PyTorch + CPU
- **目标框架**: MindSpore + Ascend AI
- **目标硬件**: 香橙派 + 华为Ascend AI处理器

### 文件清单

| 原始文件 | MindSpore版本 | 说明 |
|---------|---------------|------|
| `train.py` | `train_mindspore.py` | 训练脚本 |
| `dataloader/dataloader.py` | `dataloader/dataloader_mindspore.py` | 数据加载器 |
| `test_from_pth.ipynb` | `test_mindspore.ipynb` | 推理测试 |
| `web_app.py` | `web_app_mindspore.py` | Web应用 |
| `best_model.pth` | `best_model_converted.ckpt` | 模型文件 |
| - | `convert_model.py` | 模型转换工具 |

## 硬件要求

### 香橙派配置

- **推荐型号**: Orange Pi 5 Plus 或更高配置
- **内存**: 至少 8GB RAM
- **存储**: 至少 32GB microSD卡（推荐64GB以上）
- **网络**: WiFi 或有线网络连接

### Ascend AI 处理器

- **支持型号**: 
  - Ascend 310 系列
  - Ascend 910 系列（如果支持）
- **驱动版本**: 与MindSpore版本兼容
- **功耗**: 确保香橙派电源适配器能够支持

## 软件依赖

### 操作系统

```bash
# 推荐使用Ubuntu 20.04 或 22.04 ARM64版本
# 香橙派官方镜像或官方Ubuntu镜像
```

### Python环境

```bash
# Python 3.8-3.10 (推荐3.9)
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### 核心依赖

```bash
# 基础科学计算库
pip install numpy scipy matplotlib seaborn scikit-learn

# Web框架
pip install flask

# 数据处理
pip install pandas opencv-python pillow

# MindSpore (ARM64版本)
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.14/MindSpore/unified/aarch64/mindspore-2.2.14-cp39-cp39-linux_aarch64.whl

# Ascend驱动 (需要根据具体硬件版本安装)
```

## 环境搭建

### 1. 系统准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装必要工具
sudo apt install -y build-essential cmake git wget curl unzip

# 安装Python开发工具
sudo apt install -y python3-dev python3-pip python3-venv

# 创建项目目录
mkdir -p ~/pressure_map_ai
cd ~/pressure_map_ai
```

### 2. Python虚拟环境

```bash
# 创建虚拟环境
python3 -m venv mindspore_env

# 激活虚拟环境
source mindspore_env/bin/activate

# 升级pip
pip install --upgrade pip
```

### 3. 安装MindSpore

```bash
# 安装MindSpore ARM64版本
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.14/MindSpore/unified/aarch64/mindspore-2.2.14-cp39-cp39-linux_aarch64.whl

# 验证安装
python -c "import mindspore; print(mindspore.__version__)"
```

### 4. Ascend驱动安装

```bash
# 下载Ascend驱动包（根据具体硬件型号）
# 例如：Ascend 310驱动
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-aarch64.run

# 安装驱动
chmod +x Ascend-cann-toolkit_7.0.0_linux-aarch64.run
sudo ./Ascend-cann-toolkit_7.0.0_linux-aarch64.run --install

# 设置环境变量
echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.bashrc
source ~/.bashrc
```

### 5. 项目部署

```bash
# 克隆项目代码
git clone <your-project-repo>
cd pressure_map_project

# 安装项目依赖
pip install -r requirements_mindspore.txt

# 复制数据文件
# 将训练数据复制到 data/ 目录
```

## 模型转换

### 1. 准备原始PyTorch模型

```bash
# 确保有训练好的PyTorch模型文件
ls -la best_model.pth
```

### 2. 运行模型转换

```bash
# 激活虚拟环境
source mindspore_env/bin/activate

# 运行转换脚本
python convert_model.py

# 验证转换结果
ls -la best_model_converted.ckpt
```

### 3. 测试转换后的模型

```bash
# 运行测试脚本
python test_conversion.py

# 预期输出: 模型转换成功，输出差异在可接受范围内
```

## 部署步骤

### 1. 配置MindSpore上下文

```python
import mindspore
mindspore.set_context(
    mode=mindspore.PYNATIVE_MODE,  # 推理模式
    device_target="Ascend",        # 使用Ascend设备
    device_id=0                    # 设备ID
)
```

### 2. 启动训练（可选）

```bash
# 如果需要重新训练模型
python train_mindspore.py

# 监控训练过程
tail -f training.log
```

### 3. 启动推理服务

```bash
# 方式1: Jupyter Notebook
jupyter notebook test_mindspore.ipynb

# 方式2: Web应用
python web_app_mindspore.py

# 方式3: 命令行测试
python -c "
from train_mindspore import SimpleCNN
import mindspore
print('MindSpore推理测试成功')
"
```

### 4. 验证部署

```bash
# 检查服务状态
curl http://localhost:5000/api/system_info

# 预期返回JSON格式的系统信息
```

## 性能测试

### 1. 推理性能测试

```python
# 运行性能测试脚本
python performance_test.py

# 典型性能指标（参考）:
# - 推理时间: 10-50ms per sample
# - 内存使用: 2-4GB
# - Ascend利用率: 60-90%
```

### 2. 系统监控

```bash
# 监控系统资源
htop

# 监控GPU使用情况（如果支持）
nvidia-smi  # 或对应的Ascend监控工具

# 监控内存使用
free -h
```

### 3. 压力测试

```bash
# 并发请求测试
python stress_test.py --concurrent 10 --requests 100

# Web服务压力测试
ab -n 100 -c 10 http://localhost:5000/api/users
```

## 故障排除

### 常见问题

#### 1. MindSpore安装问题

```bash
# 问题: 找不到合适的MindSpore版本
# 解决: 检查Python版本和ARM架构
python --version
uname -m

# 手动安装特定版本
pip install mindspore==2.2.14 -f https://www.mindspore.cn/versions
```

#### 2. Ascend驱动问题

```bash
# 问题: Ascend设备未识别
# 解决: 检查驱动安装和设备连接
npu-smi info

# 重新安装驱动
sudo /usr/local/Ascend/driver/tools/uninstall.sh
sudo ./Ascend-cann-toolkit_xxx.run --install
```

#### 3. 模型转换问题

```bash
# 问题: 权重转换失败
# 解决: 检查模型结构一致性
python -c "
from train import SimpleCNN as PyTorchCNN
from train_mindspore import SimpleCNN as MindSporeCNN
print('PyTorch模型参数:', sum(p.numel() for p in PyTorchCNN().parameters()))
# print('MindSpore模型参数:', sum(p.size for p in MindSporeCNN().trainable_params()))
"
```

#### 4. 性能问题

```bash
# 问题: 推理速度慢
# 解决方案:
# 1. 检查设备使用情况
# 2. 优化batch size
# 3. 使用图模式
mindspore.set_context(mode=mindspore.GRAPH_MODE)
```

### 日志分析

```bash
# 查看MindSpore日志
export GLOG_v=2
export GLOG_logtostderr=1
python your_script.py

# 查看系统日志
sudo journalctl -f
```

## 优化建议

### 1. 性能优化

```python
# 使用图模式提升性能
mindspore.set_context(mode=mindspore.GRAPH_MODE)

# 启用混合精度
from mindspore.amp import auto_mixed_precision
model = auto_mixed_precision(model, amp_level="O1")

# 模型量化
from mindspore.compression import quant
quantized_model = quant.quantize_model(model)
```

### 2. 内存优化

```python
# 设置内存池
mindspore.set_context(memory_pool_block_size="30GB")

# 使用数据并行（如果有多个Ascend设备）
from mindspore.nn.parallel import DataParallel
model = DataParallel(model)
```

### 3. 部署优化

```bash
# 使用systemd管理服务
sudo tee /etc/systemd/system/pressure-map-ai.service > /dev/null <<EOF
[Unit]
Description=Pressure Map AI Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/pressure_map_ai
Environment=PATH=/home/pi/pressure_map_ai/mindspore_env/bin
ExecStart=/home/pi/pressure_map_ai/mindspore_env/bin/python web_app_mindspore.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 启用服务
sudo systemctl enable pressure-map-ai
sudo systemctl start pressure-map-ai
```

### 4. 监控和维护

```bash
# 设置日志轮转
sudo tee /etc/logrotate.d/pressure-map-ai > /dev/null <<EOF
/var/log/pressure-map-ai/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 pi pi
}
EOF

# 设置自动备份
crontab -e
# 添加: 0 2 * * * /home/pi/scripts/backup_models.sh
```

## 总结

通过以上步骤，您应该能够成功将压力图睡姿分类项目从PyTorch迁移到香橙派+MindSpore+Ascend的环境中。

### 关键优势

1. **硬件加速**: 利用Ascend AI处理器提升推理性能
2. **功耗效率**: 相比GPU方案更低的功耗
3. **部署灵活**: 支持边缘计算场景
4. **成本优化**: 香橙派提供了经济实惠的硬件平台

### 后续工作

1. 持续监控系统性能
2. 根据实际使用情况调优参数
3. 考虑模型压缩和量化
4. 开发移动端应用接口

如有问题，请参考MindSpore官方文档或提交Issue。
