#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLAVR环境测试脚本
用于验证所有依赖是否正确安装
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """测试模块导入"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} - 导入成功")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - 导入失败: {e}")
        if package_name:
            print(f"   请运行: pip install {package_name}")
        return False

def test_cuda():
    """测试CUDA可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA - 可用 (版本: {torch.version.cuda})")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            print(f"   当前GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  CUDA - 不可用 (将使用CPU模式)")
            return False
    except Exception as e:
        print(f"❌ CUDA测试失败: {e}")
        return False

def test_favr_modules():
    """测试FLAVR相关模块"""
    modules = [
        ("model.FLAVR_arch", "FLAVR模型架构"),
        ("dataset.transforms", "数据变换"),
        ("loss", "损失函数"),
        ("myutils", "工具函数")
    ]
    
    success = True
    for module, desc in modules:
        if not test_import(module):
            print(f"   {desc}模块缺失")
            success = False
    
    return success

def main():
    print("🔍 FLAVR环境测试开始...\n")
    
    # 测试基础依赖
    print("📦 测试基础依赖:")
    basic_deps = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
        ("gradio", "gradio"),
        ("matplotlib", "matplotlib"),
        ("av", "av")
    ]
    
    basic_success = True
    for module, package in basic_deps:
        if not test_import(module, package):
            basic_success = False
    
    print()
    
    # 测试CUDA
    print("🚀 测试CUDA:")
    cuda_available = test_cuda()
    
    print()
    
    # 测试FLAVR模块
    print("🎬 测试FLAVR模块:")
    favr_success = test_favr_modules()
    
    print()
    
    # 总结
    print("📊 测试总结:")
    if basic_success and favr_success:
        print("✅ 所有依赖安装正确!")
        if cuda_available:
            print("🚀 可以使用GPU加速")
        else:
            print("⚠️  将使用CPU模式 (处理速度较慢)")
        print("\n🎉 环境配置完成，可以启动Web应用:")
        print("   python start_web.py")
    else:
        print("❌ 存在依赖问题，请检查安装:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main() 