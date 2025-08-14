#!/usr/bin/env python3
"""
实时FLAVR补帧快速启动脚本
"""

import os
import sys
import argparse
import subprocess

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = ['torch', 'torchvision', 'opencv-python', 'pillow', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_model_files():
    """检查模型文件是否存在"""
    model_files = {
        '2x': 'models/flavr_2x.pth',
        '4x': 'models/flavr_4x.pth', 
        '8x': 'models/flavr_8x.pth'
    }
    
    available_models = []
    for factor, path in model_files.items():
        if os.path.exists(path):
            available_models.append(factor)
            print(f"✅ {factor}x模型: {path}")
        else:
            print(f"❌ {factor}x模型: {path} (未找到)")
    
    return available_models

def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ CUDA可用: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("⚠️  CUDA不可用，将使用CPU模式")
            return False
    except:
        print("⚠️  无法检查CUDA状态")
        return False

def run_realtime_interpolation(model_factor, camera_index=0, device='cuda', simple_mode=True):
    """运行实时插值"""
    model_path = f"models/flavr_{model_factor}x.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    # 构建命令
    if simple_mode:
        script = "realtime_simple.py"
        cmd = [
            sys.executable, script,
            "--model", model_path,
            "--factor", str(model_factor),
            "--camera", str(camera_index),
            "--device", device
        ]
    else:
        script = "realtime_interpolator_advanced.py"
        cmd = [
            sys.executable, script,
            "--model", model_path,
            "--factor", str(model_factor),
            "--camera", str(camera_index),
            "--device", device,
            "--scale", "1.0"  # 默认保持原始尺寸
        ]
    
    print(f"\n🚀 启动实时补帧...")
    print(f"   脚本: {script}")
    print(f"   模型: {model_factor}x")
    print(f"   摄像头: {camera_index}")
    print(f"   设备: {device}")
    print(f"   命令: {' '.join(cmd)}")
    print("\n按 'q' 退出程序")
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 程序运行失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
        return True

def main():
    parser = argparse.ArgumentParser(description="实时FLAVR补帧快速启动")
    parser.add_argument("--factor", type=str, choices=['2', '4', '8'], default='2',
                       help="插值倍数 (默认: 2)")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引 (默认: 0)")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cuda',
                       help="计算设备 (默认: cuda)")
    parser.add_argument("--advanced", action="store_true", help="使用高级模式")
    parser.add_argument("--check-only", action="store_true", help="仅检查环境")
    
    args = parser.parse_args()
    
    print("🎬 实时FLAVR补帧系统")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        return 1
    
    # 检查CUDA
    cuda_available = check_cuda()
    if not cuda_available and args.device == 'cuda':
        print("⚠️  CUDA不可用，自动切换到CPU模式")
        args.device = 'cpu'
    
    # 检查模型文件
    available_models = check_model_files()
    if not available_models:
        print("\n❌ 没有找到任何模型文件!")
        print("请下载模型文件到 models/ 目录:")
        print("  - models/flavr_2x.pth")
        print("  - models/flavr_4x.pth") 
        print("  - models/flavr_8x.pth")
        return 1
    
    # 检查请求的模型是否可用
    if args.factor not in available_models:
        print(f"\n❌ {args.factor}x模型不可用")
        print(f"可用模型: {', '.join(available_models)}")
        return 1
    
    if args.check_only:
        print("\n✅ 环境检查完成")
        return 0
    
    # 运行实时插值
    success = run_realtime_interpolation(
        model_factor=args.factor,
        camera_index=args.camera,
        device=args.device,
        simple_mode=not args.advanced
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 