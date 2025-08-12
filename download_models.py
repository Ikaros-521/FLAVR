#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLAVR模型下载脚本
自动下载并重命名预训练模型文件
"""

import os
import sys
import requests
from tqdm import tqdm

# 模型配置
MODEL_URLS = {
    "flavr_2x": {
        "url": "https://drive.google.com/uc?export=download&id=1IZe-39ZuXy3OheGJC-fT3shZocGYuNdH",
        "filename": "flavr_2x.pth",
        "description": "2x插值模型"
    },
    "flavr_4x": {
        "url": "https://drive.google.com/uc?export=download&id=1GARJK0Ti1gLH_O0spxAEqzbMwUKqE37S",
        "filename": "flavr_4x.pth",
        "description": "4x插值模型"
    },
    "flavr_8x": {
        "url": "https://drive.google.com/uc?export=download&id=1xoZqWJdIOjSaE2DtH4ifXKlRwFySm5Gq",
        "filename": "flavr_8x.pth",
        "description": "8x插值模型"
    }
}

def download_file(url, filename, description):
    """下载文件"""
    try:
        print(f"📥 正在下载 {description}...")
        
        # 创建models目录
        os.makedirs("models", exist_ok=True)
        
        filepath = os.path.join("models", filename)
        
        # 检查文件是否已存在
        if os.path.exists(filepath):
            print(f"✅ {description} 已存在，跳过下载")
            return True
        
        # 下载文件
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ {description} 下载完成: {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ {description} 下载失败: {e}")
        return False

def main():
    print("🚀 FLAVR模型下载工具")
    print("=" * 50)
    
    # 检查网络连接
    try:
        requests.get("https://www.google.com", timeout=5)
    except:
        print("❌ 网络连接失败，请检查网络设置")
        return
    
    print("📋 可下载的模型:")
    for i, (key, config) in enumerate(MODEL_URLS.items(), 1):
        status = "✅ 已存在" if os.path.exists(os.path.join("models", config["filename"])) else "❌ 未下载"
        print(f"  {i}. {config['description']} ({config['filename']}) - {status}")
    
    print("\n💡 注意: Google Drive下载可能需要手动确认")
    print("   如果自动下载失败，请手动下载并重命名文件")
    
    # 询问用户选择
    print("\n请选择要下载的模型:")
    print("1. 下载所有模型")
    print("2. 下载2x模型")
    print("3. 下载4x模型") 
    print("4. 下载8x模型")
    print("5. 退出")
    
    try:
        choice = input("\n请输入选择 (1-5): ").strip()
        
        if choice == "1":
            models_to_download = list(MODEL_URLS.keys())
        elif choice == "2":
            models_to_download = ["flavr_2x"]
        elif choice == "3":
            models_to_download = ["flavr_4x"]
        elif choice == "4":
            models_to_download = ["flavr_8x"]
        elif choice == "5":
            print("👋 退出下载")
            return
        else:
            print("❌ 无效选择")
            return
        
        # 下载选中的模型
        success_count = 0
        for model_key in models_to_download:
            config = MODEL_URLS[model_key]
            if download_file(config["url"], config["filename"], config["description"]):
                success_count += 1
        
        print(f"\n📊 下载完成: {success_count}/{len(models_to_download)} 个模型")
        
        if success_count > 0:
            print("\n🎉 现在可以启动Web应用了!")
            print("   运行: python start_web.py")
        
    except KeyboardInterrupt:
        print("\n\n👋 用户取消下载")
    except Exception as e:
        print(f"\n❌ 下载过程中出现错误: {e}")

if __name__ == "__main__":
    main() 