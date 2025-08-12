#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLAVR Web应用启动脚本
快速启动基于Gradio的视频补帧web界面
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="启动FLAVR Web应用")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口 (默认: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址 (默认: 0.0.0.0)")
    parser.add_argument("--share", action="store_true", help="是否生成公共链接")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式")
    
    args = parser.parse_args()
    
    try:
        # 导入并启动web应用
        from web_app import create_interface
        
        print("🚀 正在启动FLAVR Web应用...")
        print(f"📡 服务器地址: http://{args.host}:{args.port}")
        print("💡 提示: 请确保已安装所有依赖包 (pip install -r requirements.txt)")
        print("📁 请准备预训练模型文件 (.pth格式)")
        
        demo = create_interface()
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug,
            inbrowser=True
        )
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所有依赖包:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 