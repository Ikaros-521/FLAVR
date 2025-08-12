#!/bin/bash

echo "========================================"
echo "   FLAVR 环境安装脚本"
echo "========================================"
echo

# 检查Python环境
echo "🔍 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到Python3，请先安装Python 3.7+"
    echo "💡 Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "💡 CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "💡 macOS: brew install python3"
    exit 1
fi

echo "✅ Python环境正常"
echo

# 检查pip
echo "🔍 检查pip..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ 未找到pip3，请检查Python安装"
    exit 1
fi

echo "✅ pip正常"
echo

# 安装依赖
echo "📦 安装依赖包..."
echo "💡 这可能需要几分钟时间，请耐心等待..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ 依赖安装失败"
    echo "💡 请尝试手动安装: pip3 install -r requirements.txt"
    exit 1
fi

echo
echo "✅ 依赖安装完成"
echo

# 测试环境
echo "🔍 测试环境..."
python3 test_environment.py

if [ $? -ne 0 ]; then
    echo "❌ 环境测试失败"
    exit 1
fi

echo
echo "🎉 安装完成！"
echo
echo "📖 使用说明:"
echo "   1. 下载模型文件到 models/ 目录"
echo "   2. 重命名为: flavr_2x.pth, flavr_4x.pth, flavr_8x.pth"
echo "   3. 运行 python3 start_web.py 启动Web应用"
echo "   4. 在浏览器中访问 http://localhost:7860"
echo
echo "💡 详细说明请查看 models/README.md"
echo 