#!/bin/bash
# stock-relation 一键运行脚本
# 用法: bash run_all.sh
set -e

echo "=== 沪深300成分股关联关系建模 ==="
echo ""

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "错误: 找不到 python, 请先激活 conda 环境"
    echo "  conda activate stock-relation"
    exit 1
fi

# 检查依赖
python -c "import numpy, pandas, sklearn, scipy, networkx" 2>/dev/null || {
    echo "缺少依赖, 正在安装..."
    pip install -r requirements.txt
}

# Step 1: 下载数据
echo "[1/3] 下载沪深300成分股数据..."
if [ -f "data/returns_clean.csv" ]; then
    echo "  数据已存在, 跳过下载"
else
    python -c "
import sys; sys.path.insert(0, '.')
from src.data import download_data
download_data()
"
fi

# Step 2: 快速上手示例
echo ""
echo "[2/3] 运行快速上手示例..."
python examples/quick_start.py

# Step 3: 实际应用示例
echo ""
echo "[3/3] 运行实际应用示例..."
python examples/practical_usage.py

echo ""
echo "=== 全部完成! ==="
echo ""
echo "查看更多:"
echo "  examples/quick_start.py    — 完整流程演示"
echo "  examples/practical_usage.py — 实际应用场景"
echo "  docs/experiment_log.md     — 实验日志"
echo "  docs/iteration_log.md      — 迭代日志"
