#!/bin/bash
# 一键运行全部实验
# 用法: bash run_all.sh
set -e

echo "=== A股股票关系建模 ==="
echo ""

# 检查conda环境
if ! command -v python &> /dev/null; then
    echo "请先激活conda环境: conda activate stock-relation"
    exit 1
fi

# Step 1: 下载数据
echo "[1/9] 下载沪深300成分股数据..."
if [ -f "data/close_prices_valid.csv" ]; then
    echo "  数据已存在，跳过下载"
else
    python download_baostock.py
fi

# Step 2: 预处理
echo "[2/9] 数据预处理..."
python preprocess.py

# Step 3-9: 实验
echo "[3/9] 实验1: 静态Pearson相关性..."
python exp1_pearson.py

echo "[4/9] 实验2: 动态相关性模型..."
python exp2_dynamic.py

echo "[5/9] 实验3: Granger因果网络..."
python exp3_granger.py

echo "[6/9] 实验4: Graphical Lasso..."
python exp4_glasso.py

echo "[7/9] 实验5: 融合模型 + 下游验证..."
python exp5_ensemble.py

echo "[8/9] 实验6: 自适应动态网络..."
python exp6_adaptive.py

echo "[9/9] 实验7: 参数优化..."
python exp7_optimize.py

echo ""
echo "=== 全部完成! ==="
echo "查看结果: cat experiment.md"
echo "查看数值结果: ls results/"
