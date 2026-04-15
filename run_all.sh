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
echo "[1/12] 下载沪深300成分股数据..."
if [ -f "data/close_prices_valid.csv" ]; then
    echo "  数据已存在，跳过下载"
else
    python download_baostock.py
fi

# Step 2: 预处理
echo "[2/12] 数据预处理..."
python preprocess.py

# Step 3-12: 实验
echo "[3/12] 实验1: 静态Pearson相关性..."
python exp1_pearson.py

echo "[4/12] 实验2: 动态相关性模型..."
python exp2_dynamic.py

echo "[5/12] 实验3: Granger因果网络..."
python exp3_granger.py

echo "[6/12] 实验4: Graphical Lasso..."
python exp4_glasso.py

echo "[7/12] 实验5: 融合模型 + 下游验证..."
python exp5_ensemble.py

echo "[8/12] 实验6: 自适应动态网络..."
python exp6_adaptive.py

echo "[9/12] 实验7: 参数优化..."
python exp7_optimize.py

echo "[10/12] 实验8: 聚类数与先验权重扩展..."
python exp8_improvements.py

echo "[11/12] 实验9: 极端参数探索..."
python exp9_extreme.py

echo "[12/12] 实验10: 精细调参与最终验证..."
python exp10_validation.py

echo ""
echo "=== 全部完成! ==="
echo "查看结果: cat experiment.md"
echo "查看最终总结: cat exp11_final_summary.md"
echo "查看数值结果: ls results/"
