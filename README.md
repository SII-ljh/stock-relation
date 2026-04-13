# A股股票关系建模 (Stock Relation Modeling for China A-Shares)

捕捉A股市场沪深300成分股之间的相关关系，通过7轮迭代实验从静态相关到自适应动态网络，系统性地探索最优建模方法。

## 核心成果

| 方法 | NMI(vs 行业) | 协方差误差 |
|------|:-----------:|:---------:|
| 静态 Pearson TopK=5 | 0.668 | 0.425 |
| 偏相关 (GLasso) TopK=5 | 0.674 | 0.742 |
| Granger 因果 | 0.600 | — |
| **Adaptive EWMA + 行业先验 (最优)** | **0.748** | **0.723** |

最终模型通过 EWMA 动态协方差估计 + 行业先验正则化，在行业结构发现（NMI）上达到 0.748，比纯数据驱动方法提升 12%。

## 环境要求

- Python 3.10+
- 仅需 CPU（无 GPU 依赖）

### 依赖安装

```bash
# 创建 conda 环境（推荐）
conda create -n stock-relation python=3.10 -y
conda activate stock-relation

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 下载数据

从 baostock 下载沪深300成分股 2020-2025 年日线数据（约需20分钟）：

```bash
python download_baostock.py
```

输出文件说明：
- `data/*.csv` — 每只股票的原始日线数据
- `data/close_prices_valid.csv` — 合并后的收盘价矩阵（271只 x 1455天）
- `data/returns.csv` — 日收益率矩阵
- `data/industry_info.csv` — 行业分类信息

### 2. 数据预处理

```bash
python preprocess.py
```

- 填充缺失值、裁剪极端收益率（>20%）
- 输出 `data/returns_clean.csv` 和 `data/close_prices_clean.csv`
- 输出 EDA 统计到 `results/eda_summary.json`

### 3. 运行实验

按顺序运行 7 个实验，每个约 1-10 分钟（CPU）：

```bash
# 实验1: 静态Pearson相关性网络
python exp1_pearson.py

# 实验2: 动态相关性（滚动窗口 + LedoitWolf + EWMA）
python exp2_dynamic.py

# 实验3: Granger因果网络（50只代表性股票，40核并行）
python exp3_granger.py

# 实验4: Graphical Lasso 稀疏逆协方差 + 偏相关
python exp4_glasso.py

# 实验5: 多信号融合 + 投资组合验证
python exp5_ensemble.py

# 实验6: 自适应动态网络 + 行业先验正则化
python exp6_adaptive.py

# 实验7: 参数网格搜索 + 全方法对比
python exp7_optimize.py
```

### 4. 一键运行全部

```bash
bash run_all.sh
```

## 项目结构

```
stock-relation/
├── README.md                  # 本文件
├── requirements.txt           # Python 依赖
├── run_all.sh                 # 一键运行脚本
├── experiment.md              # 详细实验记录（含所有数值结果）
│
├── download_baostock.py       # 数据下载（baostock API）
├── preprocess.py              # 数据预处理 + EDA
│
├── exp1_pearson.py            # 实验1: 静态Pearson + 阈值/TopK网络
├── exp2_dynamic.py            # 实验2: 滚动窗口/LW/EWMA动态协方差
├── exp3_granger.py            # 实验3: Granger因果网络
├── exp4_glasso.py             # 实验4: Graphical Lasso偏相关网络
├── exp5_ensemble.py           # 实验5: 多信号融合 + 下游任务验证
├── exp6_adaptive.py           # 实验6: 自适应EWMA + 行业先验
├── exp7_optimize.py           # 实验7: 参数优化 + 最终对比
│
├── data/                      # 数据目录（运行download后生成）
│   ├── *.csv                  # 个股日线数据
│   ├── close_prices_valid.csv # 合并收盘价
│   ├── returns_clean.csv      # 清洗后收益率
│   └── industry_info.csv      # 行业分类
│
└── results/                   # 实验结果
    ├── eda_summary.json       # EDA统计
    ├── exp1/ ~ exp7/          # 各实验结果
    └── stock_statistics.csv   # 股票统计特征
```

## 方法详解

### 实验1: 静态Pearson相关性
计算全样本收益率相关矩阵，比较阈值法和 TopK 稀疏化。**发现 TopK=5（每只保留5个最强关系）效果最佳**。

### 实验2: 动态协方差估计
比较滚动窗口 Pearson、Ledoit-Wolf 收缩、EWMA 指数加权。**Ledoit-Wolf 在协方差预测上最稳健**。

### 实验3: Granger因果
对50只代表性股票两两 Granger 检验（lag=5），构建有向因果网络。**因果邻居的预测 R²=0.265，远超自回归基准**。发现因果关系主要是跨行业的。

### 实验4: Graphical Lasso
通过 L1 正则化精度矩阵估计偏相关网络。偏相关去除间接关系后，**用更少的边（814 vs 982）达到更高的 NMI（0.673 vs 0.672）**。

### 实验5: 融合与下游验证
融合 Pearson + 偏相关 + 滞后互相关。**最小方差投资组合 Sharpe=0.78，显著优于等权的0.69**。

### 实验6-7: 自适应 + 行业先验（最终方案）
核心创新：将行业分类作为先验信息注入 EWMA 协方差估计。网格搜索发现最优参数：**半衰期252天 + 行业先验权重0.30，NMI=0.748**。

## 关键发现

1. **行业结构是A股关系的最强信号**：行业先验正则化使 NMI 从 0.58 提升到 0.75（+29%）
2. **A股关系变化缓慢**：最优半衰期约1年，说明关系结构具有持续性
3. **偏相关优于全相关**：去除间接关系噪声后效果更好
4. **Granger因果提供互补信息**：捕捉的是跨行业领先-滞后关系，而非同行业共变

## 评估指标说明

| 指标 | 含义 |
|------|------|
| **NMI** | 归一化互信息，衡量网络聚类与真实行业分类的吻合度（0~1，越高越好） |
| **IC** | 网络中同行业连边占比 |
| **Cov Error** | 协方差矩阵预测的相对 Frobenius 误差（越低越好） |
| **Sharpe** | 基于协方差的最小方差组合年化风险调整收益 |

## 数据来源

- 行情数据：[baostock](http://baostock.com)（免费，无需注册，支持前复权）
- 行业分类：baostock `query_stock_industry` 接口
- 时间范围：2020-01-01 ~ 2025-12-31
- 样本：沪深300成分股（最新一期，共300只，有效271只）

## License

MIT
