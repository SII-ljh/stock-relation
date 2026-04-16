# 沪深300成分股关联关系建模

从沪深300成分股的历史收益率中，自动发现股票之间的真实关联关系，并构建最优投资组合。

**核心成果**: 聚类与真实行业匹配度 NMI=0.956，最小方差组合 Sharpe=0.979。

## 快速开始（3步上手）

```bash
# 1. 安装
git clone https://github.com/SII-ljh/stock-relation.git
cd stock-relation
conda create -n stock-relation python=3.10 -y
conda activate stock-relation
pip install -r requirements.txt

# 2. 运行（自动下载数据 + 运行模型 + 输出结果）
python examples/quick_start.py

# 3. 查看实际应用场景
python examples/practical_usage.py
```

## 这个项目有什么用？

| 应用场景 | 说明 | 示例 |
|----------|------|------|
| **行业轮动** | 跟踪聚类变化，发现行业板块动态 | 某只股票脱离原行业聚类 → 可能存在基本面变化 |
| **风险分散** | 从不同聚类中选股，避免集中风险 | 组合中每个聚类选1-2只 → 真正的分散化 |
| **配对交易** | 同聚类内股票适合配对交易策略 | 同聚类的两只银行股价差回归 |
| **组合构建** | 使用模型输出的最优权重构建低风险组合 | 最小方差组合 Sharpe=0.979，优于等权 |

## 使用示例

```python
from src.data import load_data, build_industry_prior
from src.estimators import rmt_denoise
from src.network import weighted_topk_adj
from src.dualpath import make_dual_path_estimator
from src.portfolio import min_var_weights
from src.evaluation import spectral_cluster

# 加载数据
returns_df, stocks, code_to_industry = load_data('data')
industry_prior = build_industry_prior(stocks, code_to_industry)

# 创建模型（最优配置）
estimator = make_dual_path_estimator(
    base_estimator=rmt_denoise,   # RMT去噪
    industry_prior=industry_prior,
    cp=0.8,                        # 聚类路径先验权重
)

# 运行模型
corr, cov = estimator(returns_df.values)

# 获取聚类结果（哪些股票属于同一群组）
adj = weighted_topk_adj(corr, k=5)
labels = spectral_cluster(adj, n_clusters=35)

# 获取最优组合权重
weights = min_var_weights(cov, max_weight=0.05)
```

## 技术原理

```
输入: 沪深300成分股历史日收益率
  │
  ├─ RMT去噪: 用随机矩阵理论去除相关矩阵中的噪声
  │
  ├─ DualPath 双路径框架:
  │    │
  │    ├─ 聚类路径: 融合行业先验 → 加权TopK网络 → 谱聚类
  │    │   输出: 股票聚类标签（NMI=0.956）
  │    │
  │    └─ 组合路径: 纯去噪协方差 → 最小方差优化
  │        输出: 最优组合权重（Sharpe=0.979）
  │
  └─ 核心创新: 加权TopK
       保留相关系数值作为边权重（而非二值0/1）
       NMI 提升 6.3%，Sharpe 无损失
```

关键洞察: **聚类需要放大行业差异（注入先验），组合优化需要纯净协方差（纯去噪）**。对不同任务使用不同矩阵，打破了 NMI-Sharpe 之间的权衡。

## 核心结果

### 性能排名（Top 8）

| 排名 | 方法 | NMI | Sharpe | 综合得分 |
|:----:|------|:---:|:------:|:--------:|
| 1 | **DP_RMT_cp0.8_WK5** | **0.956** | 0.979 | **0.856** |
| 2 | DP_RMT_cp0.7_WK5 | 0.943 | 0.979 | 0.830 |
| 3 | DP_RMT_cp0.6_WK4 | 0.930 | 0.979 | 0.815 |
| 4 | DP_RMT_cp0.6_WK5 | 0.934 | 0.979 | 0.814 |
| 5 | **DP_Factor_k10_WK5** | 0.932 | **1.023** | 0.797 |
| 6 | Ensemble_RMT_POET15_WK5 | 0.929 | 0.976 | 0.797 |
| 7 | DP_Factor_cp0.6_WK5 | 0.928 | 1.003 | 0.773 |
| 8 | DP_POET15_cp0.6_WK5 | 0.925 | 0.975 | 0.769 |

所有结果均在 V2 Walk-Forward 协议下评估（无数据泄露，严格样本外测试）。

### 改进历程

| 阶段 | NMI | Sharpe | 关键变化 |
|------|:---:|:------:|----------|
| 静态Pearson | 0.668 | — | 基线 |
| EWMA+先验 | 0.748 | 0.806 | 行业先验正则化 |
| 参数优化 | 0.900 | 0.806 | nc=35, pw=0.7, TopK=4 |
| DualPath二值 | 0.899 | 0.979 | 分离聚类/组合路径 |
| **DualPath加权** | **0.956** | **0.979** | **加权TopK (核心突破)** |

## 项目结构

```
stock-relation/
│
├── src/                         # 核心库（所有可复用代码）
│   ├── data.py                  #   数据下载、加载、预处理
│   ├── estimators.py            #   协方差估计器（RMT、Factor、POET等）
│   ├── network.py               #   网络构建（TopK、加权TopK）
│   ├── dualpath.py              #   DualPath双路径框架
│   ├── portfolio.py             #   组合优化（最小方差）
│   ├── evaluation.py            #   Walk-Forward评估框架
│   └── utils.py                 #   工具函数
│
├── examples/                    # 示例脚本（从这里开始）
│   ├── quick_start.py           #   快速上手: 下载→建模→输出
│   └── practical_usage.py       #   实际应用: 选股/组合/对比
│
├── experiments/                 # 研究实验（历史记录）
│   ├── phase1/                  #   基础方法探索（exp1-exp7）
│   ├── phase2/                  #   参数优化（exp8-exp10）
│   └── phase3/                  #   DualPath+加权TopK（最终阶段）
│
├── docs/                        # 文档
│   ├── experiment_log.md        #   实验日志（第一二阶段）
│   └── iteration_log.md         #   迭代日志（第三阶段）
│
├── archive/                     # 已淘汰代码
├── README.md
├── requirements.txt
├── run_all.sh                   # 一键运行
└── .gitignore
```

## 模块说明

| 模块 | 功能 | 关键函数 |
|------|------|----------|
| `src.data` | 数据管理 | `download_data()`, `load_data()`, `build_industry_prior()` |
| `src.estimators` | 协方差估计 | `rmt_denoise()`, `pca_factor()`, `poet()`, `ledoit_wolf()` |
| `src.network` | 网络构建 | `weighted_topk_adj()`, `topk_adj()` |
| `src.dualpath` | 双路径框架 | `make_dual_path_estimator()`, `make_ensemble_estimator()` |
| `src.portfolio` | 组合优化 | `min_var_weights()`, `eval_portfolio_metrics()` |
| `src.evaluation` | 评估框架 | `FlexibleEvaluator`, `eval_nmi()`, `compute_composite_score()` |

## 评估指标

| 指标 | 说明 | 方向 |
|------|------|:----:|
| **NMI** | 聚类与真实行业的匹配度 (0~1) | 越高越好 |
| **ARI** | 调整兰德指数 | 越高越好 |
| **Sharpe** | 年化风险调整收益 | 越高越好 |
| **Sortino** | 下行风险调整收益 | 越高越好 |
| **MaxDD** | 最大回撤 | 越低越好 |
| **CompositeScore** | 加权综合得分 | 越高越好 |

## 数据来源

- 行情数据: [baostock](http://baostock.com)（免费，无需注册）
- 行业分类: baostock `query_stock_industry` 接口
- 时间范围: 2020-01-01 ~ 2025-12-31
- 样本: 沪深300成分股（300只，过滤后有效271只）

## 环境要求

- Python 3.10+
- 仅需 CPU（无需 GPU）
- 依赖: numpy, pandas, scipy, scikit-learn, networkx, baostock

## 许可证

MIT
