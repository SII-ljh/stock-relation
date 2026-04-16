# 沪深300成分股关联关系建模

通过8轮迭代实验，捕捉并建模沪深300成分股之间的动态相关性关系，从静态Pearson相关逐步演进到DualPath加权TopK网络。

## 核心结果

### 表1：性能排名（按综合得分）

所有结果均在 **V2 Walk-Forward** 协议下评估（无数据泄露，严格样本外测试）。

| 排名 | 方法 | NMI | ARI | Sharpe | Sortino | MaxDD | 综合得分 |
|:----:|------|:---:|:---:|:------:|:-------:|:-----:|:--------:|
| 1 | **DP_RMT_cp0.8_WK5** | **0.956** | **0.807** | 0.979 | 1.250 | 0.116 | **0.856** |
| 2 | DP_RMT_cp0.7_WK5 | 0.943 | 0.756 | 0.979 | 1.250 | 0.116 | 0.830 |
| 3 | DP_RMT_cp0.6_WK4 | 0.930 | 0.725 | 0.979 | 1.250 | 0.116 | 0.815 |
| 4 | DP_RMT_cp0.6_WK5 | 0.934 | 0.733 | 0.979 | 1.250 | 0.116 | 0.814 |
| 5 | **DP_Factor_k10_WK5** | 0.932 | 0.713 | **1.023** | **1.303** | **0.113** | 0.797 |
| 6 | Ensemble_RMT_POET15_WK5 | 0.929 | 0.693 | 0.976 | 1.242 | 0.115 | 0.797 |
| 7 | DP_Factor_cp0.6_WK5 | 0.928 | 0.693 | 1.003 | 1.279 | 0.115 | 0.773 |
| 8 | DP_POET15_cp0.6_WK5 | 0.925 | 0.666 | 0.975 | 1.246 | 0.114 | 0.769 |

### 表2：多样性展示（按专长分类）

| 类别 | 方法 | NMI | Sharpe | MaxDD | 优势 |
|------|------|:---:|:------:|:-----:|------|
| 综合最优 | DP_RMT_cp0.8_WK5 | **0.956** | 0.979 | 0.116 | 综合得分、NMI、ARI最高 |
| 金融最优 | DP_Factor_k10_WK5 | 0.932 | **1.023** | **0.113** | Sharpe/Sortino/Calmar最高 |
| 稳健之选 | DP_RMT_cp0.6_WK4 | 0.930 | 0.979 | 0.116 | IC最高(0.857)，表现稳定 |
| 风控优先 | DP_POET15_cp0.6_WK5 | 0.925 | 0.975 | **0.114** | MaxDD最低，LogLik最高 |
| 集成方法 | Ensemble_RMT_POET15_WK5 | 0.929 | 0.976 | 0.115 | RankIC最高(0.575) |
| 聚类专家 | RMT+Prior_pw0.5 | 0.911 | 0.842 | 0.179 | 单路径NMI最佳 |
| 历史最优 | DP_RMT_cp0.6_K3 | 0.899 | 0.979 | 0.116 | 二值TopK最优 |

冠军模型 **DP_RMT_cp0.8_WK5** 达到 **NMI=0.956** 且 **Sharpe=0.979**，相比此前最优(0.899) NMI提升 **+6.3%**，Sharpe无任何下降。

## 关键技术突破：加权TopK

所有迭代中影响最大的单项改进是将二值TopK邻接矩阵(0/1)替换为**加权TopK**（使用|相关系数|作为边权重）：

| 维度 | 二值TopK（旧） | 加权TopK（新） | 提升幅度 |
|------|:--------------:|:--------------:|:--------:|
| NMI（最优） | 0.899 | **0.956** | +6.3% |
| ARI（最优） | 0.637 | **0.807** | +26.7% |
| cp敏感性 | 无 | 非常显著 | — |
| 最优K值 | 3 | 5 | — |
| Sharpe | 0.979 | 0.979 | 不变 |

**原理**：加权邻接矩阵保留了相关性强度的梯度信息。谱聚类能获取两只股票之间"连接强度"的信息，而非仅仅"是否连接"。结合高先验权重(cp=0.8)，同行业边的权重远高于跨行业边，产生更清晰的社区结构。

## 架构：DualPath（双路径）

```
输入：历史收益率 [0, t)
  │
  ├─ 基础估计器（RMT / Factor / POET）
  │    → 去噪相关矩阵 + 协方差矩阵
  │
  ├─ 聚类路径：
  │    corr_cluster = (1-cp) × 去噪相关矩阵 + cp × 行业先验矩阵
  │    → 加权TopK → 谱聚类 → NMI评估
  │
  └─ 组合路径：
       cov_portfolio = 去噪协方差矩阵（不注入先验）
       → 最小方差优化 → Sharpe评估
```

核心洞察：聚类需要**放大**行业差异（注入先验），而组合优化需要**纯净**的协方差（纯去噪）。对不同任务使用不同矩阵，打破了NMI-Sharpe之间的权衡。

## 改进历程

| 阶段 | NMI | Sharpe | 关键变化 |
|------|:---:|:------:|----------|
| Exp1：静态Pearson | 0.668 | — | 基线 |
| Exp6-7：EWMA+先验 | 0.748 | 0.806 | 行业先验正则化 |
| Exp8-10：参数优化 | 0.900 | 0.806 | nc=35, pw=0.7, TopK=4 |
| Iter3：DualPath二值 | 0.899 | 0.979 | 分离聚类/组合路径 |
| **Iter7-8：DualPath加权** | **0.956** | **0.979** | **加权TopK邻接矩阵** |

## 环境配置

- Python 3.10+
- 仅需CPU（无需GPU）

### 安装依赖

```bash
conda create -n stock-relation python=3.10 -y
conda activate stock-relation
pip install -r requirements.txt
```

## 快速开始

### 1. 下载数据

```bash
python download_baostock.py
```

输出文件：
- `data/*.csv` — 个股日度数据
- `data/close_prices_valid.csv` — 合并后的收盘价（271只股票 × 1455个交易日）
- `data/returns.csv` — 日收益率矩阵
- `data/industry_info.csv` — 行业分类信息

### 2. 数据预处理

```bash
python preprocess.py
```

### 3. 运行实验

```bash
# 第一阶段：基础方法
python exp1_pearson.py    # 静态Pearson相关网络
python exp2_dynamic.py    # 动态相关（滚动窗口 + LedoitWolf + EWMA）
python exp3_granger.py    # Granger因果网络
python exp4_glasso.py     # Graphical Lasso稀疏逆协方差
python exp5_ensemble.py   # 多信号融合 + 组合验证
python exp6_adaptive.py   # 自适应动态网络 + 行业先验
python exp7_optimize.py   # 参数网格搜索 + 方法对比

# 第二阶段：参数优化
python exp8_improvements.py
python exp9_extreme.py
python exp10_validation.py

# 第三阶段：DualPath + 加权TopK（当前最优）
python round1_cp_adj.py           # cp优化 + 邻接方法探索
python round2_weighted_ensemble.py # 加权TopK全面优化
python eval_final_v3.py           # 最终综合评估（V2 Walk-Forward）
```

## 项目结构

```
stock-relation/
├── README.md, requirements.txt, run_all.sh
├── download_baostock.py, preprocess.py
├── exp1_pearson.py ~ exp7_optimize.py      # 第一阶段：基础方法
├── exp8_improvements.py ~ exp10_validation.py  # 第二阶段：参数优化
├── round1_cp_adj.py, round2_weighted_ensemble.py  # 第三阶段：DualPath+加权TopK
├── eval_framework_v2.py                    # V2 Walk-Forward评估框架
├── eval_final_v3.py                        # 最终综合评估
├── experiment.md                           # 第一二阶段实验日志（中文）
├── iteration_log.md                        # 完整迭代日志（中文）
├── data/
└── results/
    ├── exp1/ ~ exp10/                      # 第一二阶段结果
    ├── round1/, round2/                    # 第三阶段结果
    ├── eval_final/                         # V2评估（26种策略）
    └── eval_final_v3/                      # V3评估（23种策略，最终版）
```

## 已淘汰方法

以下方法已被超越，仅作参考保留：

| 方法 | NMI | Sharpe | 淘汰原因 |
|------|:---:|:------:|----------|
| SamplePearson | 0.766 | 0.969 | 无去噪，无先验 |
| LedoitWolf | 0.769 | 0.962 | 相比样本估计提升有限 |
| RMT_Denoise | 0.722 | 0.979 | Sharpe好但NMI低 |
| POET15（单独） | 0.765 | 0.975 | 被DualPath+POET15超越 |
| PCA_Factor_k20（单独） | 0.740 | 1.003 | 被DualPath+Factor超越 |
| NonlinearShrinkage | 0.771 | 0.971 | 无明显优势 |
| Adaptive_EWMA_pw0.7 | 0.899 | 0.802 | NMI与Sharpe存在权衡，已被超越 |
| RMT+Prior_pw0.3 | 0.884 | 0.817 | Sharpe过低 |
| GLasso_Partial | 0.568 | 0.710 | 大规模(271只)表现差 |
| EqualWeight | — | 0.682 | 朴素基线 |

## 评估指标

| 指标 | 说明 | 方向 |
|------|------|:----:|
| **NMI** | 归一化互信息：聚类结果与真实行业的匹配度 (0~1) | 越高越好 |
| **ARI** | 调整兰德指数：聚类一致性 | 越高越好 |
| **Modularity** | 模块度：网络社区结构质量 | 越高越好 |
| **IC** | 行业一致性：同行业边的占比 | 越高越好 |
| **CovError** | 协方差预测的相对Frobenius误差 | 越低越好 |
| **LogLik** | 样本外收益率的对数似然 | 越高越好 |
| **RankIC** | 协方差元素的Spearman秩相关系数 | 越高越好 |
| **Sharpe** | 年化风险调整收益 | 越高越好 |
| **Sortino** | 下行风险调整收益 | 越高越好 |
| **MaxDD** | 最大回撤 | 越低越好 |
| **Calmar** | 收益/最大回撤比率 | 越高越好 |
| **CompositeScore** | 加权平均（NMI 25%、Sharpe 15%、ARI 10%、...） | 越高越好 |

## 核心发现

1. **加权TopK是变革性改进**：保留相关系数值作为边权重（vs 二值0/1），NMI提升6.3%且Sharpe无损失
2. **DualPath分离原则**：聚类需要先验放大，组合需要纯净去噪——不同任务使用不同矩阵
3. **行业结构是A股最强信号**：加权邻接矩阵下先验权重cp=0.8为最优
4. **RMT去噪是最佳基础估计器**：Marchenko-Pastur阈值能干净地分离信号与噪声
5. **Factor k=10最大化Sharpe**：更少的因子 = 更强的正则化 = 更稳定的组合
6. **A股关联关系具有结构性持续性**：长期相关性主导短期噪声
7. **聚类数~35匹配有效行业结构**（少于原始42个行业分类）

## 数据来源

- 行情数据：[baostock](http://baostock.com)（免费，无需注册）
- 行业分类：baostock `query_stock_industry` 接口
- 时间范围：2020-01-01 ~ 2025-12-31
- 样本：沪深300成分股（300只，过滤后有效271只）

## 许可证

MIT
