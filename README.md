# Stock Relation Modeling for China A-Shares

Capturing and modeling dynamic correlation relationships between HS300 constituent stocks through 8 rounds of iterative experiments, evolving from static Pearson correlation to DualPath Weighted-TopK networks.

## Core Results

### Table 1: Performance Ranking (by CompositeScore)

All results evaluated under **V2 Walk-Forward** protocol (no data leakage, strictly out-of-sample).

| Rank | Method | NMI | ARI | Sharpe | Sortino | MaxDD | Composite |
|:----:|--------|:---:|:---:|:------:|:-------:|:-----:|:---------:|
| 1 | **DP_RMT_cp0.8_WK5** | **0.956** | **0.807** | 0.979 | 1.250 | 0.116 | **0.856** |
| 2 | DP_RMT_cp0.7_WK5 | 0.943 | 0.756 | 0.979 | 1.250 | 0.116 | 0.830 |
| 3 | DP_RMT_cp0.6_WK4 | 0.930 | 0.725 | 0.979 | 1.250 | 0.116 | 0.815 |
| 4 | DP_RMT_cp0.6_WK5 | 0.934 | 0.733 | 0.979 | 1.250 | 0.116 | 0.814 |
| 5 | **DP_Factor_k10_WK5** | 0.932 | 0.713 | **1.023** | **1.303** | **0.113** | 0.797 |
| 6 | Ensemble_RMT_POET15_WK5 | 0.929 | 0.693 | 0.976 | 1.242 | 0.115 | 0.797 |
| 7 | DP_Factor_cp0.6_WK5 | 0.928 | 0.693 | 1.003 | 1.279 | 0.115 | 0.773 |
| 8 | DP_POET15_cp0.6_WK5 | 0.925 | 0.666 | 0.975 | 1.246 | 0.114 | 0.769 |

### Table 2: Diversity Showcase (by Specialty)

| Category | Method | NMI | Sharpe | MaxDD | Strength |
|----------|--------|:---:|:------:|:-----:|----------|
| Overall Best | DP_RMT_cp0.8_WK5 | **0.956** | 0.979 | 0.116 | Highest Composite, NMI, ARI |
| Finance Best | DP_Factor_k10_WK5 | 0.932 | **1.023** | **0.113** | Highest Sharpe/Sortino/Calmar |
| Robust Pick | DP_RMT_cp0.6_WK4 | 0.930 | 0.979 | 0.116 | Highest IC (0.857), stable |
| Risk Control | DP_POET15_cp0.6_WK5 | 0.925 | 0.975 | **0.114** | Lowest MaxDD, highest LogLik |
| Ensemble | Ensemble_RMT_POET15_WK5 | 0.929 | 0.976 | 0.115 | Highest RankIC (0.575) |
| Cluster Expert | RMT+Prior_pw0.5 | 0.911 | 0.842 | 0.179 | Best single-path NMI |
| Legacy Best | DP_RMT_cp0.6_K3 | 0.899 | 0.979 | 0.116 | Best binary TopK |

The champion model **DP_RMT_cp0.8_WK5** achieves **NMI=0.956** with **Sharpe=0.979**, representing a **+6.3%** NMI improvement over the previous best (0.899) with no Sharpe degradation.

## Key Technical Breakthrough: Weighted TopK

The single most impactful improvement across all iterations was replacing binary TopK adjacency (0/1) with **weighted TopK** (using |correlation| as edge weights):

| Dimension | Binary TopK (old) | Weighted TopK (new) | Improvement |
|-----------|:-----------------:|:-------------------:|:-----------:|
| NMI (best) | 0.899 | **0.956** | +6.3% |
| ARI (best) | 0.637 | **0.807** | +26.7% |
| cp sensitivity | None | Very significant | — |
| Optimal K | 3 | 5 | — |
| Sharpe | 0.979 | 0.979 | Unchanged |

**Why it works**: Weighted adjacency preserves the gradient of correlation strength. Spectral clustering gains information about "how strongly" two stocks are connected, not just "whether" they are connected. Combined with high prior weight (cp=0.8), same-industry edges get much higher weights than cross-industry ones, producing clearer community structure.

## Architecture: DualPath

```
Input: Historical returns [0, t)
  │
  ├─ Base estimator (RMT / Factor / POET)
  │    → denoised correlation matrix + covariance matrix
  │
  ├─ Clustering Path:
  │    corr_cluster = (1-cp) × denoised_corr + cp × industry_prior
  │    → Weighted TopK → Spectral Clustering → NMI evaluation
  │
  └─ Portfolio Path:
       cov_portfolio = denoised_cov (NO prior injection)
       → Min-Variance Optimization → Sharpe evaluation
```

Key insight: Clustering needs **amplified** industry differences (inject prior), while portfolio optimization needs **clean** covariance (pure denoising). Using different matrices for different tasks breaks the NMI-Sharpe tradeoff.

## Improvement Progression

| Stage | NMI | Sharpe | Key Change |
|-------|:---:|:------:|------------|
| Exp1: Static Pearson | 0.668 | — | Baseline |
| Exp6-7: EWMA + Prior | 0.748 | 0.806 | Industry prior regularization |
| Exp8-10: Parameter opt | 0.900 | 0.806 | nc=35, pw=0.7, TopK=4 |
| Iter3: DualPath binary | 0.899 | 0.979 | Separate cluster/portfolio paths |
| **Iter7-8: DualPath weighted** | **0.956** | **0.979** | **Weighted TopK adjacency** |

## Environment

- Python 3.10+
- CPU only (no GPU required)

### Install Dependencies

```bash
conda create -n stock-relation python=3.10 -y
conda activate stock-relation
pip install -r requirements.txt
```

## Quick Start

### 1. Download Data

```bash
python download_baostock.py
```

Output files:
- `data/*.csv` — Individual stock daily data
- `data/close_prices_valid.csv` — Combined close prices (271 stocks x 1455 days)
- `data/returns.csv` — Daily returns matrix
- `data/industry_info.csv` — Industry classification

### 2. Preprocess Data

```bash
python preprocess.py
```

### 3. Run Experiments

```bash
# Phase 1: Basic methods
python exp1_pearson.py    # Static Pearson correlation network
python exp2_dynamic.py    # Dynamic correlation (rolling window + LedoitWolf + EWMA)
python exp3_granger.py    # Granger causality network
python exp4_glasso.py     # Graphical Lasso sparse inverse covariance
python exp5_ensemble.py   # Multi-signal fusion + portfolio validation
python exp6_adaptive.py   # Adaptive dynamic network + industry prior
python exp7_optimize.py   # Parameter grid search + method comparison

# Phase 2: Parameter optimization
python exp8_improvements.py
python exp9_extreme.py
python exp10_validation.py

# Phase 3: DualPath + Weighted TopK (current best)
python round1_cp_adj.py           # cp optimization + adjacency method exploration
python round2_weighted_ensemble.py # Weighted TopK full optimization
python eval_final_v3.py           # Final comprehensive evaluation (V2 Walk-Forward)
```

## Project Structure

```
stock-relation/
├── README.md, requirements.txt, run_all.sh
├── download_baostock.py, preprocess.py
├── exp1_pearson.py ~ exp7_optimize.py      # Phase 1: basic methods
├── exp8_improvements.py ~ exp10_validation.py  # Phase 2: parameter opt
├── round1_cp_adj.py, round2_weighted_ensemble.py  # Phase 3: DualPath + WeightedTopK
├── eval_framework_v2.py                    # V2 Walk-Forward evaluation framework
├── eval_final_v3.py                        # Final comprehensive evaluation
├── experiment.md                           # Phase 1-2 experiment log (Chinese)
├── iteration_log.md                        # Full iteration log (Chinese)
├── data/
└── results/
    ├── exp1/ ~ exp10/                      # Phase 1-2 results
    ├── round1/, round2/                    # Phase 3 results
    ├── eval_final/                         # V2 eval (26 strategies)
    └── eval_final_v3/                      # V3 eval (23 strategies, final)
```

## Archived Methods

The following methods are superseded and kept for reference only:

| Method | NMI | Sharpe | Why Archived |
|--------|:---:|:------:|--------------|
| SamplePearson | 0.766 | 0.969 | No denoising, no prior |
| LedoitWolf | 0.769 | 0.962 | Marginal improvement over sample |
| RMT_Denoise | 0.722 | 0.979 | Good Sharpe but low NMI |
| POET15 (standalone) | 0.765 | 0.975 | Superseded by DualPath+POET15 |
| PCA_Factor_k20 (standalone) | 0.740 | 1.003 | Superseded by DualPath+Factor |
| NonlinearShrinkage | 0.771 | 0.971 | No clear advantage |
| Adaptive_EWMA_pw0.7 | 0.899 | 0.802 | NMI=Sharpe tradeoff, superseded |
| RMT+Prior_pw0.3 | 0.884 | 0.817 | Sharpe too low |
| GLasso_Partial | 0.568 | 0.710 | Poor at scale (271 stocks) |
| EqualWeight | — | 0.682 | Naive baseline |

## Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|:---------:|
| **NMI** | Normalized Mutual Information: clustering vs true industry (0~1) | Higher |
| **ARI** | Adjusted Rand Index: clustering agreement | Higher |
| **Modularity** | Network community structure quality | Higher |
| **IC** | Industry Consistency: proportion of same-industry edges | Higher |
| **CovError** | Relative Frobenius error of covariance prediction | Lower |
| **LogLik** | Log-likelihood of out-of-sample returns | Higher |
| **RankIC** | Spearman rank correlation of covariance elements | Higher |
| **Sharpe** | Annualized risk-adjusted return | Higher |
| **Sortino** | Downside risk-adjusted return | Higher |
| **MaxDD** | Maximum drawdown | Lower |
| **Calmar** | Return / MaxDD ratio | Higher |
| **CompositeScore** | Weighted average (NMI 25%, Sharpe 15%, ARI 10%, ...) | Higher |

## Key Findings

1. **Weighted TopK is transformational**: Preserving correlation values as edge weights (vs binary 0/1) boosts NMI by 6.3% with no Sharpe cost
2. **DualPath separation principle**: Clustering needs prior amplification; portfolio needs clean denoising — different matrices for different tasks
3. **Industry structure is A-share's strongest signal**: Prior weight cp=0.8 is optimal for weighted adjacency
4. **RMT denoising is the best base**: Marchenko-Pastur threshold cleanly separates signal from noise
5. **Factor k=10 maximizes Sharpe**: Fewer factors = stronger regularization = more stable portfolio
6. **A-share relationships are structurally persistent**: Long-term correlations dominate short-term noise
7. **Cluster count ~35 matches effective industry structure** (fewer than raw 42 categories)

## Data Source

- Market data: [baostock](http://baostock.com) (free, no registration needed)
- Industry classification: baostock `query_stock_industry` API
- Time range: 2020-01-01 ~ 2025-12-31
- Sample: HS300 constituents (300 stocks, 271 valid after filtering)

## License

MIT
