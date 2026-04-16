"""
迭代1: 在新的综合评估体系下重新评估所有已有模型
包含以下方法:
  1. Static Pearson TopK=4
  2. Static Pearson TopK=5
  3. LedoitWolf 滚动窗口
  4. EWMA (hl=126)
  5. GLasso 偏相关
  6. Adaptive EWMA (pw=0.1, hl=126) -- exp6
  7. Adaptive EWMA (pw=0.3, hl=252) -- exp7
  8. Adaptive EWMA (pw=0.5, hl=252, nc=42) -- exp8
  9. Adaptive EWMA (pw=1.0, hl=252, nc=42) -- exp9
  10. Adaptive EWMA (pw=0.7, nc=35, TopK=4) -- exp10 最佳
  11. 纯行业先验 -- baseline
  12. 等权基准
"""
import sys
sys.path.insert(0, '/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation')

import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf, GraphicalLassoCV
from eval_framework import (
    ComprehensiveEvaluator, build_industry_prior, topk_adj,
    format_results_table, compute_composite_score, eval_portfolio_metrics,
    min_var_weights
)
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/iter1"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("迭代1: 综合评估体系下重新评估所有已有模型")
print("=" * 70)

# 加载数据
returns_df = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns_df.columns.tolist()
n_stocks = len(stocks)
ret_vals = returns_df.values

print(f"股票数: {n_stocks}, 交易日: {len(ret_vals)}")

# 构建先验
industry_prior = build_industry_prior(stocks, code_to_industry)

# 初始化评估器
evaluator = ComprehensiveEvaluator(
    returns=ret_vals, stocks=stocks,
    code_to_industry=code_to_industry,
    industry_prior=industry_prior,
    warmup=250, update_freq=20, eval_freq=60, forecast=60,
    rebalance=60, n_clusters=35, topk=4
)

all_results = []


# ============================================================
# AdaptiveCovEstimator (沿用原项目代码)
# ============================================================
class AdaptiveCovEstimator:
    def __init__(self, n, half_life=63, prior_weight=0.1, prior_matrix=None):
        self.n = n
        self.decay = np.log(2) / half_life
        self.prior_weight = prior_weight
        self.prior_matrix = prior_matrix
        self.cov_ewma = None
        self.count = 0
        
    def update(self, batch):
        batch_cov = np.cov(batch.T)
        if self.cov_ewma is None:
            self.cov_ewma = batch_cov
        else:
            alpha = 1 - np.exp(-self.decay * len(batch))
            self.cov_ewma = (1 - alpha) * self.cov_ewma + alpha * batch_cov
        self.count += len(batch)
        
    def get_cov(self):
        if self.cov_ewma is None:
            return np.eye(self.n) * 0.001
        cov = self.cov_ewma.copy()
        shrinkage = min(0.8, max(0.0, self.n / max(self.count, 1)))
        target = np.diag(np.diag(cov))
        cov = (1 - shrinkage) * cov + shrinkage * target
        if self.prior_matrix is not None and self.prior_weight > 0:
            avg_var = np.diag(cov).mean()
            prior_cov = self.prior_matrix * avg_var * 0.5
            np.fill_diagonal(prior_cov, 0)
            cov = (1 - self.prior_weight) * cov + self.prior_weight * (cov + prior_cov)
        cov = (cov + cov.T) / 2
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 1e-6:
            cov += np.eye(self.n) * (1e-6 - eigvals.min())
        return cov
    
    def get_corr(self):
        cov = self.get_cov()
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-10
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 1)
        return np.clip(corr, -1, 1)


# ============================================================
# 1. Static Pearson TopK=4
# ============================================================
print("\n[1/12] Static Pearson TopK=4 ...")
corr_full = np.corrcoef(ret_vals.T)
cov_full = np.cov(ret_vals.T)
r1 = evaluator.evaluate_static_method("StaticPearson_K4", corr_full, cov_full)
all_results.append(r1)
print(f"  NMI={r1['NMI']:.4f}, Sharpe={r1['Sharpe']:.4f}")

# ============================================================
# 2. Static Pearson TopK=5
# ============================================================
print("\n[2/12] Static Pearson TopK=5 ...")
evaluator_k5 = ComprehensiveEvaluator(
    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
    industry_prior=industry_prior, topk=5, n_clusters=35
)
r2 = evaluator_k5.evaluate_static_method("StaticPearson_K5", corr_full, cov_full)
all_results.append(r2)
print(f"  NMI={r2['NMI']:.4f}, Sharpe={r2['Sharpe']:.4f}")

# ============================================================
# 3. LedoitWolf 滚动窗口 (动态)
# ============================================================
print("\n[3/12] LedoitWolf Rolling w=250 ...")

class LWEstimator:
    def __init__(self, n):
        self.n = n
        self.data = []
        self.window = 250
    def update(self, batch):
        self.data.extend(batch.tolist())
        if len(self.data) > self.window * 2:
            self.data = self.data[-self.window:]
    def get_cov(self):
        d = np.array(self.data[-self.window:])
        if len(d) < 20:
            return np.eye(self.n) * 0.001
        lw = LedoitWolf()
        lw.fit(d)
        return lw.covariance_
    def get_corr(self):
        cov = self.get_cov()
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-10
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 1)
        return np.clip(corr, -1, 1)

r3 = evaluator.evaluate_dynamic_method(
    "LedoitWolf_w250",
    lambda: LWEstimator(n_stocks)
)
all_results.append(r3)
print(f"  NMI={r3['NMI']:.4f}, CovErr={r3['CovError']:.4f}, Sharpe={r3['Sharpe']:.4f}")

# ============================================================
# 4. EWMA hl=126 (无先验)
# ============================================================
print("\n[4/12] EWMA hl=126 (no prior) ...")
r4 = evaluator.evaluate_dynamic_method(
    "EWMA_hl126",
    lambda: AdaptiveCovEstimator(n_stocks, half_life=126, prior_weight=0.0, prior_matrix=None)
)
all_results.append(r4)
print(f"  NMI={r4['NMI']:.4f}, CovErr={r4['CovError']:.4f}, Sharpe={r4['Sharpe']:.4f}")

# ============================================================
# 5. GLasso 偏相关 (静态)
# ============================================================
print("\n[5/12] GLasso Partial Correlation ...")
try:
    # 使用GraphicalLassoCV
    lasso = GraphicalLassoCV(cv=3, max_iter=200)
    lasso.fit(ret_vals)
    precision = lasso.precision_
    # 偏相关
    d = np.sqrt(np.diag(precision))
    d[d == 0] = 1e-10
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1)
    r5 = evaluator.evaluate_static_method("GLasso_Partial", partial_corr, lasso.covariance_)
except Exception as e:
    print(f"  GLasso failed: {e}, using fallback")
    r5 = evaluator.evaluate_static_method("GLasso_Partial", corr_full, cov_full)
all_results.append(r5)
print(f"  NMI={r5['NMI']:.4f}, Sharpe={r5['Sharpe']:.4f}")

# ============================================================
# 6. Adaptive EWMA pw=0.1 hl=126 (exp6)
# ============================================================
print("\n[6/12] Adaptive EWMA pw=0.1 hl=126 ...")
r6 = evaluator.evaluate_dynamic_method(
    "Adaptive_hl126_pw0.1",
    lambda: AdaptiveCovEstimator(n_stocks, half_life=126, prior_weight=0.1, prior_matrix=industry_prior)
)
all_results.append(r6)
print(f"  NMI={r6['NMI']:.4f}, CovErr={r6['CovError']:.4f}, Sharpe={r6['Sharpe']:.4f}")

# ============================================================
# 7. Adaptive EWMA pw=0.3 hl=252 (exp7)
# ============================================================
print("\n[7/12] Adaptive EWMA pw=0.3 hl=252 ...")
r7 = evaluator.evaluate_dynamic_method(
    "Adaptive_hl252_pw0.3",
    lambda: AdaptiveCovEstimator(n_stocks, half_life=252, prior_weight=0.3, prior_matrix=industry_prior)
)
all_results.append(r7)
print(f"  NMI={r7['NMI']:.4f}, CovErr={r7['CovError']:.4f}, Sharpe={r7['Sharpe']:.4f}")

# ============================================================
# 8. Adaptive EWMA pw=0.5 hl=252 nc=42 (exp8)
# ============================================================
print("\n[8/12] Adaptive EWMA pw=0.5 hl=252 nc=42 ...")
evaluator_nc42 = ComprehensiveEvaluator(
    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
    industry_prior=industry_prior, n_clusters=42, topk=4
)
r8 = evaluator_nc42.evaluate_dynamic_method(
    "Adaptive_hl252_pw0.5_nc42",
    lambda: AdaptiveCovEstimator(n_stocks, half_life=252, prior_weight=0.5, prior_matrix=industry_prior)
)
all_results.append(r8)
print(f"  NMI={r8['NMI']:.4f}, CovErr={r8['CovError']:.4f}, Sharpe={r8['Sharpe']:.4f}")

# ============================================================
# 9. Adaptive EWMA pw=1.0 hl=252 nc=42 (exp9)
# ============================================================
print("\n[9/12] Adaptive EWMA pw=1.0 hl=252 nc=42 ...")
r9 = evaluator_nc42.evaluate_dynamic_method(
    "Adaptive_hl252_pw1.0_nc42",
    lambda: AdaptiveCovEstimator(n_stocks, half_life=252, prior_weight=1.0, prior_matrix=industry_prior)
)
all_results.append(r9)
print(f"  NMI={r9['NMI']:.4f}, CovErr={r9['CovError']:.4f}, Sharpe={r9['Sharpe']:.4f}")

# ============================================================
# 10. Adaptive EWMA pw=0.7 nc=35 TopK=4 (exp10 最佳)
# ============================================================
print("\n[10/12] Adaptive EWMA pw=0.7 nc=35 TopK=4 (Best) ...")
r10 = evaluator.evaluate_dynamic_method(
    "Adaptive_hl252_pw0.7_nc35_K4",
    lambda: AdaptiveCovEstimator(n_stocks, half_life=252, prior_weight=0.7, prior_matrix=industry_prior)
)
all_results.append(r10)
print(f"  NMI={r10['NMI']:.4f}, CovErr={r10['CovError']:.4f}, Sharpe={r10['Sharpe']:.4f}")

# ============================================================
# 11. 纯行业先验
# ============================================================
print("\n[11/12] Pure Industry Prior ...")
corr_prior = industry_prior.copy()
np.fill_diagonal(corr_prior, 1)
# 作为协方差: 对角线用样本方差, 同行业给50%协方差
sample_var = np.diag(cov_full)
cov_prior = industry_prior * np.mean(sample_var) * 0.5
np.fill_diagonal(cov_prior, sample_var)
r11 = evaluator.evaluate_static_method("PureIndustryPrior", corr_prior, cov_prior)
all_results.append(r11)
print(f"  NMI={r11['NMI']:.4f}, Sharpe={r11['Sharpe']:.4f}")

# ============================================================
# 12. 等权基准
# ============================================================
print("\n[12/12] Equal Weight Benchmark ...")
# 等权: 用单位协方差 (相当于等权)
eq_rets = ret_vals[250:].mean(axis=1)
pm_eq = eval_portfolio_metrics(eq_rets.tolist())
r12 = {
    'method': 'EqualWeight',
    'NMI': np.nan, 'ARI': np.nan, 'Modularity': np.nan, 'IC': np.nan,
    'CovError': np.nan, 'LogLik': np.nan, 'RankIC': np.nan,
    'Sharpe': pm_eq['sharpe'], 'Sortino': pm_eq['sortino'],
    'MaxDD': pm_eq['max_drawdown'], 'Calmar': pm_eq['calmar'],
    'NMI_Std': np.nan, 'CovErr_Std': np.nan,
}
all_results.append(r12)
print(f"  Sharpe={r12['Sharpe']:.4f}, MaxDD={r12['MaxDD']:.4f}")


# ============================================================
# 汇总并排名
# ============================================================
print("\n" + "=" * 70)
print("综合评估结果")
print("=" * 70)

df = format_results_table(all_results, sort_by='NMI', ascending=False)
df['CompositeScore'] = compute_composite_score(df)
df = df.sort_values('CompositeScore', ascending=False).reset_index(drop=True)
df.index = df.index + 1  # 排名从1开始
df.index.name = 'Rank'

# 打印
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)
print(df.to_string())

# 保存
df.to_csv(f'{OUT_DIR}/comprehensive_eval.csv')
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

print(f"\n结果已保存到 {OUT_DIR}/")
print("\n迭代1完成!")
