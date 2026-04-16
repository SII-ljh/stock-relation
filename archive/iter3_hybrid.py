"""
迭代3: 混合模型与精调
基于迭代2发现:
  - RMT+Prior_pw0.3 综合最优 (NMI=0.89, Sharpe=0.77, RankIC=0.87)
  - pw=0.3附近需精调
  - POET估计器 (Factor+Sparse residual)
  - 双路径: 高先验做聚类, 低先验做投资
  - RMT+Factor融合

新方法:
  1. RMT+Prior 精调 (pw=0.15~0.40)
  2. POET估计器 (Principal Orthogonal complement Thresholding)
  3. RMT+Factor融合 (先去噪再做因子分解)
  4. 双路径架构 (DualPath)
  5. Adaptive RMT+Prior (动态版RMT+低先验)
  6. POET + Industry Prior
"""
import sys
sys.path.insert(0, '/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation')

import pandas as pd
import numpy as np
from eval_framework import (
    ComprehensiveEvaluator, build_industry_prior, topk_adj,
    format_results_table, compute_composite_score, eval_nmi,
    eval_portfolio_metrics, min_var_weights
)
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/iter3"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("迭代3: 混合模型与精调")
print("=" * 70)

returns_df = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns_df.columns.tolist()
n_stocks = len(stocks)
ret_vals = returns_df.values
industry_prior = build_industry_prior(stocks, code_to_industry)

print(f"股票数: {n_stocks}, 交易日: {len(ret_vals)}")

evaluator = ComprehensiveEvaluator(
    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
    industry_prior=industry_prior, n_clusters=35, topk=4
)

all_results = []


def cov_to_corr(cov):
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1e-10
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1)
    return np.clip(corr, -1, 1)

def ensure_psd(cov, eps=1e-6):
    cov = (cov + cov.T) / 2
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < eps:
        cov += np.eye(cov.shape[0]) * (eps - eigvals.min())
    return cov

def rmt_denoise(returns, n_factors=None):
    T, N = returns.shape
    q = N / T
    cov_sample = np.cov(returns.T)
    std = np.sqrt(np.diag(cov_sample))
    std[std == 0] = 1e-10
    corr = cov_sample / np.outer(std, std)
    np.fill_diagonal(corr, 1)
    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    lambda_plus = (1 + np.sqrt(q)) ** 2
    if n_factors is None:
        n_signal = max(np.sum(eigvals > lambda_plus), 1)
    else:
        n_signal = n_factors
    noise_eigvals = eigvals[n_signal:]
    noise_mean = np.mean(noise_eigvals) if len(noise_eigvals) > 0 else 1.0
    denoised_eigvals = eigvals.copy()
    denoised_eigvals[n_signal:] = noise_mean
    corr_denoised = eigvecs @ np.diag(denoised_eigvals) @ eigvecs.T
    np.fill_diagonal(corr_denoised, 1)
    cov_denoised = corr_denoised * np.outer(std, std)
    cov_denoised = ensure_psd(cov_denoised)
    return corr_denoised, cov_denoised, n_signal


# ============================================================
# 1. RMT+Prior 精调 (pw=0.15~0.40)
# ============================================================
print("\n[1/6] RMT+Prior Fine-tuning ...")

corr_rmt, cov_rmt, n_sig = rmt_denoise(ret_vals)

def rmt_with_prior(cov_rmt, prior, prior_weight):
    avg_var = np.diag(cov_rmt).mean()
    prior_cov = prior * avg_var * 0.5
    np.fill_diagonal(prior_cov, np.diag(cov_rmt))
    cov_combined = (1 - prior_weight) * cov_rmt + prior_weight * prior_cov
    cov_combined = ensure_psd(cov_combined)
    corr_combined = cov_to_corr(cov_combined)
    return corr_combined, cov_combined

fine_pws = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
for pw in fine_pws:
    corr_rp, cov_rp = rmt_with_prior(cov_rmt, industry_prior, pw)
    r = evaluator.evaluate_static_method(f"RMT+Prior_pw{pw:.2f}", corr_rp, cov_rp)
    all_results.append(r)
    print(f"  pw={pw:.2f}: NMI={r['NMI']:.4f}, Sharpe={r['Sharpe']:.4f}, RankIC={r['RankIC']:.4f}, Composite will be computed later")


# ============================================================
# 2. POET估计器 (Factor + Sparse Residual)
# ============================================================
print("\n[2/6] POET Estimator ...")

def poet_estimator(returns, n_factors=10, threshold_const=0.5):
    """
    POET: Principal Orthogonal complement Thresholding
    Sigma = B * Lambda * B' + Sigma_u
    其中Sigma_u做soft thresholding
    """
    T, N = returns.shape
    cov_sample = np.cov(returns.T)
    
    eigvals, eigvecs = np.linalg.eigh(cov_sample)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 因子部分
    B = eigvecs[:, :n_factors]
    factor_var = eigvals[:n_factors]
    cov_factor = B @ np.diag(factor_var) @ B.T
    
    # 残差
    residual = cov_sample - cov_factor
    
    # Soft thresholding on residual (off-diagonal only)
    threshold = threshold_const * np.sqrt(np.log(N) / T)
    residual_thresh = residual.copy()
    for i in range(N):
        for j in range(N):
            if i != j:
                val = residual_thresh[i, j]
                residual_thresh[i, j] = np.sign(val) * max(abs(val) - threshold * np.sqrt(residual[i,i] * residual[j,j]), 0)
    
    cov_poet = cov_factor + residual_thresh
    cov_poet = ensure_psd(cov_poet)
    corr_poet = cov_to_corr(cov_poet)
    
    return corr_poet, cov_poet

# 尝试不同参数
for nf, tc in [(10, 0.5), (15, 0.5), (20, 0.3), (20, 0.5)]:
    corr_p, cov_p = poet_estimator(ret_vals, n_factors=nf, threshold_const=tc)
    adj = topk_adj(corr_p, k=4)
    np.fill_diagonal(adj, 0)
    nmi = eval_nmi(adj, stocks, code_to_industry, 35)
    print(f"  POET nf={nf} tc={tc}: NMI={nmi:.4f}")

# Best
corr_poet, cov_poet = poet_estimator(ret_vals, n_factors=20, threshold_const=0.5)
r2 = evaluator.evaluate_static_method("POET_k20_tc0.5", corr_poet, cov_poet)
all_results.append(r2)
print(f"  POET best: NMI={r2['NMI']:.4f}, CovErr={r2['CovError']:.4f}, Sharpe={r2['Sharpe']:.4f}")

# POET + prior
corr_poet15, cov_poet15 = poet_estimator(ret_vals, n_factors=15, threshold_const=0.5)
for pw in [0.2, 0.3]:
    avg_var = np.diag(cov_poet15).mean()
    prior_cov = industry_prior * avg_var * 0.5
    np.fill_diagonal(prior_cov, np.diag(cov_poet15))
    cov_pp = (1 - pw) * cov_poet15 + pw * prior_cov
    cov_pp = ensure_psd(cov_pp)
    corr_pp = cov_to_corr(cov_pp)
    r = evaluator.evaluate_static_method(f"POET+Prior_k15_pw{pw}", corr_pp, cov_pp)
    all_results.append(r)
    print(f"  POET+Prior k=15 pw={pw}: NMI={r['NMI']:.4f}, Sharpe={r['Sharpe']:.4f}")


# ============================================================
# 3. RMT+Factor融合
# ============================================================
print("\n[3/6] RMT + Factor Fusion ...")

def rmt_factor_fusion(returns, n_factors=20, rmt_weight=0.5):
    """先分别做RMT和Factor, 再加权融合"""
    corr_rmt, cov_rmt, _ = rmt_denoise(returns)
    
    # Factor
    T, N = returns.shape
    cov_sample = np.cov(returns.T)
    eigvals, eigvecs = np.linalg.eigh(cov_sample)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    B = eigvecs[:, :n_factors]
    factor_var = eigvals[:n_factors]
    cov_factor = B @ np.diag(factor_var) @ B.T
    residual_var = np.diag(cov_sample) - np.sum(B**2 * factor_var, axis=1)
    residual_var = np.maximum(residual_var, 1e-8)
    cov_f = cov_factor + np.diag(residual_var)
    
    cov_fused = rmt_weight * cov_rmt + (1 - rmt_weight) * cov_f
    cov_fused = ensure_psd(cov_fused)
    corr_fused = cov_to_corr(cov_fused)
    return corr_fused, cov_fused

corr_rf, cov_rf = rmt_factor_fusion(ret_vals, n_factors=20, rmt_weight=0.5)
r3 = evaluator.evaluate_static_method("RMT+Factor_rw0.5", corr_rf, cov_rf)
all_results.append(r3)
print(f"  RMT+Factor rw=0.5: NMI={r3['NMI']:.4f}, Sharpe={r3['Sharpe']:.4f}")

# RMT+Factor+Prior
for pw in [0.2, 0.3]:
    avg_var = np.diag(cov_rf).mean()
    prior_cov = industry_prior * avg_var * 0.5
    np.fill_diagonal(prior_cov, np.diag(cov_rf))
    cov_rfp = (1 - pw) * cov_rf + pw * prior_cov
    cov_rfp = ensure_psd(cov_rfp)
    corr_rfp = cov_to_corr(cov_rfp)
    r = evaluator.evaluate_static_method(f"RMT+Factor+Prior_pw{pw}", corr_rfp, cov_rfp)
    all_results.append(r)
    print(f"  RMT+Factor+Prior pw={pw}: NMI={r['NMI']:.4f}, Sharpe={r['Sharpe']:.4f}")


# ============================================================
# 4. 双路径架构 (DualPath)
# ============================================================
print("\n[4/6] Dual Path Architecture ...")

# 思路: 用高先验做聚类NMI评估, 用低先验做金融表现
# 但在综合评估中, 需要一个统一的矩阵
# 换一种思路: 构建最优相关矩阵的方法
# 即在RMT去噪基础上, 用行业先验修正off-diagonal的行业内元素, 但保持协方差精度

def dual_path_matrix(returns, prior, cluster_pw=0.5, cov_pw=0.15):
    """
    双路径:
    - 相关矩阵 = RMT + higher prior (用于聚类)
    - 协方差矩阵 = RMT + lower prior (用于投资组合)
    """
    corr_rmt, cov_rmt, _ = rmt_denoise(returns)
    
    # 聚类用相关矩阵 (高先验)
    corr_cluster = (1 - cluster_pw) * corr_rmt + cluster_pw * prior
    np.fill_diagonal(corr_cluster, 1)
    
    # 投资组合用协方差 (低先验)
    avg_var = np.diag(cov_rmt).mean()
    prior_cov = prior * avg_var * 0.5
    np.fill_diagonal(prior_cov, np.diag(cov_rmt))
    cov_portfolio = (1 - cov_pw) * cov_rmt + cov_pw * prior_cov
    cov_portfolio = ensure_psd(cov_portfolio)
    
    return corr_cluster, cov_portfolio

corr_dp, cov_dp = dual_path_matrix(ret_vals, industry_prior, cluster_pw=0.5, cov_pw=0.15)
r4 = evaluator.evaluate_static_method("DualPath_cp0.5_pp0.15", corr_dp, cov_dp)
all_results.append(r4)
print(f"  DualPath cp=0.5 pp=0.15: NMI={r4['NMI']:.4f}, Sharpe={r4['Sharpe']:.4f}, RankIC={r4['RankIC']:.4f}")

# 尝试不同组合
for cp, pp in [(0.6, 0.10), (0.5, 0.20), (0.7, 0.15), (0.6, 0.15)]:
    corr_dp, cov_dp = dual_path_matrix(ret_vals, industry_prior, cluster_pw=cp, cov_pw=pp)
    r = evaluator.evaluate_static_method(f"DualPath_cp{cp}_pp{pp}", corr_dp, cov_dp)
    all_results.append(r)
    print(f"  DualPath cp={cp} pp={pp}: NMI={r['NMI']:.4f}, Sharpe={r['Sharpe']:.4f}")


# ============================================================
# 5. Adaptive RMT+Prior (动态版)
# ============================================================
print("\n[5/6] Adaptive RMT+Prior (dynamic) ...")

class AdaptiveRMTPrior:
    """动态EWMA + 每步RMT去噪 + 低先验权重"""
    def __init__(self, n, half_life=252, prior_weight=0.25, prior_matrix=None):
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
    
    def _rmt_denoise(self, cov):
        n = cov.shape[0]
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-10
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 1)
        T = max(self.count, n + 1)
        q = n / T
        lambda_plus = (1 + np.sqrt(q)) ** 2
        eigvals, eigvecs = np.linalg.eigh(corr)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        n_signal = max(np.sum(eigvals > lambda_plus), 1)
        noise_mean = np.mean(eigvals[n_signal:]) if n_signal < n else 1.0
        denoised = eigvals.copy()
        denoised[n_signal:] = noise_mean
        corr_d = eigvecs @ np.diag(denoised) @ eigvecs.T
        np.fill_diagonal(corr_d, 1)
        return corr_d * np.outer(std, std)
    
    def get_cov(self):
        if self.cov_ewma is None:
            return np.eye(self.n) * 0.001
        cov = self.cov_ewma.copy()
        # LW收缩
        shrinkage = min(0.8, max(0.0, self.n / max(self.count, 1)))
        target = np.diag(np.diag(cov))
        cov = (1 - shrinkage) * cov + shrinkage * target
        # RMT去噪
        cov = self._rmt_denoise(cov)
        # 低先验
        if self.prior_matrix is not None and self.prior_weight > 0:
            avg_var = np.diag(cov).mean()
            prior_cov = self.prior_matrix * avg_var * 0.5
            np.fill_diagonal(prior_cov, np.diag(cov))
            cov = (1 - self.prior_weight) * cov + self.prior_weight * prior_cov
        cov = ensure_psd(cov)
        return cov
    
    def get_corr(self):
        cov = self.get_cov()
        return cov_to_corr(cov)

for pw in [0.20, 0.25, 0.30]:
    r = evaluator.evaluate_dynamic_method(
        f"AdaptRMT+Prior_pw{pw:.2f}",
        lambda pw=pw: AdaptiveRMTPrior(n_stocks, half_life=252, prior_weight=pw, prior_matrix=industry_prior)
    )
    all_results.append(r)
    print(f"  AdaptRMT pw={pw}: NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}")


# ============================================================
# 6. POET + Industry Prior (最优组合)
# ============================================================
print("\n[6/6] POET + Prior fine-tuning ...")

for nf in [15, 20]:
    for pw in [0.25, 0.30, 0.35]:
        corr_p, cov_p = poet_estimator(ret_vals, n_factors=nf, threshold_const=0.5)
        avg_var = np.diag(cov_p).mean()
        prior_cov = industry_prior * avg_var * 0.5
        np.fill_diagonal(prior_cov, np.diag(cov_p))
        cov_pp = (1 - pw) * cov_p + pw * prior_cov
        cov_pp = ensure_psd(cov_pp)
        corr_pp = cov_to_corr(cov_pp)
        r = evaluator.evaluate_static_method(f"POET+Prior_k{nf}_pw{pw:.2f}", corr_pp, cov_pp)
        all_results.append(r)
        print(f"  POET+Prior k={nf} pw={pw:.2f}: NMI={r['NMI']:.4f}, Sharpe={r['Sharpe']:.4f}")


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print("迭代3: 混合模型 - 结果汇总")
print("=" * 70)

# 加入参考
iter_best = [
    {'method': '[REF] RMT+Prior_pw0.3 (iter2)', 'NMI': 0.8924, 'ARI': 0.6554,
     'Modularity': 0.7616, 'IC': 0.8286, 'CovError': 0.4238,
     'LogLik': 711.6, 'RankIC': 0.8663, 'Sharpe': 0.7727, 'Sortino': 1.0235,
     'MaxDD': 0.1284, 'Calmar': 0.8339, 'NMI_Std': 0.0, 'CovErr_Std': 0.0},
]

display_results = iter_best + all_results
df = format_results_table(display_results, sort_by='NMI', ascending=False)
df['CompositeScore'] = compute_composite_score(df)
df = df.sort_values('CompositeScore', ascending=False).reset_index(drop=True)
df.index = df.index + 1
df.index.name = 'Rank'

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 220)
pd.set_option('display.float_format', '{:.4f}'.format)
print(df.head(20).to_string())

df.to_csv(f'{OUT_DIR}/comprehensive_eval.csv')
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

# Top 5
print("\n--- 迭代3 Top 5 (by CompositeScore) ---")
new_only = df[~df['method'].str.startswith('[REF]')].head(5)
for _, row in new_only.iterrows():
    print(f"  {row['method']}: NMI={row['NMI']:.4f}, Sharpe={row['Sharpe']:.4f}, RankIC={row.get('RankIC', 0):.4f}, Composite={row['CompositeScore']:.4f}")

print(f"\n结果已保存到 {OUT_DIR}/")
print("\n迭代3完成!")
