"""
迭代5: 最终精调与全量排名
1. DP_RMT_K3 + nc=40 验证
2. DP_POET_K3 验证
3. 跨nc和TopK的完整网格
4. 全量模型 (5轮迭代) 综合排名
5. 生成Top-10表
"""
import sys
sys.path.insert(0, '/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation')

import pandas as pd
import numpy as np
from eval_framework import (
    ComprehensiveEvaluator, build_industry_prior, topk_adj,
    format_results_table, compute_composite_score, eval_nmi, eval_ari,
    eval_portfolio_metrics, min_var_weights
)
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/iter5"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("迭代5: 最终精调与全量排名")
print("=" * 70)

returns_df = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns_df.columns.tolist()
n_stocks = len(stocks)
ret_vals = returns_df.values
industry_prior = build_industry_prior(stocks, code_to_industry)

print(f"股票数: {n_stocks}, 交易日: {len(ret_vals)}")


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
    n_signal = max(np.sum(eigvals > lambda_plus), 1) if n_factors is None else n_factors
    noise_mean = np.mean(eigvals[n_signal:]) if n_signal < N else 1.0
    denoised = eigvals.copy()
    denoised[n_signal:] = noise_mean
    corr_d = eigvecs @ np.diag(denoised) @ eigvecs.T
    np.fill_diagonal(corr_d, 1)
    cov_d = corr_d * np.outer(std, std)
    return cov_to_corr(ensure_psd(cov_d)), ensure_psd(cov_d), n_signal

def poet_estimator(returns, n_factors=15, threshold_const=0.5):
    T, N = returns.shape
    cov_sample = np.cov(returns.T)
    eigvals, eigvecs = np.linalg.eigh(cov_sample)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    B = eigvecs[:, :n_factors]
    factor_var = eigvals[:n_factors]
    cov_factor = B @ np.diag(factor_var) @ B.T
    residual = cov_sample - cov_factor
    threshold = threshold_const * np.sqrt(np.log(N) / T)
    residual_thresh = residual.copy()
    for i in range(N):
        for j in range(N):
            if i != j:
                val = residual_thresh[i, j]
                scale = np.sqrt(abs(residual[i,i] * residual[j,j]))
                residual_thresh[i, j] = np.sign(val) * max(abs(val) - threshold * scale, 0)
    cov_poet = cov_factor + residual_thresh
    return cov_to_corr(ensure_psd(cov_poet)), ensure_psd(cov_poet)

def dual_path(corr_base, cov_base, prior, cluster_pw, cov_pw=0.0):
    corr_cluster = (1 - cluster_pw) * corr_base + cluster_pw * prior
    np.fill_diagonal(corr_cluster, 1)
    if cov_pw > 0:
        avg_var = np.diag(cov_base).mean()
        prior_cov = prior * avg_var * 0.5
        np.fill_diagonal(prior_cov, np.diag(cov_base))
        cov_portfolio = (1 - cov_pw) * cov_base + cov_pw * prior_cov
        cov_portfolio = ensure_psd(cov_portfolio)
    else:
        cov_portfolio = cov_base
    return corr_cluster, cov_portfolio

# 预计算基座
corr_rmt, cov_rmt, n_sig = rmt_denoise(ret_vals)
corr_poet, cov_poet = poet_estimator(ret_vals, n_factors=15, threshold_const=0.5)
corr_poet20, cov_poet20 = poet_estimator(ret_vals, n_factors=20, threshold_const=0.5)
cov_sample = np.cov(ret_vals.T)
corr_sample = np.corrcoef(ret_vals.T)

all_results = []

# ============================================================
# 1. 完整网格: base × cp × TopK × nc (pp=0.00)
# ============================================================
print("\n[1/3] Complete Grid Search ...")

bases = {
    'RMT': (corr_rmt, cov_rmt),
    'POET15': (corr_poet, cov_poet),
    'POET20': (corr_poet20, cov_poet20),
}

cp_values = [0.5, 0.6]
topk_values = [3, 4]
nc_values = [35, 40]

for base_name, (corr_b, cov_b) in bases.items():
    for cp in cp_values:
        corr_dp, cov_dp = dual_path(corr_b, cov_b, industry_prior, cp, 0.0)
        for tk in topk_values:
            for nc in nc_values:
                evaluator = ComprehensiveEvaluator(
                    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
                    industry_prior=industry_prior, n_clusters=nc, topk=tk
                )
                name = f"DP_{base_name}_cp{cp}_K{tk}_nc{nc}"
                r = evaluator.evaluate_static_method(name, corr_dp, cov_dp)
                all_results.append(r)
                print(f"  {name}: NMI={r['NMI']:.4f}, Sharpe={r['Sharpe']:.4f}")


# ============================================================
# 2. 原始方法的最佳变体作为参考
# ============================================================
print("\n[2/3] Reference Methods ...")

# Static Pearson (iter1 #1 composite)
evaluator_ref = ComprehensiveEvaluator(
    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
    industry_prior=industry_prior, n_clusters=35, topk=4
)
r_sp = evaluator_ref.evaluate_static_method("StaticPearson_K4", corr_sample, cov_sample)
all_results.append(r_sp)
print(f"  StaticPearson K4: NMI={r_sp['NMI']:.4f}, Sharpe={r_sp['Sharpe']:.4f}")

# Nonlinear Shrinkage
def nonlinear_shrinkage(returns):
    T, N = returns.shape
    cov_sample = np.cov(returns.T)
    eigvals, eigvecs = np.linalg.eigh(cov_sample)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    shrunk_eigvals = np.zeros_like(eigvals)
    q = N / T
    for i in range(N):
        diffs = eigvals - eigvals[i]
        diffs[i] = 1
        h = max(eigvals[i] * (N/T)**0.5 * 0.1, 1e-8)
        kernel_vals = np.sum(h / (diffs**2 + h**2)) / (N * np.pi)
        sf = 1.0 / (1 + q * kernel_vals * np.pi * eigvals[i])
        shrunk_eigvals[i] = eigvals[i] * max(sf, 0.01)
    shrunk_eigvals = np.maximum(shrunk_eigvals, 1e-8)
    cov_s = eigvecs @ np.diag(shrunk_eigvals) @ eigvecs.T
    return cov_to_corr(ensure_psd(cov_s)), ensure_psd(cov_s)

corr_nls, cov_nls = nonlinear_shrinkage(ret_vals)
r_nls = evaluator_ref.evaluate_static_method("NonlinearShrinkage", corr_nls, cov_nls)
all_results.append(r_nls)
print(f"  NonlinearShrinkage: NMI={r_nls['NMI']:.4f}, Sharpe={r_nls['Sharpe']:.4f}")

# RMT+Prior pw=0.3 (iter2 #1)
avg_var = np.diag(cov_rmt).mean()
prior_cov_03 = industry_prior * avg_var * 0.5
np.fill_diagonal(prior_cov_03, np.diag(cov_rmt))
cov_rp03 = 0.7 * cov_rmt + 0.3 * prior_cov_03
cov_rp03 = ensure_psd(cov_rp03)
corr_rp03 = cov_to_corr(cov_rp03)
r_rp = evaluator_ref.evaluate_static_method("RMT+Prior_pw0.3", corr_rp03, cov_rp03)
all_results.append(r_rp)
print(f"  RMT+Prior pw=0.3: NMI={r_rp['NMI']:.4f}, Sharpe={r_rp['Sharpe']:.4f}")

# Adaptive EWMA pw=0.7 (原始最佳 NMI)
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

r_adap = evaluator_ref.evaluate_dynamic_method(
    "Adaptive_pw0.7_nc35_K4",
    lambda: AdaptiveCovEstimator(n_stocks, half_life=252, prior_weight=0.7, prior_matrix=industry_prior)
)
all_results.append(r_adap)
print(f"  Adaptive pw=0.7: NMI={r_adap['NMI']:.4f}, Sharpe={r_adap['Sharpe']:.4f}")

# PCA Factor k=20 (iter2 best Sharpe)
def pca_factor_cov(returns, n_factors=20):
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
    cov_est = cov_factor + np.diag(residual_var)
    return cov_to_corr(ensure_psd(cov_est)), ensure_psd(cov_est)

corr_f20, cov_f20 = pca_factor_cov(ret_vals, 20)
r_f20 = evaluator_ref.evaluate_static_method("PCA_Factor_k20", corr_f20, cov_f20)
all_results.append(r_f20)
print(f"  PCA Factor k=20: NMI={r_f20['NMI']:.4f}, Sharpe={r_f20['Sharpe']:.4f}")

# Pure RMT Denoise
r_rmt = evaluator_ref.evaluate_static_method("RMT_Denoise", corr_rmt, cov_rmt)
all_results.append(r_rmt)
print(f"  RMT Denoise: NMI={r_rmt['NMI']:.4f}, Sharpe={r_rmt['Sharpe']:.4f}")


# ============================================================
# 3. 最终精调: 最佳配置的微调
# ============================================================
print("\n[3/3] Final Fine-tuning ...")

# TopK=3 + 不同nc for RMT base
for nc in [33, 34, 35, 36, 37, 38, 39, 40]:
    corr_dp, cov_dp = dual_path(corr_rmt, cov_rmt, industry_prior, 0.6, 0.0)
    evaluator_nc = ComprehensiveEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, n_clusters=nc, topk=3
    )
    r = evaluator_nc.evaluate_static_method(f"DP_RMT_K3_nc{nc}", corr_dp, cov_dp)
    all_results.append(r)
    print(f"  DP_RMT K3 nc={nc}: NMI={r['NMI']:.4f}, ARI={r['ARI']:.4f}")

# TopK=3 + 不同nc for POET15 base
for nc in [35, 38, 40]:
    corr_dp, cov_dp = dual_path(corr_poet, cov_poet, industry_prior, 0.6, 0.0)
    evaluator_nc = ComprehensiveEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, n_clusters=nc, topk=3
    )
    r = evaluator_nc.evaluate_static_method(f"DP_POET15_K3_nc{nc}", corr_dp, cov_dp)
    all_results.append(r)
    print(f"  DP_POET15 K3 nc={nc}: NMI={r['NMI']:.4f}")


# ============================================================
# 综合排名
# ============================================================
print("\n" + "=" * 70)
print("迭代5: 最终排名 (All Methods)")
print("=" * 70)

df = format_results_table(all_results, sort_by='NMI', ascending=False)
df['CompositeScore'] = compute_composite_score(df)
df = df.sort_values('CompositeScore', ascending=False).reset_index(drop=True)
df.index = df.index + 1
df.index.name = 'Rank'

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 240)
pd.set_option('display.float_format', '{:.4f}'.format)

# Top 20
print("\n--- Top 20 ---")
print(df.head(20).to_string())

# 保存全部
df.to_csv(f'{OUT_DIR}/final_ranking.csv')
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

# Top 10 for README
print("\n" + "=" * 70)
print("TOP 10 最终结果 (for README)")
print("=" * 70)
top10 = df.head(10)
print(top10.to_string())

# 保存top10
top10.to_csv(f'{OUT_DIR}/top10.csv')

# 生成markdown格式
print("\n--- Markdown格式 ---")
print("| Rank | Method | NMI | ARI | Modularity | IC | CovError | LogLik | RankIC | Sharpe | Sortino | MaxDD | Calmar | Composite |")
print("|:----:|--------|:---:|:---:|:----------:|:--:|:--------:|:------:|:------:|:------:|:-------:|:-----:|:------:|:---------:|")
for idx, row in top10.iterrows():
    print(f"| {idx} | {row['method']} | {row['NMI']:.4f} | {row['ARI']:.4f} | {row['Modularity']:.4f} | {row['IC']:.4f} | {row['CovError']:.4f} | {row['LogLik']:.1f} | {row['RankIC']:.4f} | {row['Sharpe']:.4f} | {row['Sortino']:.4f} | {row['MaxDD']:.4f} | {row['Calmar']:.4f} | {row['CompositeScore']:.4f} |")

print(f"\n结果已保存到 {OUT_DIR}/")
print("\n迭代5完成! 全部5轮迭代结束。")
