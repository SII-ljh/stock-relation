"""
迭代4: 高级集成与优化
基于迭代3的DualPath突破:
  1. DualPath精调: pp=0.05/0.08, 不同n_clusters
  2. DualPath + POET基座
  3. DualPath + NLS基座 (Nonlinear Shrinkage)
  4. NMI提升: 尝试更高cp和不同n_clusters
  5. 加权集成
  6. RMT去噪 + 目标收缩 + DualPath
"""
import sys
sys.path.insert(0, '/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation')

import pandas as pd
import numpy as np
from eval_framework import (
    ComprehensiveEvaluator, build_industry_prior, topk_adj,
    format_results_table, compute_composite_score, eval_nmi
)
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/iter4"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("迭代4: 高级集成与优化")
print("=" * 70)

returns_df = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns_df.columns.tolist()
n_stocks = len(stocks)
ret_vals = returns_df.values
industry_prior = build_industry_prior(stocks, code_to_industry)

print(f"股票数: {n_stocks}, 交易日: {len(ret_vals)}")

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
    n_signal = max(np.sum(eigvals > lambda_plus), 1) if n_factors is None else n_factors
    noise_mean = np.mean(eigvals[n_signal:]) if n_signal < N else 1.0
    denoised = eigvals.copy()
    denoised[n_signal:] = noise_mean
    corr_d = eigvecs @ np.diag(denoised) @ eigvecs.T
    np.fill_diagonal(corr_d, 1)
    cov_d = corr_d * np.outer(std, std)
    return cov_to_corr(cov_d), ensure_psd(cov_d), n_signal

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

def dual_path(corr_base, cov_base, prior, cluster_pw, cov_pw):
    """通用双路径构造"""
    corr_cluster = (1 - cluster_pw) * corr_base + cluster_pw * prior
    np.fill_diagonal(corr_cluster, 1)
    
    avg_var = np.diag(cov_base).mean()
    prior_cov = prior * avg_var * 0.5
    np.fill_diagonal(prior_cov, np.diag(cov_base))
    cov_portfolio = (1 - cov_pw) * cov_base + cov_pw * prior_cov
    cov_portfolio = ensure_psd(cov_portfolio)
    
    return corr_cluster, cov_portfolio


# ============================================================
# 1. DualPath + RMT 精调 (极低pp)
# ============================================================
print("\n[1/6] DualPath+RMT Fine-tuning (very low pp) ...")

corr_rmt, cov_rmt, _ = rmt_denoise(ret_vals)

for cp in [0.5, 0.6, 0.7]:
    for pp in [0.0, 0.05, 0.08, 0.10, 0.12]:
        corr_dp, cov_dp = dual_path(corr_rmt, cov_rmt, industry_prior, cp, pp)
        evaluator_tmp = ComprehensiveEvaluator(
            returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
            industry_prior=industry_prior, n_clusters=35, topk=4
        )
        r = evaluator_tmp.evaluate_static_method(f"DP_RMT_cp{cp}_pp{pp:.2f}", corr_dp, cov_dp)
        all_results.append(r)
        print(f"  cp={cp} pp={pp:.2f}: NMI={r['NMI']:.4f}, Sharpe={r['Sharpe']:.4f}, RankIC={r['RankIC']:.4f}")


# ============================================================
# 2. DualPath + 不同n_clusters
# ============================================================
print("\n[2/6] DualPath + n_clusters tuning ...")

best_cp, best_pp = 0.6, 0.05  # 基于上面的结果选最佳
corr_dp, cov_dp = dual_path(corr_rmt, cov_rmt, industry_prior, best_cp, best_pp)

for nc in [30, 32, 35, 38, 40, 42]:
    evaluator_nc = ComprehensiveEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, n_clusters=nc, topk=4
    )
    r = evaluator_nc.evaluate_static_method(f"DP_RMT_nc{nc}", corr_dp, cov_dp)
    print(f"  nc={nc}: NMI={r['NMI']:.4f}, ARI={r['ARI']:.4f}")
    if nc in [35, 38, 42]:  # 保存感兴趣的
        r['method'] = f"DP_RMT_cp0.6_pp0.05_nc{nc}"
        all_results.append(r)


# ============================================================
# 3. DualPath + POET基座
# ============================================================
print("\n[3/6] DualPath + POET base ...")

corr_poet, cov_poet = poet_estimator(ret_vals, n_factors=15, threshold_const=0.5)

for cp in [0.5, 0.6, 0.7]:
    for pp in [0.0, 0.05, 0.10]:
        corr_dp, cov_dp = dual_path(corr_poet, cov_poet, industry_prior, cp, pp)
        evaluator_tmp = ComprehensiveEvaluator(
            returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
            industry_prior=industry_prior, n_clusters=35, topk=4
        )
        r = evaluator_tmp.evaluate_static_method(f"DP_POET_cp{cp}_pp{pp:.2f}", corr_dp, cov_dp)
        all_results.append(r)
        print(f"  POET cp={cp} pp={pp:.2f}: NMI={r['NMI']:.4f}, Sharpe={r['Sharpe']:.4f}")


# ============================================================
# 4. DualPath + TopK tuning
# ============================================================
print("\n[4/6] DualPath + TopK tuning ...")

corr_dp_best, cov_dp_best = dual_path(corr_rmt, cov_rmt, industry_prior, 0.6, 0.05)

for tk in [3, 4, 5, 6]:
    evaluator_tk = ComprehensiveEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, n_clusters=35, topk=tk
    )
    r = evaluator_tk.evaluate_static_method(f"DP_RMT_K{tk}", corr_dp_best, cov_dp_best)
    all_results.append(r)
    print(f"  TopK={tk}: NMI={r['NMI']:.4f}, ARI={r['ARI']:.4f}")


# ============================================================
# 5. 加权集成 (Ensemble)
# ============================================================
print("\n[5/6] Weighted Ensemble ...")

# 集成RMT和POET的协方差
for w_rmt in [0.3, 0.5, 0.7]:
    cov_ens = w_rmt * cov_rmt + (1 - w_rmt) * cov_poet
    cov_ens = ensure_psd(cov_ens)
    corr_ens = cov_to_corr(cov_ens)
    
    # 用DualPath
    corr_dp_e, cov_dp_e = dual_path(corr_ens, cov_ens, industry_prior, 0.6, 0.05)
    evaluator_tmp = ComprehensiveEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, n_clusters=35, topk=4
    )
    r = evaluator_tmp.evaluate_static_method(f"DP_Ensemble_rmt{w_rmt}", corr_dp_e, cov_dp_e)
    all_results.append(r)
    print(f"  Ensemble rmt_w={w_rmt}: NMI={r['NMI']:.4f}, Sharpe={r['Sharpe']:.4f}")


# ============================================================
# 6. 增强RMT去噪 (Target Shrinkage + RMT)
# ============================================================
print("\n[6/6] Enhanced RMT (target shrinkage + RMT) ...")

def enhanced_rmt(returns, shrinkage_alpha=0.1):
    """先做目标收缩, 再RMT去噪"""
    T, N = returns.shape
    cov_sample = np.cov(returns.T)
    
    # 目标收缩到对角
    target = np.diag(np.diag(cov_sample))
    cov_shrunk = (1 - shrinkage_alpha) * cov_sample + shrinkage_alpha * target
    
    # RMT on shrunk matrix
    std = np.sqrt(np.diag(cov_shrunk))
    std[std == 0] = 1e-10
    corr = cov_shrunk / np.outer(std, std)
    np.fill_diagonal(corr, 1)
    
    q = N / T
    lambda_plus = (1 + np.sqrt(q)) ** 2
    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    n_signal = max(np.sum(eigvals > lambda_plus), 1)
    noise_mean = np.mean(eigvals[n_signal:]) if n_signal < N else 1.0
    denoised = eigvals.copy()
    denoised[n_signal:] = noise_mean
    
    corr_d = eigvecs @ np.diag(denoised) @ eigvecs.T
    np.fill_diagonal(corr_d, 1)
    cov_d = corr_d * np.outer(std, std)
    return cov_to_corr(ensure_psd(cov_d)), ensure_psd(cov_d)

for sa in [0.05, 0.1, 0.15]:
    corr_ermt, cov_ermt = enhanced_rmt(ret_vals, shrinkage_alpha=sa)
    corr_dp_e, cov_dp_e = dual_path(corr_ermt, cov_ermt, industry_prior, 0.6, 0.05)
    evaluator_tmp = ComprehensiveEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, n_clusters=35, topk=4
    )
    r = evaluator_tmp.evaluate_static_method(f"DP_EnhRMT_sa{sa}", corr_dp_e, cov_dp_e)
    all_results.append(r)
    print(f"  Enhanced RMT sa={sa}: NMI={r['NMI']:.4f}, Sharpe={r['Sharpe']:.4f}")


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print("迭代4: 高级集成 - 结果汇总 (Top 20)")
print("=" * 70)

df = format_results_table(all_results, sort_by='NMI', ascending=False)
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

print("\n--- 迭代4 Top 5 ---")
for _, row in df.head(5).iterrows():
    print(f"  {row['method']}: NMI={row['NMI']:.4f}, Sharpe={row['Sharpe']:.4f}, RankIC={row.get('RankIC',0):.4f}, Composite={row['CompositeScore']:.4f}")

print(f"\n结果已保存到 {OUT_DIR}/")
print("\n迭代4完成!")
