"""
迭代6 (修正版): Walk-Forward评估, 消除数据泄露
所有方法统一接口: estimator_fn(returns_history) -> (corr, cov)
在每个评估时刻t, 只用[0,t)的数据估计, 用[t,t+60)的数据验证
"""
import sys
sys.path.insert(0, '/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation')

import pandas as pd
import numpy as np
from eval_framework_v2 import (
    WalkForwardEvaluator, build_industry_prior, cov_to_corr, ensure_psd,
    format_results_table, compute_composite_score, eval_portfolio_metrics
)
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/iter6_corrected"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("迭代6 (修正版): Walk-Forward 无泄露评估")
print("=" * 70)

returns_df = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns_df.columns.tolist()
n_stocks = len(stocks)
ret_vals = returns_df.values
industry_prior = build_industry_prior(stocks, code_to_industry)

print(f"股票数: {n_stocks}, 交易日: {len(ret_vals)}")

# warmup=500 确保足够的历史数据
evaluator = WalkForwardEvaluator(
    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
    industry_prior=industry_prior,
    warmup=500, eval_freq=60, forecast=60, rebalance=60,
    n_clusters=35, topk=4
)

evaluator_k3 = WalkForwardEvaluator(
    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
    industry_prior=industry_prior,
    warmup=500, eval_freq=60, forecast=60, rebalance=60,
    n_clusters=35, topk=3
)

all_results = []


# ============================================================
# 辅助: 去噪/估计函数 (只用输入的history数据)
# ============================================================

def rmt_denoise_fn(history):
    """RMT去噪: 只用history数据"""
    T, N = history.shape
    q = N / T
    cov = np.cov(history.T)
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1e-10
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1)
    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    lambda_plus = (1 + np.sqrt(q)) ** 2
    n_signal = max(np.sum(eigvals > lambda_plus), 1)
    noise_mean = np.mean(eigvals[n_signal:]) if n_signal < N else 1.0
    denoised = eigvals.copy()
    denoised[n_signal:] = noise_mean
    corr_d = eigvecs @ np.diag(denoised) @ eigvecs.T
    np.fill_diagonal(corr_d, 1)
    cov_d = corr_d * np.outer(std, std)
    return cov_to_corr(ensure_psd(cov_d)), ensure_psd(cov_d)


def poet_fn(history, n_factors=15, threshold_const=0.5):
    """POET估计: 只用history数据"""
    T, N = history.shape
    cov_sample = np.cov(history.T)
    eigvals, eigvecs = np.linalg.eigh(cov_sample)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    B = eigvecs[:, :n_factors]
    fv = eigvals[:n_factors]
    cov_factor = B @ np.diag(fv) @ B.T
    residual = cov_sample - cov_factor
    threshold = threshold_const * np.sqrt(np.log(N) / T)
    res_thresh = residual.copy()
    for i in range(N):
        for j in range(N):
            if i != j:
                val = res_thresh[i, j]
                scale = np.sqrt(abs(residual[i,i] * residual[j,j]))
                res_thresh[i, j] = np.sign(val) * max(abs(val) - threshold * scale, 0)
    cov_poet = ensure_psd(cov_factor + res_thresh)
    return cov_to_corr(cov_poet), cov_poet


def dual_path_fn(base_fn, prior, cluster_pw=0.6, cov_pw=0.0):
    """DualPath: 聚类用先验融合相关矩阵, 协方差不加先验"""
    def estimator(history):
        corr_base, cov_base = base_fn(history)
        corr_cluster = (1 - cluster_pw) * corr_base + cluster_pw * prior
        np.fill_diagonal(corr_cluster, 1)
        if cov_pw > 0:
            avg_var = np.diag(cov_base).mean()
            prior_cov = prior * avg_var * 0.5
            np.fill_diagonal(prior_cov, np.diag(cov_base))
            cov_out = (1 - cov_pw) * cov_base + cov_pw * prior_cov
            cov_out = ensure_psd(cov_out)
        else:
            cov_out = cov_base
        return corr_cluster, cov_out
    return estimator


# ============================================================
# 1. 基线方法
# ============================================================
print("\n--- 基线方法 ---")

# 1a. 样本Pearson
print("[1] Sample Pearson ...")
def sample_pearson(history):
    cov = np.cov(history.T)
    return cov_to_corr(cov), cov
r = evaluator.evaluate("SamplePearson_K4", sample_pearson)
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}, RankIC={r['RankIC']:.4f}")

# 1b. 等权
print("[2] Equal Weight ...")
eq_rets = ret_vals[500:].mean(axis=1)
pm_eq = eval_portfolio_metrics(eq_rets.tolist())
r_eq = {'method': 'EqualWeight',
        'NMI': np.nan, 'ARI': np.nan, 'Modularity': np.nan, 'IC': np.nan,
        'CovError': np.nan, 'LogLik': np.nan, 'RankIC': np.nan,
        'Sharpe': pm_eq['sharpe'], 'Sortino': pm_eq['sortino'],
        'MaxDD': pm_eq['max_drawdown'], 'Calmar': pm_eq['calmar'],
        'NMI_Std': np.nan, 'CovErr_Std': np.nan}
all_results.append(r_eq)
print(f"  Sharpe={r_eq['Sharpe']:.4f}")

# 1c. LedoitWolf
print("[3] LedoitWolf ...")
from sklearn.covariance import LedoitWolf
def lw_estimator(history):
    lw = LedoitWolf()
    lw.fit(history)
    return cov_to_corr(lw.covariance_), lw.covariance_
r = evaluator.evaluate("LedoitWolf_K4", lw_estimator)
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}, RankIC={r['RankIC']:.4f}")

# ============================================================
# 2. 去噪方法
# ============================================================
print("\n--- 去噪方法 ---")

# 2a. RMT
print("[4] RMT Denoise ...")
r = evaluator.evaluate("RMT_K4", rmt_denoise_fn)
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}, RankIC={r['RankIC']:.4f}")

# 2b. POET15
print("[5] POET15 ...")
def poet15_fn(history): return poet_fn(history, 15)
r = evaluator.evaluate("POET15_K4", poet15_fn)
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}, RankIC={r['RankIC']:.4f}")

# 2c. PCA Factor k=20
print("[6] PCA Factor k=20 ...")
def pca_factor_fn(history, n_factors=20):
    cov_sample = np.cov(history.T)
    eigvals, eigvecs = np.linalg.eigh(cov_sample)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    B = eigvecs[:, :n_factors]
    fv = eigvals[:n_factors]
    cov_f = B @ np.diag(fv) @ B.T
    res_var = np.maximum(np.diag(cov_sample) - np.sum(B**2 * fv, axis=1), 1e-8)
    cov_est = ensure_psd(cov_f + np.diag(res_var))
    return cov_to_corr(cov_est), cov_est
r = evaluator.evaluate("PCA_Factor_k20", pca_factor_fn)
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}")

# ============================================================
# 3. 先验融合方法 (单路径)
# ============================================================
print("\n--- 先验融合方法 (单路径) ---")

# 3a. RMT + Prior pw=0.3
print("[7] RMT+Prior pw=0.3 ...")
def rmt_prior_fn(pw):
    def fn(history):
        corr_rmt, cov_rmt = rmt_denoise_fn(history)
        avg_var = np.diag(cov_rmt).mean()
        prior_cov = industry_prior * avg_var * 0.5
        np.fill_diagonal(prior_cov, np.diag(cov_rmt))
        cov_out = (1 - pw) * cov_rmt + pw * prior_cov
        return cov_to_corr(ensure_psd(cov_out)), ensure_psd(cov_out)
    return fn
r = evaluator.evaluate("RMT+Prior_pw0.3", rmt_prior_fn(0.3))
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}, RankIC={r['RankIC']:.4f}")

# 3b. Adaptive EWMA pw=0.7 (原项目最佳NMI方法)
print("[8] Adaptive EWMA pw=0.7 ...")
def adaptive_ewma_fn(history, hl=252, pw=0.7):
    """用EWMA方式处理, 但只基于history"""
    T, N = history.shape
    decay = np.log(2) / hl
    cov_ewma = np.cov(history[:20].T)
    count = 20
    for t in range(20, T, 20):
        batch = history[t:t+20]
        if len(batch) < 5:
            continue
        batch_cov = np.cov(batch.T)
        alpha = 1 - np.exp(-decay * len(batch))
        cov_ewma = (1 - alpha) * cov_ewma + alpha * batch_cov
        count += len(batch)
    # 收缩
    shrinkage = min(0.8, max(0.0, N / max(count, 1)))
    target = np.diag(np.diag(cov_ewma))
    cov = (1 - shrinkage) * cov_ewma + shrinkage * target
    # 行业先验
    avg_var = np.diag(cov).mean()
    prior_cov = industry_prior * avg_var * 0.5
    np.fill_diagonal(prior_cov, 0)
    cov = (1 - pw) * cov + pw * (cov + prior_cov)
    cov = ensure_psd(cov)
    return cov_to_corr(cov), cov
r = evaluator.evaluate("Adaptive_EWMA_pw0.7", adaptive_ewma_fn)
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}, RankIC={r['RankIC']:.4f}")


# ============================================================
# 4. DualPath方法 (核心, K=3和K=4)
# ============================================================
print("\n--- DualPath方法 ---")

# 4a. DP_RMT K4
print("[9] DP_RMT cp0.6 K4 ...")
r = evaluator.evaluate("DP_RMT_cp0.6_K4",
    dual_path_fn(rmt_denoise_fn, industry_prior, 0.6, 0.0))
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}, RankIC={r['RankIC']:.4f}")

# 4b. DP_RMT K3
print("[10] DP_RMT cp0.6 K3 ...")
r = evaluator_k3.evaluate("DP_RMT_cp0.6_K3",
    dual_path_fn(rmt_denoise_fn, industry_prior, 0.6, 0.0))
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}, RankIC={r['RankIC']:.4f}")

# 4c. DP_POET15 K4
print("[11] DP_POET15 cp0.6 K4 ...")
r = evaluator.evaluate("DP_POET15_cp0.6_K4",
    dual_path_fn(poet15_fn, industry_prior, 0.6, 0.0))
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}, RankIC={r['RankIC']:.4f}")

# 4d. DP_POET15 K3
print("[12] DP_POET15 cp0.6 K3 ...")
r = evaluator_k3.evaluate("DP_POET15_cp0.6_K3",
    dual_path_fn(poet15_fn, industry_prior, 0.6, 0.0))
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}, RankIC={r['RankIC']:.4f}")

# 4e. DP_RMT+Prior pw=0.3 (corr用先验, cov也轻微融合)
print("[13] DP_RMT cp0.6 pp0.1 K4 ...")
r = evaluator.evaluate("DP_RMT_cp0.6_pp0.1_K4",
    dual_path_fn(rmt_denoise_fn, industry_prior, 0.6, 0.1))
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}")

# 4f. 纯行业先验 + RMT协方差 (DualPath极端情况)
print("[14] DP_RMT cp1.0 K4 (pure prior clustering) ...")
r = evaluator.evaluate("DP_RMT_cp1.0_K4",
    dual_path_fn(rmt_denoise_fn, industry_prior, 1.0, 0.0))
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}")

# 4g. DP_POET15 + higher cp
print("[15] DP_POET15 cp0.7 K3 ...")
r = evaluator_k3.evaluate("DP_POET15_cp0.7_K3",
    dual_path_fn(poet15_fn, industry_prior, 0.7, 0.0))
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}")

# 4h. DP with Factor base
print("[16] DP_Factor_k20 cp0.6 K3 ...")
r = evaluator_k3.evaluate("DP_Factor_cp0.6_K3",
    dual_path_fn(pca_factor_fn, industry_prior, 0.6, 0.0))
all_results.append(r)
print(f"  NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}")


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print("Walk-Forward 修正版 - 最终排名")
print("=" * 70)

df = format_results_table(all_results, sort_by='NMI', ascending=False)
df['CompositeScore'] = compute_composite_score(df)
df = df.sort_values('CompositeScore', ascending=False).reset_index(drop=True)
df.index = df.index + 1
df.index.name = 'Rank'

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 240)
pd.set_option('display.float_format', '{:.4f}'.format)
print(df.to_string())

df.to_csv(f'{OUT_DIR}/final_ranking_corrected.csv')
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

# Markdown格式
print("\n--- Top 10 Markdown ---")
print("| Rank | Method | NMI | ARI | Modul. | IC | CovErr | LogLik | RankIC | Sharpe | Sortino | MaxDD | Calmar | Composite |")
print("|:----:|--------|:---:|:---:|:------:|:--:|:------:|:------:|:------:|:------:|:-------:|:-----:|:------:|:---------:|")
for idx, row in df.head(10).iterrows():
    nmi = f"{row['NMI']:.4f}" if not np.isnan(row['NMI']) else "—"
    print(f"| {idx} | {row['method']} | {nmi} | {row['ARI']:.4f} | {row['Modularity']:.4f} | {row['IC']:.4f} | {row['CovError']:.4f} | {row['LogLik']:.1f} | {row['RankIC']:.4f} | {row['Sharpe']:.4f} | {row['Sortino']:.4f} | {row['MaxDD']:.4f} | {row['Calmar']:.4f} | {row['CompositeScore']:.4f} |")

print(f"\n结果已保存到 {OUT_DIR}/")
print("\n修正版评估完成!")
