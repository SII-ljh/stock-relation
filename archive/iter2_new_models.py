"""
迭代2: 新模型架构
基于迭代1的发现, 需要同时改善NMI和金融表现。

新方法:
  1. RMT去噪 (Random Matrix Theory) - 去除噪声特征值
  2. PCA因子模型 - 低秩协方差估计
  3. 非线性收缩 (Oracle Approximating Shrinkage) - Ledoit-Wolf 2020
  4. RMT + Industry Prior - 去噪后融合行业先验
  5. Factor + Industry Prior - 因子模型+行业先验
  6. 多尺度EWMA - 融合多个半衰期
  7. Robust MCD - 最小协方差行列式
"""
import sys
sys.path.insert(0, '/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation')

import pandas as pd
import numpy as np
from sklearn.covariance import MinCovDet
from eval_framework import (
    ComprehensiveEvaluator, build_industry_prior, topk_adj,
    format_results_table, compute_composite_score
)
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/iter2"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("迭代2: 新模型架构")
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

# ============================================================
# 辅助函数
# ============================================================

def cov_to_corr(cov):
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1e-10
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1)
    return np.clip(corr, -1, 1)


def ensure_psd(cov, eps=1e-6):
    """确保正定"""
    cov = (cov + cov.T) / 2
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < eps:
        cov += np.eye(cov.shape[0]) * (eps - eigvals.min())
    return cov


# ============================================================
# 1. RMT去噪 (Marchenko-Pastur)
# ============================================================
print("\n[1/7] RMT Denoising (Marchenko-Pastur) ...")

def rmt_denoise(returns, n_factors=None):
    """
    Random Matrix Theory去噪:
    1. 对相关矩阵做特征分解
    2. 用Marchenko-Pastur分布确定噪声特征值上界
    3. 将噪声特征值替换为均值
    4. 重建去噪协方差矩阵
    """
    T, N = returns.shape
    q = N / T  # 比率
    
    # 样本协方差和相关矩阵
    cov_sample = np.cov(returns.T)
    std = np.sqrt(np.diag(cov_sample))
    std[std == 0] = 1e-10
    corr = cov_sample / np.outer(std, std)
    np.fill_diagonal(corr, 1)
    
    # 特征分解
    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Marchenko-Pastur 上界: lambda_+ = (1 + sqrt(q))^2
    lambda_plus = (1 + np.sqrt(q)) ** 2
    
    # 确定信号特征值 (超过MP上界)
    if n_factors is None:
        n_signal = np.sum(eigvals > lambda_plus)
        n_signal = max(n_signal, 1)
    else:
        n_signal = n_factors
    
    # 去噪: 保留信号特征值, 噪声特征值用均值替换
    noise_eigvals = eigvals[n_signal:]
    noise_mean = np.mean(noise_eigvals) if len(noise_eigvals) > 0 else 1.0
    
    denoised_eigvals = eigvals.copy()
    denoised_eigvals[n_signal:] = noise_mean
    
    # 重建
    corr_denoised = eigvecs @ np.diag(denoised_eigvals) @ eigvecs.T
    np.fill_diagonal(corr_denoised, 1)
    
    # 重建协方差
    cov_denoised = corr_denoised * np.outer(std, std)
    cov_denoised = ensure_psd(cov_denoised)
    
    return corr_denoised, cov_denoised, n_signal

corr_rmt, cov_rmt, n_signal = rmt_denoise(ret_vals)
print(f"  信号特征值数: {n_signal}")
r1 = evaluator.evaluate_static_method("RMT_Denoise", corr_rmt, cov_rmt)
all_results.append(r1)
print(f"  NMI={r1['NMI']:.4f}, CovErr={r1['CovError']:.4f}, Sharpe={r1['Sharpe']:.4f}, RankIC={r1['RankIC']:.4f}")


# ============================================================
# 2. PCA因子模型
# ============================================================
print("\n[2/7] PCA Factor Model ...")

def pca_factor_cov(returns, n_factors=10):
    """
    PCA因子模型:
    R = B * F + epsilon
    Cov = B * Cov_F * B' + diag(Var_eps)
    """
    T, N = returns.shape
    cov_sample = np.cov(returns.T)
    
    eigvals, eigvecs = np.linalg.eigh(cov_sample)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 取前k个因子
    B = eigvecs[:, :n_factors]  # N x k
    factor_var = eigvals[:n_factors]  # k
    
    # 因子协方差
    cov_factor = B @ np.diag(factor_var) @ B.T
    
    # 残差方差
    residual_var = np.diag(cov_sample) - np.sum(B**2 * factor_var, axis=1)
    residual_var = np.maximum(residual_var, 1e-8)
    
    cov_est = cov_factor + np.diag(residual_var)
    cov_est = ensure_psd(cov_est)
    corr_est = cov_to_corr(cov_est)
    
    return corr_est, cov_est

# 尝试不同因子数
best_nmi = 0
best_k = 10
for k in [5, 10, 15, 20]:
    corr_f, cov_f = pca_factor_cov(ret_vals, n_factors=k)
    adj = topk_adj(corr_f, k=4)
    np.fill_diagonal(adj, 0)
    from eval_framework import eval_nmi
    nmi = eval_nmi(adj, stocks, code_to_industry, 35)
    print(f"  k={k}: NMI={nmi:.4f}")
    if nmi > best_nmi:
        best_nmi = nmi
        best_k = k

corr_factor, cov_factor = pca_factor_cov(ret_vals, n_factors=best_k)
r2 = evaluator.evaluate_static_method(f"PCA_Factor_k{best_k}", corr_factor, cov_factor)
all_results.append(r2)
print(f"  Best k={best_k}: NMI={r2['NMI']:.4f}, CovErr={r2['CovError']:.4f}, Sharpe={r2['Sharpe']:.4f}")


# ============================================================
# 3. 非线性收缩 (Analytical Nonlinear Shrinkage)
# ============================================================
print("\n[3/7] Nonlinear Shrinkage (Oracle Approximating) ...")

def nonlinear_shrinkage(returns):
    """
    Ledoit-Wolf 2017/2020 分析式非线性收缩
    简化实现: 对每个特征值应用最优收缩函数
    d_i^* = lambda_i / (1 + q * lambda_i * m(-lambda_i))
    其中m是Stieltjes变换
    """
    T, N = returns.shape
    q = N / T
    
    cov_sample = np.cov(returns.T)
    eigvals, eigvecs = np.linalg.eigh(cov_sample)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # 简化的非线性收缩: isotonic regression on eigvals
    # 使用 Stein-type 收缩公式
    lambda_sum = np.sum(eigvals)
    lambda_sq_sum = np.sum(eigvals**2)
    
    # Oracle shrinkage for each eigenvalue
    shrunk_eigvals = np.zeros_like(eigvals)
    for i in range(N):
        # 与其他特征值的距离
        diffs = eigvals - eigvals[i]
        diffs[i] = 1  # 避免除0
        
        # Stieltjes-like transform
        h = max(eigvals[i] * (N/T)**0.5 * 0.1, 1e-8)
        
        # 核密度估计
        kernel_vals = np.sum(h / (diffs**2 + h**2)) / (N * np.pi)
        
        # 收缩
        shrinkage_factor = 1.0 / (1 + q * kernel_vals * np.pi * eigvals[i])
        shrunk_eigvals[i] = eigvals[i] * max(shrinkage_factor, 0.01)
    
    # 确保非负
    shrunk_eigvals = np.maximum(shrunk_eigvals, 1e-8)
    
    cov_shrunk = eigvecs @ np.diag(shrunk_eigvals) @ eigvecs.T
    cov_shrunk = ensure_psd(cov_shrunk)
    corr_shrunk = cov_to_corr(cov_shrunk)
    
    return corr_shrunk, cov_shrunk

corr_nls, cov_nls = nonlinear_shrinkage(ret_vals)
r3 = evaluator.evaluate_static_method("NonlinearShrinkage", corr_nls, cov_nls)
all_results.append(r3)
print(f"  NMI={r3['NMI']:.4f}, CovErr={r3['CovError']:.4f}, Sharpe={r3['Sharpe']:.4f}, RankIC={r3['RankIC']:.4f}")


# ============================================================
# 4. RMT + Industry Prior (去噪后融合行业先验)
# ============================================================
print("\n[4/7] RMT + Industry Prior ...")

def rmt_with_prior(returns, prior, prior_weight=0.5):
    """RMT去噪 + 行业先验正则化"""
    corr_rmt, cov_rmt, n_sig = rmt_denoise(returns)
    
    # 行业先验协方差
    avg_var = np.diag(cov_rmt).mean()
    prior_cov = prior * avg_var * 0.5
    np.fill_diagonal(prior_cov, np.diag(cov_rmt))
    
    # 融合
    cov_combined = (1 - prior_weight) * cov_rmt + prior_weight * prior_cov
    cov_combined = ensure_psd(cov_combined)
    corr_combined = cov_to_corr(cov_combined)
    
    return corr_combined, cov_combined

# 尝试不同先验权重
best_composite = -1
best_pw = 0.3
for pw in [0.2, 0.3, 0.4, 0.5, 0.6]:
    corr_rp, cov_rp = rmt_with_prior(ret_vals, industry_prior, pw)
    adj = topk_adj(corr_rp, k=4)
    np.fill_diagonal(adj, 0)
    nmi = eval_nmi(adj, stocks, code_to_industry, 35)
    print(f"  pw={pw}: NMI={nmi:.4f}")

for pw in [0.3, 0.5, 0.7]:
    corr_rp, cov_rp = rmt_with_prior(ret_vals, industry_prior, pw)
    r = evaluator.evaluate_static_method(f"RMT+Prior_pw{pw}", corr_rp, cov_rp)
    all_results.append(r)
    print(f"  RMT+Prior pw={pw}: NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}")


# ============================================================
# 5. Factor + Industry Prior
# ============================================================
print("\n[5/7] Factor + Industry Prior ...")

def factor_with_prior(returns, prior, n_factors=10, prior_weight=0.5):
    """PCA因子模型 + 行业先验"""
    corr_f, cov_f = pca_factor_cov(returns, n_factors=n_factors)
    
    avg_var = np.diag(cov_f).mean()
    prior_cov = prior * avg_var * 0.5
    np.fill_diagonal(prior_cov, np.diag(cov_f))
    
    cov_combined = (1 - prior_weight) * cov_f + prior_weight * prior_cov
    cov_combined = ensure_psd(cov_combined)
    corr_combined = cov_to_corr(cov_combined)
    
    return corr_combined, cov_combined

for pw in [0.3, 0.5, 0.7]:
    corr_fp, cov_fp = factor_with_prior(ret_vals, industry_prior, n_factors=best_k, prior_weight=pw)
    r = evaluator.evaluate_static_method(f"Factor+Prior_k{best_k}_pw{pw}", corr_fp, cov_fp)
    all_results.append(r)
    print(f"  Factor+Prior k={best_k} pw={pw}: NMI={r['NMI']:.4f}, CovErr={r['CovError']:.4f}, Sharpe={r['Sharpe']:.4f}")


# ============================================================
# 6. 多尺度EWMA (融合多个半衰期)
# ============================================================
print("\n[6/7] Multi-scale EWMA ...")

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


class MultiScaleEWMA:
    """多尺度EWMA: 融合短期、中期、长期半衰期"""
    def __init__(self, n, half_lives=(42, 126, 252), weights=(0.2, 0.3, 0.5),
                 prior_weight=0.5, prior_matrix=None):
        self.estimators = [
            AdaptiveCovEstimator(n, hl, prior_weight, prior_matrix)
            for hl in half_lives
        ]
        self.weights = weights
        self.n = n
    
    def update(self, batch):
        for est in self.estimators:
            est.update(batch)
    
    def get_cov(self):
        covs = [est.get_cov() for est in self.estimators]
        return sum(w * c for w, c in zip(self.weights, covs))
    
    def get_corr(self):
        cov = self.get_cov()
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-10
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 1)
        return np.clip(corr, -1, 1)

r6 = evaluator.evaluate_dynamic_method(
    "MultiScale_EWMA_pw0.5",
    lambda: MultiScaleEWMA(n_stocks, half_lives=(42, 126, 252), weights=(0.2, 0.3, 0.5),
                           prior_weight=0.5, prior_matrix=industry_prior)
)
all_results.append(r6)
print(f"  NMI={r6['NMI']:.4f}, CovErr={r6['CovError']:.4f}, Sharpe={r6['Sharpe']:.4f}")

# 也试 pw=0.7
r6b = evaluator.evaluate_dynamic_method(
    "MultiScale_EWMA_pw0.7",
    lambda: MultiScaleEWMA(n_stocks, half_lives=(42, 126, 252), weights=(0.2, 0.3, 0.5),
                           prior_weight=0.7, prior_matrix=industry_prior)
)
all_results.append(r6b)
print(f"  pw=0.7: NMI={r6b['NMI']:.4f}, CovErr={r6b['CovError']:.4f}, Sharpe={r6b['Sharpe']:.4f}")


# ============================================================
# 7. RMT去噪 + 动态EWMA + Industry Prior
# ============================================================
print("\n[7/7] RMT Dynamic (EWMA + RMT denoise at each step) ...")

class RMTDynamicEstimator:
    """动态RMT: 在每次获取协方差时做RMT去噪"""
    def __init__(self, n, half_life=252, prior_weight=0.5, prior_matrix=None):
        self.n = n
        self.decay = np.log(2) / half_life
        self.prior_weight = prior_weight
        self.prior_matrix = prior_matrix
        self.cov_ewma = None
        self.count = 0
        self.all_data = []
    
    def update(self, batch):
        batch_cov = np.cov(batch.T)
        if self.cov_ewma is None:
            self.cov_ewma = batch_cov
        else:
            alpha = 1 - np.exp(-self.decay * len(batch))
            self.cov_ewma = (1 - alpha) * self.cov_ewma + alpha * batch_cov
        self.count += len(batch)
        self.all_data.extend(batch.tolist())
        # 保留最近500天
        if len(self.all_data) > 500:
            self.all_data = self.all_data[-500:]
    
    def _rmt_denoise_cov(self, cov):
        """对协方差矩阵做RMT去噪"""
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
        cov_d = corr_d * np.outer(std, std)
        return cov_d
    
    def get_cov(self):
        if self.cov_ewma is None:
            return np.eye(self.n) * 0.001
        
        cov = self.cov_ewma.copy()
        # 收缩
        shrinkage = min(0.8, max(0.0, self.n / max(self.count, 1)))
        target = np.diag(np.diag(cov))
        cov = (1 - shrinkage) * cov + shrinkage * target
        
        # RMT去噪
        cov = self._rmt_denoise_cov(cov)
        
        # 行业先验
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

r7 = evaluator.evaluate_dynamic_method(
    "RMT_Dynamic_pw0.5",
    lambda: RMTDynamicEstimator(n_stocks, half_life=252, prior_weight=0.5, prior_matrix=industry_prior)
)
all_results.append(r7)
print(f"  NMI={r7['NMI']:.4f}, CovErr={r7['CovError']:.4f}, Sharpe={r7['Sharpe']:.4f}")

r7b = evaluator.evaluate_dynamic_method(
    "RMT_Dynamic_pw0.7",
    lambda: RMTDynamicEstimator(n_stocks, half_life=252, prior_weight=0.7, prior_matrix=industry_prior)
)
all_results.append(r7b)
print(f"  pw=0.7: NMI={r7b['NMI']:.4f}, CovErr={r7b['CovError']:.4f}, Sharpe={r7b['Sharpe']:.4f}")


# ============================================================
# 汇总结果
# ============================================================
print("\n" + "=" * 70)
print("迭代2: 新模型架构 - 结果汇总")
print("=" * 70)

# 加入迭代1的最佳结果作为参考
iter1_best = [
    {'method': '[REF] StaticPearson_K4', 'NMI': 0.7648, 'ARI': 0.3797,
     'Modularity': 0.5470, 'IC': 0.6061, 'CovError': 0.2970,
     'LogLik': 772.7, 'RankIC': 0.8829, 'Sharpe': 0.9258, 'Sortino': 1.2330,
     'MaxDD': 0.1179, 'Calmar': 0.9864, 'NMI_Std': 0.0, 'CovErr_Std': 0.0},
    {'method': '[REF] Adaptive_pw0.7_K4', 'NMI': 0.8995, 'ARI': 0.6445,
     'Modularity': 0.7828, 'IC': 0.8470, 'CovError': 0.7814,
     'LogLik': 693.8, 'RankIC': 0.5240, 'Sharpe': 0.7540, 'Sortino': 1.0059,
     'MaxDD': 0.1450, 'Calmar': 0.7249, 'NMI_Std': 0.0167, 'CovErr_Std': 0.1501},
]

display_results = iter1_best + all_results
df = format_results_table(display_results, sort_by='NMI', ascending=False)
df['CompositeScore'] = compute_composite_score(df)
df = df.sort_values('CompositeScore', ascending=False).reset_index(drop=True)
df.index = df.index + 1
df.index.name = 'Rank'

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.4f}'.format)
print(df.to_string())

# 保存
df.to_csv(f'{OUT_DIR}/comprehensive_eval.csv')
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

# 找出新方法的top结果
new_methods = [r for r in all_results if not r['method'].startswith('[REF]')]
new_df = format_results_table(new_methods, sort_by='NMI', ascending=False)
new_df['CompositeScore'] = compute_composite_score(new_df)
print("\n--- 新方法Top 5 (by CompositeScore) ---")
top5 = new_df.sort_values('CompositeScore', ascending=False).head(5)
for _, row in top5.iterrows():
    print(f"  {row['method']}: NMI={row['NMI']:.4f}, Sharpe={row['Sharpe']:.4f}, Composite={row['CompositeScore']:.4f}")

print(f"\n结果已保存到 {OUT_DIR}/")
print("\n迭代2完成!")
