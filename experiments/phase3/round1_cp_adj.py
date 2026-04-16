"""
Round 1: cp优化 + 加权邻接矩阵 + 互选KNN
目标: 
  1. 在DualPath框架下搜索最优cp (0.3-0.9)
  2. 测试加权TopK (用相关系数值作为边权)
  3. 测试互选KNN (两个节点互相选中才连边)
"""
import sys
sys.path.insert(0, '/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation')

import pandas as pd
import numpy as np
from eval_framework_v2 import (
    WalkForwardEvaluator, build_industry_prior, cov_to_corr, ensure_psd,
    format_results_table, compute_composite_score, topk_adj, eval_nmi,
    eval_ari, eval_modularity, eval_ic, eval_cov_error, eval_log_likelihood,
    eval_rank_ic, eval_portfolio_metrics, min_var_weights, _get_labels
)
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering
import json, os, time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/round1"
os.makedirs(OUT_DIR, exist_ok=True)

returns_df = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns_df.columns.tolist()
n_stocks = len(stocks)
ret_vals = returns_df.values
industry_prior = build_industry_prior(stocks, code_to_industry)

print(f"股票数: {n_stocks}, 交易日: {len(ret_vals)}")

# ============================================================
# 基座函数 (复用)
# ============================================================

def rmt_denoise_fn(history):
    T, N = history.shape
    q = N / T
    cov = np.cov(history.T)
    std = np.sqrt(np.diag(cov)); std[std==0] = 1e-10
    corr = cov / np.outer(std, std); np.fill_diagonal(corr, 1)
    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    lam_plus = (1 + np.sqrt(q))**2
    n_sig = max(np.sum(eigvals > lam_plus), 1)
    noise_mean = np.mean(eigvals[n_sig:]) if n_sig < N else 1.0
    d = eigvals.copy(); d[n_sig:] = noise_mean
    corr_d = eigvecs @ np.diag(d) @ eigvecs.T; np.fill_diagonal(corr_d, 1)
    cov_d = corr_d * np.outer(std, std)
    return cov_to_corr(ensure_psd(cov_d)), ensure_psd(cov_d)

def pca_factor_fn(history, nf=20):
    cov_s = np.cov(history.T)
    eigvals, eigvecs = np.linalg.eigh(cov_s)
    idx = np.argsort(eigvals)[::-1]; eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    B = eigvecs[:, :nf]; fv = eigvals[:nf]
    cov_f = B @ np.diag(fv) @ B.T
    rv = np.maximum(np.diag(cov_s) - np.sum(B**2 * fv, axis=1), 1e-8)
    c = ensure_psd(cov_f + np.diag(rv))
    return cov_to_corr(c), c

# ============================================================
# 新的邻接矩阵构建方法
# ============================================================

def weighted_topk_adj(corr, k=3):
    """加权TopK: 保留相关系数值作为边权"""
    n = corr.shape[0]
    adj = np.zeros_like(corr)
    for i in range(n):
        row = np.abs(corr[i].copy())
        row[i] = -np.inf
        top = np.argsort(row)[-k:]
        for j in top:
            adj[i, j] = np.abs(corr[i, j])
    # 对称化: 取较大值
    adj = np.maximum(adj, adj.T)
    return adj

def mutual_knn_adj(corr, k=5):
    """互选KNN: 两个节点互相选中才连边"""
    n = corr.shape[0]
    sel = np.zeros((n, n), dtype=bool)
    for i in range(n):
        row = np.abs(corr[i].copy())
        row[i] = -np.inf
        top = np.argsort(row)[-k:]
        sel[i, top] = True
    # 互选: 两者都选中
    mutual = (sel & sel.T).astype(float)
    return mutual

# ============================================================
# 自定义评估器 (支持不同邻接方法)
# ============================================================

class FlexibleEvaluator:
    """支持自定义邻接矩阵构建的评估器"""
    
    def __init__(self, returns, stocks, code_to_industry, industry_prior,
                 warmup=500, eval_freq=60, forecast=60, rebalance=60,
                 n_clusters=35, adj_fn=None):
        self.returns = returns
        self.stocks = stocks
        self.code_to_industry = code_to_industry
        self.industry_prior = industry_prior
        self.n_stocks = len(stocks)
        self.warmup = warmup
        self.eval_freq = eval_freq
        self.forecast = forecast
        self.rebalance = rebalance
        self.n_clusters = n_clusters
        self.adj_fn = adj_fn or (lambda c: topk_adj(c, k=3))
        self.eval_points = list(range(warmup, len(returns) - forecast, eval_freq))
    
    def evaluate(self, name, estimator_fn):
        results = {'method': name}
        nmis, aris, mods, ics = [], [], [], []
        cov_errors, log_liks, rank_ics = [], [], []
        
        for t in self.eval_points:
            history = self.returns[:t]
            corr_est, cov_est = estimator_fn(history)
            
            adj = self.adj_fn(corr_est)
            np.fill_diagonal(adj, 0)
            
            # NMI/ARI用谱聚类
            try:
                abs_adj = np.abs(adj).astype(float)
                np.fill_diagonal(abs_adj, abs_adj.max() if abs_adj.max() > 0 else 1.0)
                sc = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed',
                                       random_state=42, n_init=3)
                pred = sc.fit_predict(abs_adj)
                true_labels, _ = _get_labels(self.stocks, self.code_to_industry)
                nmis.append(normalized_mutual_info_score(true_labels, pred))
                aris.append(adjusted_rand_score(true_labels, pred))
            except:
                nmis.append(0.0)
                aris.append(0.0)
            
            if t == self.eval_points[-1]:
                mods.append(eval_modularity(adj, self.stocks, self.code_to_industry))
                ics.append(eval_ic(adj, self.stocks, self.code_to_industry))
            
            future = self.returns[t:t + self.forecast]
            if len(future) >= self.forecast:
                cov_true = np.cov(future.T)
                cov_errors.append(eval_cov_error(cov_est, cov_true))
                log_liks.append(eval_log_likelihood(cov_est, future))
                rank_ics.append(eval_rank_ic(cov_est, cov_true))
        
        results['NMI'] = np.mean(nmis) if nmis else 0.0
        results['ARI'] = np.mean(aris) if aris else 0.0
        results['Modularity'] = np.mean(mods) if mods else 0.0
        results['IC'] = np.mean(ics) if ics else 0.0
        results['NMI_Std'] = np.std(nmis) if len(nmis) > 1 else 0.0
        results['CovError'] = np.mean(cov_errors) if cov_errors else np.nan
        results['CovErr_Std'] = np.std(cov_errors) if len(cov_errors) > 1 else 0.0
        results['LogLik'] = np.mean(log_liks) if log_liks else np.nan
        results['RankIC'] = np.mean(rank_ics) if rank_ics else np.nan
        
        portfolio_rets = []
        for t in range(self.warmup, len(self.returns) - self.rebalance, self.rebalance):
            history = self.returns[:t]
            _, cov_est = estimator_fn(history)
            cov_reg = ensure_psd(cov_est) + np.eye(self.n_stocks) * 1e-6
            w = min_var_weights(cov_reg)
            period_rets = self.returns[t:t + self.rebalance] @ w
            portfolio_rets.extend(period_rets.tolist())
        
        pm = eval_portfolio_metrics(portfolio_rets)
        results.update({
            'Sharpe': pm['sharpe'], 'Sortino': pm['sortino'],
            'MaxDD': pm['max_drawdown'], 'Calmar': pm['calmar'],
        })
        return results


# ============================================================
# DualPath生成器
# ============================================================

def make_dual_path_fn(base_fn, prior, cp=0.6, pp=0.0):
    def fn(history):
        corr_b, cov_b = base_fn(history)
        corr_c = (1 - cp) * corr_b + cp * prior
        np.fill_diagonal(corr_c, 1)
        return corr_c, cov_b
    return fn


# ============================================================
# 实验开始
# ============================================================
all_results = []
t0 = time.time()

def run_eval(evaluator, name, fn):
    t1 = time.time()
    r = evaluator.evaluate(name, fn)
    elapsed = time.time() - t1
    print(f"  [{len(all_results)+1:2d}] {name:45s} NMI={r['NMI']:.4f} Sharpe={r['Sharpe']:.4f} ({elapsed:.0f}s)")
    all_results.append(r)
    return r


# ===== 实验1: cp参数搜索 (RMT base, K=3) =====
print("\n" + "="*70)
print("实验1: DualPath cp参数搜索 (RMT base, K=3)")
print("="*70)

ev_k3 = FlexibleEvaluator(
    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
    industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
    rebalance=60, n_clusters=35, adj_fn=lambda c: topk_adj(c, k=3)
)

for cp in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    run_eval(ev_k3, f"DP_RMT_cp{cp}_K3",
             make_dual_path_fn(rmt_denoise_fn, industry_prior, cp=cp))


# ===== 实验2: cp参数搜索 (Factor base, K=3) =====
print("\n" + "="*70)
print("实验2: DualPath cp参数搜索 (Factor base, K=3)")
print("="*70)

for cp in [0.4, 0.5, 0.7, 0.8]:
    run_eval(ev_k3, f"DP_Factor_cp{cp}_K3",
             make_dual_path_fn(pca_factor_fn, industry_prior, cp=cp))


# ===== 实验3: 加权TopK (RMT, cp=0.6) =====
print("\n" + "="*70)
print("实验3: 加权TopK vs 二值TopK")
print("="*70)

for k in [3, 4, 5]:
    ev_wk = FlexibleEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
        rebalance=60, n_clusters=35, adj_fn=lambda c, kk=k: weighted_topk_adj(c, k=kk)
    )
    run_eval(ev_wk, f"DP_RMT_cp0.6_WeightedK{k}",
             make_dual_path_fn(rmt_denoise_fn, industry_prior, cp=0.6))


# ===== 实验4: 互选KNN =====
print("\n" + "="*70)
print("实验4: 互选KNN")
print("="*70)

for k in [4, 5, 6, 8]:
    ev_mk = FlexibleEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
        rebalance=60, n_clusters=35, adj_fn=lambda c, kk=k: mutual_knn_adj(c, k=kk)
    )
    run_eval(ev_mk, f"DP_RMT_cp0.6_MutualK{k}",
             make_dual_path_fn(rmt_denoise_fn, industry_prior, cp=0.6))


# ===== 实验5: 最优cp + 最优K组合 =====
print("\n" + "="*70)
print("实验5: 最优cp + K=2精细搜索")
print("="*70)

ev_k2 = FlexibleEvaluator(
    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
    industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
    rebalance=60, n_clusters=35, adj_fn=lambda c: topk_adj(c, k=2)
)

for cp in [0.6, 0.7, 0.8]:
    run_eval(ev_k2, f"DP_RMT_cp{cp}_K2",
             make_dual_path_fn(rmt_denoise_fn, industry_prior, cp=cp))
    run_eval(ev_k2, f"DP_Factor_cp{cp}_K2",
             make_dual_path_fn(pca_factor_fn, industry_prior, cp=cp))


# ============================================================
# 汇总
# ============================================================
elapsed_total = time.time() - t0
print(f"\n总耗时: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

df = format_results_table(all_results, sort_by='NMI', ascending=False)
df['CompositeScore'] = compute_composite_score(df)
df = df.sort_values('CompositeScore', ascending=False).reset_index(drop=True)
df.index = df.index + 1
df.index.name = 'Rank'

print("\n" + "="*70)
print("Round 1 Results")
print("="*70)
print(df[['method','NMI','ARI','Modularity','IC','CovError','Sharpe','Sortino','MaxDD','CompositeScore']].to_string())

df.to_csv(f'{OUT_DIR}/round1_results.csv')
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

# 找到最佳结果
best = df.iloc[0]
print(f"\n最佳: {best['method']}, Composite={best['CompositeScore']:.4f}, NMI={best['NMI']:.4f}, Sharpe={best['Sharpe']:.4f}")
print(f"\n对比基准 DP_RMT_cp0.6_K3: Composite=0.922, NMI=0.899, Sharpe=0.979")
print("\nRound 1 完成!")
