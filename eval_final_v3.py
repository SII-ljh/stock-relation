"""
最终全量评估 V3: 旧方法 + 新方法统一在V2 Walk-Forward框架下评估

新增方法 (来自Round 1-2):
  - WeightedTopK系列 (加权邻接矩阵, 本轮核心发现)
  - 最佳新模型:
    1. DP_RMT_cp0.8_WK5 (NMI冠军)
    2. DP_Factor_k10_WK5 (Sharpe冠军)  
    3. DP_RMT_cp0.7_WK5 (平衡)
    4. DP_RMT_cp0.6_WK4 (稳健)
    5. DP_Factor_WK5 (Factor+加权)
    6. Ensemble_RMT_POET15_WK5 (集成)
"""
import sys
sys.path.insert(0, '/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation')

import pandas as pd
import numpy as np
from eval_framework_v2 import (
    WalkForwardEvaluator, build_industry_prior, cov_to_corr, ensure_psd,
    format_results_table, compute_composite_score, topk_adj,
    eval_nmi, eval_ari, eval_modularity, eval_ic, eval_cov_error,
    eval_log_likelihood, eval_rank_ic, eval_portfolio_metrics,
    min_var_weights, _get_labels
)
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering
from sklearn.covariance import LedoitWolf, GraphicalLassoCV
import json, os, time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/eval_final_v3"
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
# 基座函数
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

def poet_fn(history, n_factors=15, tc=0.5):
    T, N = history.shape
    cov_s = np.cov(history.T)
    eigvals, eigvecs = np.linalg.eigh(cov_s)
    idx = np.argsort(eigvals)[::-1]; eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    B = eigvecs[:, :n_factors]; fv = eigvals[:n_factors]
    cov_f = B @ np.diag(fv) @ B.T
    res = cov_s - cov_f
    thr = tc * np.sqrt(np.log(N) / T)
    rt = res.copy()
    for i in range(N):
        for j in range(N):
            if i != j:
                scale = np.sqrt(abs(res[i,i]*res[j,j]))
                rt[i,j] = np.sign(res[i,j]) * max(abs(res[i,j]) - thr*scale, 0)
    c = ensure_psd(cov_f + rt)
    return cov_to_corr(c), c

def nonlinear_shrinkage_fn(history):
    T, N = history.shape
    cov_s = np.cov(history.T)
    eigvals, eigvecs = np.linalg.eigh(cov_s)
    idx = np.argsort(eigvals)[::-1]; eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    shrunk = np.zeros_like(eigvals)
    for i in range(N):
        diffs = eigvals - eigvals[i]; diffs[i] = 1
        h = max(eigvals[i] * (N/T)**0.5 * 0.1, 1e-8)
        kv = np.sum(h / (diffs**2 + h**2)) / (N * np.pi)
        sf = 1.0 / (1 + (N/T) * kv * np.pi * eigvals[i])
        shrunk[i] = eigvals[i] * max(sf, 0.01)
    shrunk = np.maximum(shrunk, 1e-8)
    c = ensure_psd(eigvecs @ np.diag(shrunk) @ eigvecs.T)
    return cov_to_corr(c), c

def adaptive_ewma_fn(history, hl=252, pw=0.7):
    T, N = history.shape
    decay = np.log(2) / hl; step = 20
    cov_ewma = np.cov(history[:step].T); count = step
    for t in range(step, T, step):
        batch = history[t:t+step]
        if len(batch) < 5: continue
        bc = np.cov(batch.T)
        alpha = 1 - np.exp(-decay * len(batch))
        cov_ewma = (1 - alpha) * cov_ewma + alpha * bc
        count += len(batch)
    shrinkage = min(0.8, max(0.0, N / max(count, 1)))
    target = np.diag(np.diag(cov_ewma))
    cov = (1 - shrinkage) * cov_ewma + shrinkage * target
    if pw > 0:
        avg_var = np.diag(cov).mean()
        pc = industry_prior * avg_var * 0.5
        np.fill_diagonal(pc, 0)
        cov = (1 - pw) * cov + pw * (cov + pc)
    cov = ensure_psd(cov)
    return cov_to_corr(cov), cov

# ============================================================
# 新方法: 加权TopK
# ============================================================

def weighted_topk_adj(corr, k=5):
    n = corr.shape[0]
    adj = np.zeros_like(corr)
    for i in range(n):
        row = np.abs(corr[i].copy())
        row[i] = -np.inf
        top = np.argsort(row)[-k:]
        for j in top:
            adj[i, j] = np.abs(corr[i, j])
    return np.maximum(adj, adj.T)

def make_dual_path_fn(base_fn, prior, cp=0.6, pp=0.0):
    def fn(history):
        corr_b, cov_b = base_fn(history)
        corr_c = (1 - cp) * corr_b + cp * prior
        np.fill_diagonal(corr_c, 1)
        if pp > 0:
            avg_var = np.diag(cov_b).mean()
            pc = prior * avg_var * 0.5
            np.fill_diagonal(pc, np.diag(cov_b))
            cov_o = (1 - pp) * cov_b + pp * pc
            cov_o = ensure_psd(cov_o)
        else:
            cov_o = cov_b
        return corr_c, cov_o
    return fn

def make_prior_fn(base_fn, prior, pw):
    def fn(history):
        corr_b, cov_b = base_fn(history)
        avg_var = np.diag(cov_b).mean()
        pc = prior * avg_var * 0.5
        np.fill_diagonal(pc, np.diag(cov_b))
        co = (1 - pw) * cov_b + pw * pc
        co = ensure_psd(co)
        return cov_to_corr(co), co
    return fn

def make_ensemble_dp_fn(base_fns, prior, cp=0.6):
    def fn(history):
        corrs, covs = [], []
        for bf in base_fns:
            c, cv = bf(history)
            corrs.append(c); covs.append(cv)
        corr_avg = np.mean(corrs, axis=0)
        corr_c = (1 - cp) * corr_avg + cp * prior
        np.fill_diagonal(corr_c, 1)
        cov_avg = ensure_psd(np.mean(covs, axis=0))
        return corr_c, cov_avg
    return fn

# ============================================================
# 自定义评估器
# ============================================================

class FlexibleEvaluator:
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
            try:
                abs_adj = np.abs(adj).astype(float)
                diag_val = abs_adj.max() if abs_adj.max() > 0 else 1.0
                np.fill_diagonal(abs_adj, diag_val)
                sc = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed',
                                       random_state=42, n_init=3)
                pred = sc.fit_predict(abs_adj)
                true_labels, _ = _get_labels(self.stocks, self.code_to_industry)
                nmis.append(normalized_mutual_info_score(true_labels, pred))
                aris.append(adjusted_rand_score(true_labels, pred))
            except:
                nmis.append(0.0); aris.append(0.0)
            
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
# 评估所有方法
# ============================================================
all_results = []
t0 = time.time()

def run(ev, name, fn):
    t1 = time.time()
    r = ev.evaluate(name, fn)
    elapsed = time.time() - t1
    print(f"  [{len(all_results)+1:2d}] {name:50s} NMI={r['NMI']:.4f} Sharpe={r['Sharpe']:.4f} ({elapsed:.0f}s)")
    all_results.append(r)

# 评估器: 二值TopK
ev_k3 = FlexibleEvaluator(ret_vals, stocks, code_to_industry, industry_prior,
                           n_clusters=35, adj_fn=lambda c: topk_adj(c, k=3))
ev_k4 = FlexibleEvaluator(ret_vals, stocks, code_to_industry, industry_prior,
                           n_clusters=35, adj_fn=lambda c: topk_adj(c, k=4))

# 评估器: 加权TopK
ev_wk4 = FlexibleEvaluator(ret_vals, stocks, code_to_industry, industry_prior,
                            n_clusters=35, adj_fn=lambda c: weighted_topk_adj(c, k=4))
ev_wk5 = FlexibleEvaluator(ret_vals, stocks, code_to_industry, industry_prior,
                            n_clusters=35, adj_fn=lambda c: weighted_topk_adj(c, k=5))

# === A. 基线 ===
print("\n=== A. 基线 ===")
run(ev_k4, "SamplePearson", lambda h: (cov_to_corr(np.cov(h.T)), np.cov(h.T)))
run(ev_k4, "LedoitWolf", lambda h: (cov_to_corr(LedoitWolf().fit(h).covariance_), LedoitWolf().fit(h).covariance_))

eq_rets = ret_vals[500:].mean(axis=1)
pm_eq = eval_portfolio_metrics(eq_rets.tolist())
r_eq = {'method': 'EqualWeight', 'NMI': np.nan, 'ARI': np.nan, 'Modularity': np.nan, 'IC': np.nan,
        'CovError': np.nan, 'LogLik': np.nan, 'RankIC': np.nan,
        'Sharpe': pm_eq['sharpe'], 'Sortino': pm_eq['sortino'],
        'MaxDD': pm_eq['max_drawdown'], 'Calmar': pm_eq['calmar'],
        'NMI_Std': np.nan, 'CovErr_Std': np.nan}
all_results.append(r_eq)
print(f"  [ {len(all_results):2d}] {'EqualWeight':50s} Sharpe={r_eq['Sharpe']:.4f}")

# === B. 去噪 ===
print("\n=== B. 去噪 ===")
run(ev_k4, "RMT_Denoise", rmt_denoise_fn)
run(ev_k4, "POET15", lambda h: poet_fn(h, 15))
run(ev_k4, "PCA_Factor_k20", pca_factor_fn)
run(ev_k4, "NonlinearShrinkage", nonlinear_shrinkage_fn)

# === C. 单路径+先验 ===
print("\n=== C. 单路径+先验 ===")
run(ev_k4, "RMT+Prior_pw0.3", make_prior_fn(rmt_denoise_fn, industry_prior, 0.3))
run(ev_k4, "RMT+Prior_pw0.5", make_prior_fn(rmt_denoise_fn, industry_prior, 0.5))
run(ev_k4, "Adaptive_EWMA_pw0.7", lambda h: adaptive_ewma_fn(h, 252, 0.7))

# === D. DualPath 二值TopK (旧方法) ===
print("\n=== D. DualPath 二值TopK ===")
run(ev_k3, "DP_RMT_cp0.6_K3", make_dual_path_fn(rmt_denoise_fn, industry_prior, 0.6))
run(ev_k4, "DP_RMT_cp0.6_K4", make_dual_path_fn(rmt_denoise_fn, industry_prior, 0.6))
run(ev_k3, "DP_Factor_cp0.6_K3", make_dual_path_fn(pca_factor_fn, industry_prior, 0.6))
run(ev_k4, "DP_Factor_cp0.6_K4", make_dual_path_fn(pca_factor_fn, industry_prior, 0.6))
run(ev_k3, "DP_POET15_cp0.6_K3", make_dual_path_fn(lambda h: poet_fn(h,15), industry_prior, 0.6))

# === E. DualPath 加权TopK (新方法, 本轮核心) ===
print("\n=== E. DualPath 加权TopK (NEW) ===")

# E1. RMT base + WeightedK
run(ev_wk4, "DP_RMT_cp0.6_WK4", make_dual_path_fn(rmt_denoise_fn, industry_prior, 0.6))
run(ev_wk5, "DP_RMT_cp0.6_WK5", make_dual_path_fn(rmt_denoise_fn, industry_prior, 0.6))
run(ev_wk5, "DP_RMT_cp0.7_WK5", make_dual_path_fn(rmt_denoise_fn, industry_prior, 0.7))
run(ev_wk5, "DP_RMT_cp0.8_WK5", make_dual_path_fn(rmt_denoise_fn, industry_prior, 0.8))

# E2. Factor base + WeightedK
run(ev_wk5, "DP_Factor_cp0.6_WK5", make_dual_path_fn(pca_factor_fn, industry_prior, 0.6))
run(ev_wk5, "DP_Factor_k10_cp0.6_WK5",
    make_dual_path_fn(lambda h: pca_factor_fn(h, 10), industry_prior, 0.6))

# E3. POET15 base + WeightedK
run(ev_wk5, "DP_POET15_cp0.6_WK5",
    make_dual_path_fn(lambda h: poet_fn(h,15), industry_prior, 0.6))

# E4. Ensemble
run(ev_wk5, "Ensemble_RMT_POET15_cp0.6_WK5",
    make_ensemble_dp_fn([rmt_denoise_fn, lambda h: poet_fn(h,15)], industry_prior, 0.6))


# ============================================================
# 汇总排名
# ============================================================
elapsed_total = time.time() - t0
print(f"\n总耗时: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

df = format_results_table(all_results, sort_by='NMI', ascending=False)
df['CompositeScore'] = compute_composite_score(df)
df = df.sort_values('CompositeScore', ascending=False).reset_index(drop=True)
df.index = df.index + 1
df.index.name = 'Rank'

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', '{:.4f}'.format)

print("\n" + "="*70)
print("最终全量排名 (V3 Walk-Forward)")
print("="*70)
print(df[['method','NMI','ARI','Modularity','IC','CovError','RankIC','Sharpe','Sortino','MaxDD','Calmar','NMI_Std','CompositeScore']].to_string())

# 保存
df.to_csv(f'{OUT_DIR}/final_ranking_v3.csv')
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

# Markdown表格
print("\n\n--- 性能排名表 (Markdown) ---")
print("| Rank | Method | NMI | ARI | Modul. | IC | CovErr | RankIC | Sharpe | Sortino | MaxDD | Calmar | Composite |")
print("|:----:|--------|:---:|:---:|:------:|:--:|:------:|:------:|:------:|:-------:|:-----:|:------:|:---------:|")
for idx, row in df.iterrows():
    def fmt(v, f='.4f'):
        return f"{v:{f}}" if not (isinstance(v, float) and np.isnan(v)) else "—"
    print(f"| {idx} | {row['method']} | {fmt(row['NMI'])} | {fmt(row['ARI'])} | {fmt(row['Modularity'])} | {fmt(row['IC'])} | {fmt(row['CovError'])} | {fmt(row['RankIC'])} | {fmt(row['Sharpe'])} | {fmt(row['Sortino'])} | {fmt(row['MaxDD'])} | {fmt(row['Calmar'])} | {fmt(row['CompositeScore'])} |")

print(f"\n结果已保存到 {OUT_DIR}/")
print("全量评估完成!")
