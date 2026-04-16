"""
Round 2: 加权TopK全面优化 + 集成方法 + 最终全量评估

Round 1发现: 加权TopK(用相关系数值作边权)大幅提升NMI
  - WeightedK5: NMI=0.9342 (vs 二值K3: 0.899, +3.9%)
  - WeightedK4: NMI=0.9304

本轮目标:
  1. WeightedTopK + 不同base (RMT, Factor, POET15)
  2. WeightedTopK + 不同cp值
  3. WeightedTopK + K=4,5,6,7精细搜索
  4. 集成方法: 平均多个base的相关矩阵
  5. 软先验: 用数据驱动的行业内平均相关作为先验强度
  6. 全量评估: 新方法 + 旧方法全部在V2框架下评估
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
from sklearn.covariance import LedoitWolf
import json, os, time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/round2"
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

# ============================================================
# 新方法
# ============================================================

def weighted_topk_adj(corr, k=5):
    """加权TopK: 保留相关系数值作为边权"""
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

def make_ensemble_dp_fn(base_fns, prior, cp=0.6):
    """集成多个base的DualPath: 平均协方差"""
    def fn(history):
        corrs = []
        covs = []
        for bf in base_fns:
            c, cv = bf(history)
            corrs.append(c)
            covs.append(cv)
        # 平均相关矩阵
        corr_avg = np.mean(corrs, axis=0)
        # 聚类路径: 融合先验
        corr_c = (1 - cp) * corr_avg + cp * prior
        np.fill_diagonal(corr_c, 1)
        # 投资组合路径: 平均协方差(不加先验)
        cov_avg = np.mean(covs, axis=0)
        cov_avg = ensure_psd(cov_avg)
        return corr_c, cov_avg
    return fn

def make_soft_prior_dp_fn(base_fn, stocks, code_to_industry, cp=0.6):
    """软先验DualPath: 用数据驱动的行业内相关作为先验强度"""
    def fn(history):
        corr_b, cov_b = base_fn(history)
        # 构建软先验: 行业内用平均相关,行业间用0
        n = len(stocks)
        soft_prior = np.zeros((n, n))
        ind_groups = {}
        for i, s in enumerate(stocks):
            ind = code_to_industry.get(s, 'Unknown')
            if ind not in ind_groups:
                ind_groups[ind] = []
            ind_groups[ind].append(i)
        
        for ind, members in ind_groups.items():
            if len(members) < 2:
                continue
            # 计算行业内平均相关
            intra_corrs = []
            for ii in range(len(members)):
                for jj in range(ii+1, len(members)):
                    intra_corrs.append(corr_b[members[ii], members[jj]])
            avg_intra = np.mean(intra_corrs) if intra_corrs else 0.5
            # 赋值
            for ii in members:
                for jj in members:
                    if ii != jj:
                        soft_prior[ii, jj] = max(avg_intra, 0.1)
        
        corr_c = (1 - cp) * corr_b + cp * soft_prior
        np.fill_diagonal(corr_c, 1)
        return corr_c, cov_b
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
# 运行实验
# ============================================================
all_results = []
t0 = time.time()

def run_eval(evaluator, name, fn):
    t1 = time.time()
    r = evaluator.evaluate(name, fn)
    elapsed = time.time() - t1
    print(f"  [{len(all_results)+1:2d}] {name:50s} NMI={r['NMI']:.4f} Sharpe={r['Sharpe']:.4f} ({elapsed:.0f}s)")
    all_results.append(r)
    return r


# ===== 实验1: WeightedTopK + RMT, 精细K搜索 =====
print("\n" + "="*70)
print("实验1: WeightedTopK + RMT base, K=3-8 (cp=0.6)")
print("="*70)

for k in [3, 4, 5, 6, 7, 8]:
    ev = FlexibleEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
        rebalance=60, n_clusters=35, adj_fn=lambda c, kk=k: weighted_topk_adj(c, k=kk)
    )
    run_eval(ev, f"DP_RMT_cp0.6_WK{k}",
             make_dual_path_fn(rmt_denoise_fn, industry_prior, cp=0.6))


# ===== 实验2: WeightedTopK + Factor base =====
print("\n" + "="*70)
print("实验2: WeightedTopK + Factor base (cp=0.6)")
print("="*70)

for k in [4, 5, 6]:
    ev = FlexibleEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
        rebalance=60, n_clusters=35, adj_fn=lambda c, kk=k: weighted_topk_adj(c, k=kk)
    )
    run_eval(ev, f"DP_Factor_cp0.6_WK{k}",
             make_dual_path_fn(pca_factor_fn, industry_prior, cp=0.6))


# ===== 实验3: WeightedTopK + POET15 base =====
print("\n" + "="*70)
print("实验3: WeightedTopK + POET15 base (cp=0.6)")
print("="*70)

for k in [4, 5]:
    ev = FlexibleEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
        rebalance=60, n_clusters=35, adj_fn=lambda c, kk=k: weighted_topk_adj(c, k=kk)
    )
    run_eval(ev, f"DP_POET15_cp0.6_WK{k}",
             make_dual_path_fn(lambda h: poet_fn(h, 15), industry_prior, cp=0.6))


# ===== 实验4: WeightedTopK + cp搜索 (RMT, K=5) =====
print("\n" + "="*70)
print("实验4: WeightedK5 + cp搜索 (RMT base)")
print("="*70)

ev_wk5 = FlexibleEvaluator(
    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
    industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
    rebalance=60, n_clusters=35, adj_fn=lambda c: weighted_topk_adj(c, k=5)
)

for cp in [0.3, 0.4, 0.5, 0.7, 0.8]:
    run_eval(ev_wk5, f"DP_RMT_cp{cp}_WK5",
             make_dual_path_fn(rmt_denoise_fn, industry_prior, cp=cp))


# ===== 实验5: 集成DualPath (RMT+Factor平均) =====
print("\n" + "="*70)
print("实验5: 集成DualPath (RMT+Factor)")
print("="*70)

for k in [4, 5]:
    ev = FlexibleEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
        rebalance=60, n_clusters=35, adj_fn=lambda c, kk=k: weighted_topk_adj(c, k=kk)
    )
    run_eval(ev, f"Ensemble_RMT_Factor_cp0.6_WK{k}",
             make_ensemble_dp_fn([rmt_denoise_fn, pca_factor_fn], industry_prior, cp=0.6))

# RMT+POET15集成
for k in [4, 5]:
    ev = FlexibleEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
        rebalance=60, n_clusters=35, adj_fn=lambda c, kk=k: weighted_topk_adj(c, k=kk)
    )
    run_eval(ev, f"Ensemble_RMT_POET15_cp0.6_WK{k}",
             make_ensemble_dp_fn([rmt_denoise_fn, lambda h: poet_fn(h,15)], industry_prior, cp=0.6))

# 三基座集成
ev_wk5 = FlexibleEvaluator(
    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
    industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
    rebalance=60, n_clusters=35, adj_fn=lambda c: weighted_topk_adj(c, k=5)
)
run_eval(ev_wk5, "Ensemble_3Base_cp0.6_WK5",
         make_ensemble_dp_fn([rmt_denoise_fn, pca_factor_fn, lambda h: poet_fn(h,15)],
                            industry_prior, cp=0.6))


# ===== 实验6: 软先验DualPath =====
print("\n" + "="*70)
print("实验6: 软先验DualPath (数据驱动行业内相关)")
print("="*70)

for k in [4, 5]:
    ev = FlexibleEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
        rebalance=60, n_clusters=35, adj_fn=lambda c, kk=k: weighted_topk_adj(c, k=kk)
    )
    run_eval(ev, f"SoftPrior_RMT_cp0.6_WK{k}",
             make_soft_prior_dp_fn(rmt_denoise_fn, stocks, code_to_industry, cp=0.6))


# ===== 实验7: n_clusters搜索 (WeightedK5, RMT) =====
print("\n" + "="*70)
print("实验7: n_clusters搜索 (DP_RMT_cp0.6_WK5)")
print("="*70)

dp_rmt_wk5_fn = make_dual_path_fn(rmt_denoise_fn, industry_prior, cp=0.6)
for nc in [30, 33, 35, 38, 40, 42]:
    ev = FlexibleEvaluator(
        returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
        industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
        rebalance=60, n_clusters=nc, adj_fn=lambda c: weighted_topk_adj(c, k=5)
    )
    run_eval(ev, f"DP_RMT_cp0.6_WK5_nc{nc}", dp_rmt_wk5_fn)


# ===== 实验8: 因子数搜索 (Factor base, WeightedK5) =====
print("\n" + "="*70)
print("实验8: 因子数搜索 (Factor base, WK5)")
print("="*70)

ev_wk5 = FlexibleEvaluator(
    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
    industry_prior=industry_prior, warmup=500, eval_freq=60, forecast=60,
    rebalance=60, n_clusters=35, adj_fn=lambda c: weighted_topk_adj(c, k=5)
)

for nf in [10, 15, 25, 30]:
    run_eval(ev_wk5, f"DP_Factor_k{nf}_cp0.6_WK5",
             make_dual_path_fn(lambda h, nnf=nf: pca_factor_fn(h, nnf), industry_prior, cp=0.6))


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
print("Round 2 Full Results (sorted by CompositeScore)")
print("="*70)

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 300)
pd.set_option('display.float_format', '{:.4f}'.format)
print(df[['method','NMI','ARI','Modularity','IC','CovError','RankIC','Sharpe','Sortino','MaxDD','Calmar','NMI_Std','CompositeScore']].to_string())

df.to_csv(f'{OUT_DIR}/round2_results.csv')
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

# Top 5 
print("\n\n=== Top 5 ===")
for idx, row in df.head(5).iterrows():
    print(f"  #{idx}: {row['method']:50s} NMI={row['NMI']:.4f} Sharpe={row['Sharpe']:.4f} Composite={row['CompositeScore']:.4f}")

print(f"\n对比基准 DP_RMT_cp0.6_K3 (旧冠军): NMI=0.899, Sharpe=0.979, Composite=0.922")
print("\nRound 2 完成!")
