"""
全量策略 Walk-Forward 严格评估
覆盖6轮迭代中出现的所有不同策略, 统一在V2框架下评估。

策略分类:
  A. 基线 (3): SamplePearson, LedoitWolf, EqualWeight
  B. 去噪 (4): RMT, POET15, POET20, PCA Factor k20, NonlinearShrinkage
  C. 单路径+先验 (7): RMT+Prior (pw=0.2/0.3/0.5), POET+Prior k15 pw0.3,
                       Adaptive EWMA (pw=0.3/0.7), MultiScale EWMA, Pure Prior
  D. DualPath pp=0 (7): DP_RMT K3/K4, DP_POET15 K3/K4, DP_POET20 K3, DP_Factor K3/K4
  E. DualPath pp>0 (2): DP_RMT pp0.1 K4, DP_RMT+Prior pp0.3 K4
  F. GLasso (1): GLasso偏相关

共 ~25 种策略
"""
import sys
sys.path.insert(0, '/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation')

import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf, GraphicalLassoCV
from eval_framework_v2 import (
    WalkForwardEvaluator, build_industry_prior, cov_to_corr, ensure_psd,
    format_results_table, compute_composite_score, eval_portfolio_metrics,
    topk_adj, eval_nmi
)
import json, os, time
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/eval_final"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("全量策略 Walk-Forward 严格评估")
print("=" * 70)

returns_df = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns_df.columns.tolist()
n_stocks = len(stocks)
ret_vals = returns_df.values
industry_prior = build_industry_prior(stocks, code_to_industry)

print(f"股票数: {n_stocks}, 交易日: {len(ret_vals)}")
print(f"Warmup: 500天, 评估间隔: 60天, 预测窗口: 60天")

# 两套评估器: K=3 和 K=4
ev4 = WalkForwardEvaluator(
    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
    industry_prior=industry_prior,
    warmup=500, eval_freq=60, forecast=60, rebalance=60,
    n_clusters=35, topk=4
)
ev3 = WalkForwardEvaluator(
    returns=ret_vals, stocks=stocks, code_to_industry=code_to_industry,
    industry_prior=industry_prior,
    warmup=500, eval_freq=60, forecast=60, rebalance=60,
    n_clusters=35, topk=3
)
print(f"评估点数: {len(ev4.eval_points)}")

all_results = []
t0 = time.time()

# ============================================================
# 通用估计函数
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

def pca_factor_fn(history, nf=20):
    cov_s = np.cov(history.T)
    eigvals, eigvecs = np.linalg.eigh(cov_s)
    idx = np.argsort(eigvals)[::-1]; eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    B = eigvecs[:, :nf]; fv = eigvals[:nf]
    cov_f = B @ np.diag(fv) @ B.T
    rv = np.maximum(np.diag(cov_s) - np.sum(B**2 * fv, axis=1), 1e-8)
    c = ensure_psd(cov_f + np.diag(rv))
    return cov_to_corr(c), c

def nonlinear_shrinkage_fn(history):
    T, N = history.shape
    cov_s = np.cov(history.T)
    eigvals, eigvecs = np.linalg.eigh(cov_s)
    idx = np.argsort(eigvals)[::-1]; eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    q = N / T
    shrunk = np.zeros_like(eigvals)
    for i in range(N):
        diffs = eigvals - eigvals[i]; diffs[i] = 1
        h = max(eigvals[i] * (N/T)**0.5 * 0.1, 1e-8)
        kv = np.sum(h / (diffs**2 + h**2)) / (N * np.pi)
        sf = 1.0 / (1 + q * kv * np.pi * eigvals[i])
        shrunk[i] = eigvals[i] * max(sf, 0.01)
    shrunk = np.maximum(shrunk, 1e-8)
    c = ensure_psd(eigvecs @ np.diag(shrunk) @ eigvecs.T)
    return cov_to_corr(c), c

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

def adaptive_ewma_fn(history, hl=252, pw=0.7):
    T, N = history.shape
    decay = np.log(2) / hl
    step = 20
    cov_ewma = np.cov(history[:step].T)
    count = step
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

def multiscale_ewma_fn(history, hls=(42,126,252), ws=(0.2,0.3,0.5), pw=0.5):
    covs = []
    for hl in hls:
        _, c = adaptive_ewma_fn(history, hl=hl, pw=pw)
        covs.append(c)
    cov = sum(w*c for w,c in zip(ws, covs))
    return cov_to_corr(cov), cov


def run_eval(evaluator, name, fn):
    t1 = time.time()
    r = evaluator.evaluate(name, fn)
    elapsed = time.time() - t1
    print(f"  [{len(all_results)+1:2d}] {name:40s} NMI={r['NMI']:.4f} CovErr={r['CovError']:.4f} RankIC={r['RankIC']:.4f} Sharpe={r['Sharpe']:.4f} ({elapsed:.0f}s)")
    all_results.append(r)
    return r


# ============================================================
# A. 基线方法
# ============================================================
print("\n=== A. 基线方法 ===")

run_eval(ev4, "SamplePearson",
    lambda h: (cov_to_corr(np.cov(h.T)), np.cov(h.T)))

run_eval(ev4, "LedoitWolf",
    lambda h: (cov_to_corr(LedoitWolf().fit(h).covariance_), LedoitWolf().fit(h).covariance_))

# 等权
eq_rets = ret_vals[500:].mean(axis=1)
pm_eq = eval_portfolio_metrics(eq_rets.tolist())
r_eq = {'method': 'EqualWeight',
        'NMI': np.nan, 'ARI': np.nan, 'Modularity': np.nan, 'IC': np.nan,
        'CovError': np.nan, 'LogLik': np.nan, 'RankIC': np.nan,
        'Sharpe': pm_eq['sharpe'], 'Sortino': pm_eq['sortino'],
        'MaxDD': pm_eq['max_drawdown'], 'Calmar': pm_eq['calmar'],
        'NMI_Std': np.nan, 'CovErr_Std': np.nan}
all_results.append(r_eq)
print(f"  [ {len(all_results):2d}] {'EqualWeight':40s} Sharpe={r_eq['Sharpe']:.4f}")


# ============================================================
# B. 去噪方法 (无先验)
# ============================================================
print("\n=== B. 去噪方法 ===")

run_eval(ev4, "RMT_Denoise", rmt_denoise_fn)
run_eval(ev4, "POET15", lambda h: poet_fn(h, 15))
run_eval(ev4, "POET20", lambda h: poet_fn(h, 20))
run_eval(ev4, "PCA_Factor_k20", pca_factor_fn)
run_eval(ev4, "NonlinearShrinkage", nonlinear_shrinkage_fn)


# ============================================================
# C. 单路径+先验
# ============================================================
print("\n=== C. 单路径+先验 ===")

run_eval(ev4, "RMT+Prior_pw0.2", make_prior_fn(rmt_denoise_fn, industry_prior, 0.2))
run_eval(ev4, "RMT+Prior_pw0.3", make_prior_fn(rmt_denoise_fn, industry_prior, 0.3))
run_eval(ev4, "RMT+Prior_pw0.5", make_prior_fn(rmt_denoise_fn, industry_prior, 0.5))
run_eval(ev4, "POET15+Prior_pw0.3", make_prior_fn(lambda h: poet_fn(h,15), industry_prior, 0.3))
run_eval(ev4, "Adaptive_EWMA_pw0.3", lambda h: adaptive_ewma_fn(h, 252, 0.3))
run_eval(ev4, "Adaptive_EWMA_pw0.7", lambda h: adaptive_ewma_fn(h, 252, 0.7))
run_eval(ev4, "MultiScale_EWMA_pw0.5", lambda h: multiscale_ewma_fn(h, pw=0.5))

# 纯行业先验
def pure_prior_fn(history):
    cov_s = np.cov(history.T)
    corr_p = industry_prior.copy(); np.fill_diagonal(corr_p, 1)
    sample_var = np.diag(cov_s)
    cov_p = industry_prior * np.mean(sample_var) * 0.5
    np.fill_diagonal(cov_p, sample_var)
    return corr_p, ensure_psd(cov_p)
run_eval(ev4, "PureIndustryPrior", pure_prior_fn)


# ============================================================
# D. DualPath pp=0 (核心)
# ============================================================
print("\n=== D. DualPath pp=0 ===")

run_eval(ev4, "DP_RMT_cp0.6_K4", make_dual_path_fn(rmt_denoise_fn, industry_prior, 0.6, 0.0))
run_eval(ev3, "DP_RMT_cp0.6_K3", make_dual_path_fn(rmt_denoise_fn, industry_prior, 0.6, 0.0))
run_eval(ev4, "DP_POET15_cp0.6_K4", make_dual_path_fn(lambda h: poet_fn(h,15), industry_prior, 0.6, 0.0))
run_eval(ev3, "DP_POET15_cp0.6_K3", make_dual_path_fn(lambda h: poet_fn(h,15), industry_prior, 0.6, 0.0))
run_eval(ev3, "DP_POET20_cp0.6_K3", make_dual_path_fn(lambda h: poet_fn(h,20), industry_prior, 0.6, 0.0))
run_eval(ev3, "DP_Factor_cp0.6_K3", make_dual_path_fn(pca_factor_fn, industry_prior, 0.6, 0.0))
run_eval(ev4, "DP_Factor_cp0.6_K4", make_dual_path_fn(pca_factor_fn, industry_prior, 0.6, 0.0))


# ============================================================
# E. DualPath pp>0
# ============================================================
print("\n=== E. DualPath pp>0 ===")

run_eval(ev4, "DP_RMT_cp0.6_pp0.1_K4", make_dual_path_fn(rmt_denoise_fn, industry_prior, 0.6, 0.1))
run_eval(ev4, "DP_RMT_cp0.6_pp0.3_K4", make_dual_path_fn(rmt_denoise_fn, industry_prior, 0.6, 0.3))


# ============================================================
# F. GLasso
# ============================================================
print("\n=== F. GLasso ===")

def glasso_fn(history):
    try:
        gl = GraphicalLassoCV(cv=3, max_iter=100)
        gl.fit(history[-500:])  # 用最近500天
        prec = gl.precision_
        d = np.sqrt(np.diag(prec)); d[d==0] = 1e-10
        partial_corr = -prec / np.outer(d, d)
        np.fill_diagonal(partial_corr, 1)
        return partial_corr, gl.covariance_
    except:
        cov = np.cov(history.T)
        return cov_to_corr(cov), cov
run_eval(ev4, "GLasso_Partial", glasso_fn)


# ============================================================
# 汇总排名
# ============================================================
elapsed_total = time.time() - t0
print(f"\n总耗时: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")

print("\n" + "=" * 70)
print("全量策略 Walk-Forward 严格评估 - 最终排名")
print("=" * 70)

df = format_results_table(all_results, sort_by='NMI', ascending=False)
df['CompositeScore'] = compute_composite_score(df)
df = df.sort_values('CompositeScore', ascending=False).reset_index(drop=True)
df.index = df.index + 1
df.index.name = 'Rank'

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 250)
pd.set_option('display.float_format', '{:.4f}'.format)
print(df.to_string())

# 保存
df.to_csv(f'{OUT_DIR}/final_ranking.csv')
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

# Top 10 markdown
print("\n\n--- Top 10 Markdown ---")
print("| Rank | Method | NMI | ARI | Modul. | IC | CovErr | LogLik | RankIC | Sharpe | Sortino | MaxDD | Calmar | Composite |")
print("|:----:|--------|:---:|:---:|:------:|:--:|:------:|:------:|:------:|:------:|:-------:|:-----:|:------:|:---------:|")
for idx, row in df.head(10).iterrows():
    nmi_s = f"{row['NMI']:.4f}" if not np.isnan(row.get('NMI', np.nan)) else "—"
    ari_s = f"{row['ARI']:.4f}" if not np.isnan(row.get('ARI', np.nan)) else "—"
    mod_s = f"{row['Modularity']:.4f}" if not np.isnan(row.get('Modularity', np.nan)) else "—"
    ic_s = f"{row['IC']:.4f}" if not np.isnan(row.get('IC', np.nan)) else "—"
    ce_s = f"{row['CovError']:.4f}" if not np.isnan(row.get('CovError', np.nan)) else "—"
    ll_s = f"{row['LogLik']:.1f}" if not np.isnan(row.get('LogLik', np.nan)) else "—"
    ri_s = f"{row['RankIC']:.4f}" if not np.isnan(row.get('RankIC', np.nan)) else "—"
    print(f"| {idx} | {row['method']} | {nmi_s} | {ari_s} | {mod_s} | {ic_s} | {ce_s} | {ll_s} | {ri_s} | {row['Sharpe']:.4f} | {row['Sortino']:.4f} | {row['MaxDD']:.4f} | {row['Calmar']:.4f} | {row['CompositeScore']:.4f} |")

print(f"\n结果已保存到 {OUT_DIR}/")
print("\n全量评估完成!")
