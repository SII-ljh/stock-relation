"""
实验2: 动态相关性模型
比较3种方法:
1. 滚动窗口Pearson相关性
2. Ledoit-Wolf收缩估计（更稳健）
3. EWMA指数加权（近期权重更大）
评估: 协方差预测精度、稳定性、行业一致性
"""
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering
import networkx as nx
import json
import os
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/exp2"
os.makedirs(OUT_DIR, exist_ok=True)

returns = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns.columns.tolist()
n_stocks = len(stocks)

print(f"股票数: {n_stocks}, 交易日: {len(returns)}")

# ===== 辅助函数 =====
def corr_to_topk_adj(corr, k=5):
    n = corr.shape[0]
    adj = np.zeros_like(corr)
    for i in range(n):
        row = corr[i].copy()
        row[i] = -np.inf
        top_indices = np.argsort(row)[-k:]
        adj[i, top_indices] = 1
    adj = ((adj + adj.T) > 0).astype(float)
    np.fill_diagonal(adj, 0)
    return adj

def eval_industry_nmi(adj, stocks, code_to_industry, n_clusters=10):
    try:
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                               random_state=42, n_init=5)
        adj_sc = adj.copy()
        np.fill_diagonal(adj_sc, 1)
        pred = sc.fit_predict(adj_sc)
        unique_inds = sorted(set(code_to_industry.values()))
        ind_map = {ind: i for i, ind in enumerate(unique_inds)}
        true_labels = [ind_map.get(code_to_industry.get(s, 'Unknown'), 0) for s in stocks]
        return normalized_mutual_info_score(true_labels, pred)
    except:
        return 0.0

def industry_consistency(adj, stocks, code_to_industry):
    n = adj.shape[0]
    same = 0
    total = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0:
                total += 1
                if code_to_industry.get(stocks[i], 'X') == code_to_industry.get(stocks[j], 'Y'):
                    same += 1
    return same / max(total, 1)

# ===== 1. 滚动窗口Pearson =====
print("\n=== 1. 滚动窗口Pearson ===")
WINDOW_SIZES = [60, 120, 250]
TOPK = 5

def rolling_pearson_eval(returns, window_size, test_start_idx):
    """用window_size窗口估计协方差，预测下一个窗口"""
    train_end = test_start_idx
    train_start = max(0, train_end - window_size)
    test_end = min(len(returns), test_start_idx + window_size)
    
    train = returns.iloc[train_start:train_end]
    test = returns.iloc[test_start_idx:test_end]
    
    if len(train) < 30 or len(test) < 30:
        return None
    
    cov_pred = train.cov().values
    cov_true = test.cov().values
    
    # 相关矩阵
    std_pred = np.sqrt(np.diag(cov_pred))
    std_pred[std_pred == 0] = 1e-10
    corr_pred = cov_pred / np.outer(std_pred, std_pred)
    np.fill_diagonal(corr_pred, 1)
    corr_pred = np.clip(corr_pred, -1, 1)
    
    frob = np.linalg.norm(cov_pred - cov_true, 'fro')
    rel_err = frob / np.linalg.norm(cov_true, 'fro')
    
    return {
        'frobenius': frob,
        'relative_error': rel_err,
        'corr_pred': corr_pred,
    }

# ===== 2. Ledoit-Wolf收缩估计 =====
print("\n=== 2. Ledoit-Wolf收缩估计 ===")

def lw_eval(returns, window_size, test_start_idx):
    train_end = test_start_idx
    train_start = max(0, train_end - window_size)
    test_end = min(len(returns), test_start_idx + window_size)
    
    train = returns.iloc[train_start:train_end].values
    test = returns.iloc[test_start_idx:test_end].values
    
    if len(train) < 30 or len(test) < 30:
        return None
    
    lw = LedoitWolf()
    lw.fit(train)
    cov_pred = lw.covariance_
    cov_true = np.cov(test.T)
    
    std_pred = np.sqrt(np.diag(cov_pred))
    std_pred[std_pred == 0] = 1e-10
    corr_pred = cov_pred / np.outer(std_pred, std_pred)
    np.fill_diagonal(corr_pred, 1)
    corr_pred = np.clip(corr_pred, -1, 1)
    
    frob = np.linalg.norm(cov_pred - cov_true, 'fro')
    rel_err = frob / np.linalg.norm(cov_true, 'fro')
    
    return {
        'frobenius': frob,
        'relative_error': rel_err,
        'shrinkage': lw.shrinkage_,
        'corr_pred': corr_pred,
    }

# ===== 3. EWMA指数加权 =====
print("\n=== 3. EWMA指数加权 ===")
HALF_LIVES = [21, 63, 126]  # 1个月、3个月、半年

def ewma_eval(returns, half_life, test_start_idx, window_size=250):
    train_end = test_start_idx
    train_start = max(0, train_end - window_size)
    test_end = min(len(returns), test_start_idx + window_size)
    
    train = returns.iloc[train_start:train_end].values
    test = returns.iloc[test_start_idx:test_end].values
    
    if len(train) < 30 or len(test) < 30:
        return None
    
    # EWMA权重
    decay = np.log(2) / half_life
    n_train = len(train)
    weights = np.exp(-decay * np.arange(n_train)[::-1])
    weights /= weights.sum()
    
    # 加权均值和协方差
    mean = np.average(train, axis=0, weights=weights)
    centered = train - mean
    cov_pred = np.zeros((train.shape[1], train.shape[1]))
    for t in range(n_train):
        cov_pred += weights[t] * np.outer(centered[t], centered[t])
    
    cov_true = np.cov(test.T)
    
    std_pred = np.sqrt(np.diag(cov_pred))
    std_pred[std_pred == 0] = 1e-10
    corr_pred = cov_pred / np.outer(std_pred, std_pred)
    np.fill_diagonal(corr_pred, 1)
    corr_pred = np.clip(corr_pred, -1, 1)
    
    frob = np.linalg.norm(cov_pred - cov_true, 'fro')
    rel_err = frob / np.linalg.norm(cov_true, 'fro')
    
    return {
        'frobenius': frob,
        'relative_error': rel_err,
        'corr_pred': corr_pred,
    }

# ===== 滚动评估 =====
print("\n=== 滚动评估 ===")
STEP = 60  # 每60天评估一次
test_starts = list(range(250, len(returns) - 60, STEP))  # 从第250天开始
print(f"评估窗口数: {len(test_starts)}")

all_results = {}

# 1) 滚动Pearson
for ws in WINDOW_SIZES:
    key = f'Rolling_Pearson_w{ws}'
    results = []
    for ts in test_starts:
        r = rolling_pearson_eval(returns, ws, ts)
        if r:
            results.append(r)
    
    avg_frob = np.mean([r['frobenius'] for r in results])
    avg_rel = np.mean([r['relative_error'] for r in results])
    
    # 用最后一个窗口评估行业NMI
    last_corr = results[-1]['corr_pred']
    adj = corr_to_topk_adj(last_corr, TOPK)
    nmi = eval_industry_nmi(adj, stocks, code_to_industry)
    ind_cons = industry_consistency(adj, stocks, code_to_industry)
    
    all_results[key] = {
        'method': key,
        'avg_cov_relative_error': float(avg_rel),
        'avg_cov_frobenius': float(avg_frob),
        'nmi': float(nmi),
        'industry_consistency': float(ind_cons),
    }
    print(f"{key}: rel_err={avg_rel:.4f}, NMI={nmi:.4f}, ind_cons={ind_cons:.4f}")

# 2) Ledoit-Wolf
for ws in WINDOW_SIZES:
    key = f'LedoitWolf_w{ws}'
    results = []
    for ts in test_starts:
        r = lw_eval(returns, ws, ts)
        if r:
            results.append(r)
    
    avg_frob = np.mean([r['frobenius'] for r in results])
    avg_rel = np.mean([r['relative_error'] for r in results])
    avg_shrink = np.mean([r['shrinkage'] for r in results])
    
    last_corr = results[-1]['corr_pred']
    adj = corr_to_topk_adj(last_corr, TOPK)
    nmi = eval_industry_nmi(adj, stocks, code_to_industry)
    ind_cons = industry_consistency(adj, stocks, code_to_industry)
    
    all_results[key] = {
        'method': key,
        'avg_cov_relative_error': float(avg_rel),
        'avg_cov_frobenius': float(avg_frob),
        'avg_shrinkage': float(avg_shrink),
        'nmi': float(nmi),
        'industry_consistency': float(ind_cons),
    }
    print(f"{key}: rel_err={avg_rel:.4f}, shrinkage={avg_shrink:.4f}, NMI={nmi:.4f}")

# 3) EWMA
for hl in HALF_LIVES:
    key = f'EWMA_hl{hl}'
    results = []
    for ts in test_starts:
        r = ewma_eval(returns, hl, ts)
        if r:
            results.append(r)
    
    avg_frob = np.mean([r['frobenius'] for r in results])
    avg_rel = np.mean([r['relative_error'] for r in results])
    
    last_corr = results[-1]['corr_pred']
    adj = corr_to_topk_adj(last_corr, TOPK)
    nmi = eval_industry_nmi(adj, stocks, code_to_industry)
    ind_cons = industry_consistency(adj, stocks, code_to_industry)
    
    all_results[key] = {
        'method': key,
        'avg_cov_relative_error': float(avg_rel),
        'avg_cov_frobenius': float(avg_frob),
        'nmi': float(nmi),
        'industry_consistency': float(ind_cons),
    }
    print(f"{key}: rel_err={avg_rel:.4f}, NMI={nmi:.4f}, ind_cons={ind_cons:.4f}")

# 静态基准
corr_full = returns.corr().values
adj_full = corr_to_topk_adj(corr_full, TOPK)
nmi_full = eval_industry_nmi(adj_full, stocks, code_to_industry)
all_results['Static_Pearson'] = {
    'method': 'Static_Pearson',
    'avg_cov_relative_error': 0.4245,  # 从exp1
    'nmi': float(nmi_full),
    'industry_consistency': float(industry_consistency(adj_full, stocks, code_to_industry)),
}
print(f"\nStatic_Pearson (baseline): NMI={nmi_full:.4f}")

# 保存
results_df = pd.DataFrame(all_results.values())
results_df.to_csv(f'{OUT_DIR}/results.csv', index=False)

with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

# 找最优
best = results_df.loc[results_df['avg_cov_relative_error'].idxmin()]
print(f"\n协方差预测最优: {best['method']} (rel_err={best['avg_cov_relative_error']:.4f})")
best_nmi = results_df.loc[results_df['nmi'].idxmax()]
print(f"NMI最优: {best_nmi['method']} (NMI={best_nmi['nmi']:.4f})")

print("\n实验2完成!")
