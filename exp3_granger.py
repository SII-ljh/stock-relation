"""
实验3: Granger因果网络
- 两两进行Granger因果检验
- 构建有向因果网络
- 评估因果关系的行业一致性和预测价值
- 由于271只股票两两检验太慢，先用主成分降维到50只代表性股票
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
import json
import os
import warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/exp3"
os.makedirs(OUT_DIR, exist_ok=True)

returns = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))

stocks = returns.columns.tolist()
n_stocks = len(stocks)
print(f"原始股票数: {n_stocks}")

# 选取每个行业的代表性股票（市值最大的前N只，这里用方差代理）
# 先选50只有代表性的股票
N_REPR = 50
stock_var = returns.var().sort_values(ascending=False)

# 从每个行业均匀选取
industry_stocks = {}
for s in stocks:
    ind = code_to_industry.get(s, 'Unknown')
    if ind not in industry_stocks:
        industry_stocks[ind] = []
    industry_stocks[ind].append(s)

selected = []
n_industries = len(industry_stocks)
per_industry = max(1, N_REPR // n_industries)

for ind, s_list in sorted(industry_stocks.items(), key=lambda x: -len(x[1])):
    if len(selected) >= N_REPR:
        break
    # 选方差最大的几只
    s_by_var = sorted(s_list, key=lambda x: returns[x].var(), reverse=True)
    n_pick = min(per_industry + 1, len(s_by_var), N_REPR - len(selected))
    selected.extend(s_by_var[:n_pick])

selected = selected[:N_REPR]
print(f"选取 {len(selected)} 只代表性股票进行Granger检验")
selected_returns = returns[selected]

# Granger因果检验
MAX_LAG = 5
SIG_LEVEL = 0.05

def granger_test_pair(i, j, data, max_lag):
    """测试 j -> i 的Granger因果"""
    try:
        test_data = data[[data.columns[i], data.columns[j]]].dropna()
        if len(test_data) < max_lag * 3:
            return i, j, 1.0, 0
        
        result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
        # 取所有lag中最小的p值
        min_p = min(result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1))
        best_lag = min(range(1, max_lag + 1), key=lambda lag: result[lag][0]['ssr_ftest'][1])
        return i, j, min_p, best_lag
    except:
        return i, j, 1.0, 0

print(f"\n开始Granger因果检验 ({len(selected)}x{len(selected)} = {len(selected)**2} 对)...")

# 并行计算
n = len(selected)
pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

results = Parallel(n_jobs=40, verbose=5)(
    delayed(granger_test_pair)(i, j, selected_returns, MAX_LAG)
    for i, j in pairs
)

# 构建因果矩阵
p_matrix = np.ones((n, n))
lag_matrix = np.zeros((n, n))
for i, j, p, lag in results:
    p_matrix[i, j] = p
    lag_matrix[i, j] = lag

# 显著性网络
sig_adj = (p_matrix < SIG_LEVEL).astype(float)
np.fill_diagonal(sig_adj, 0)

n_sig_edges = int(sig_adj.sum())
n_total_pairs = n * (n - 1)
print(f"\n显著因果关系: {n_sig_edges}/{n_total_pairs} ({n_sig_edges/n_total_pairs:.2%})")

# 更严格的阈值
for alpha in [0.05, 0.01, 0.001]:
    adj_alpha = (p_matrix < alpha).astype(float)
    np.fill_diagonal(adj_alpha, 0)
    print(f"alpha={alpha}: {int(adj_alpha.sum())} 条有向边")

# 评估行业一致性（有向）
def directed_industry_consistency(adj, stocks, code_to_industry):
    same = 0
    total = 0
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] > 0 and i != j:
                total += 1
                if code_to_industry.get(stocks[i], 'X') == code_to_industry.get(stocks[j], 'Y'):
                    same += 1
    return same / max(total, 1)

# 无向化（取OR）
undirected_adj = ((sig_adj + sig_adj.T) > 0).astype(float)
np.fill_diagonal(undirected_adj, 0)

# NMI评估
def eval_nmi(adj, stocks, code_to_industry, n_clusters=10):
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

dir_ic = directed_industry_consistency(sig_adj, selected, code_to_industry)
undir_ic = directed_industry_consistency(undirected_adj, selected, code_to_industry)
nmi = eval_nmi(undirected_adj, selected, code_to_industry, n_clusters=min(10, n//3))

print(f"\n行业一致性 (有向): {dir_ic:.4f}")
print(f"行业一致性 (无向): {undir_ic:.4f}")
print(f"NMI (vs Industry): {nmi:.4f}")

# 节点中心性分析
import networkx as nx
G = nx.from_numpy_array(sig_adj, create_using=nx.DiGraph)

# 入度（被其他股票Granger因果的次数）
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())

# 映射回股票代码
top_in = sorted(in_degrees.items(), key=lambda x: -x[1])[:10]
top_out = sorted(out_degrees.items(), key=lambda x: -x[1])[:10]

print(f"\n入度Top10 (被其他股票预测):")
for idx, deg in top_in:
    s = selected[idx]
    ind = code_to_industry.get(s, '?')
    print(f"  {s} ({ind}): 入度={deg}")

print(f"\n出度Top10 (能预测其他股票):")
for idx, deg in top_out:
    s = selected[idx]
    ind = code_to_industry.get(s, '?')
    print(f"  {s} ({ind}): 出度={deg}")

# 预测价值评估：用Granger因果邻居的加权收益预测
print("\n=== 预测价值评估 ===")
train_end = int(len(returns) * 0.7)
test_returns = selected_returns.iloc[train_end:].values
train_returns = selected_returns.iloc[:train_end].values

# 对每只股票，用其Granger父节点（p<0.05）的昨日收益预测今日收益
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

pred_r2 = []
baseline_r2 = []

for i in range(n):
    # 找Granger父节点
    parents = [j for j in range(n) if sig_adj[i, j] > 0 and j != i]
    
    if len(parents) == 0:
        continue
    
    # 特征：父节点的滞后收益
    X_train = train_returns[1:, parents]  # 用t-1预测t
    y_train = train_returns[1:, i]
    X_test = test_returns[1:, parents]
    y_test = test_returns[1:, i]
    
    if len(X_train) < 30:
        continue
    
    # Granger预测
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    pred_r2.append(r2)
    
    # 基准：自回归
    X_train_ar = train_returns[:-1, i:i+1]
    X_test_ar = test_returns[:-1, i:i+1]
    lr_ar = LinearRegression()
    lr_ar.fit(X_train_ar, y_train)
    y_pred_ar = lr_ar.predict(X_test_ar)
    r2_ar = r2_score(y_test, y_pred_ar)
    baseline_r2.append(r2_ar)

if pred_r2:
    print(f"Granger预测 R²: {np.mean(pred_r2):.6f} (std={np.std(pred_r2):.6f})")
    print(f"AR基准 R²: {np.mean(baseline_r2):.6f}")
    print(f"提升: {np.mean(pred_r2) - np.mean(baseline_r2):.6f}")

# 保存
summary = {
    'experiment': 'Exp3: Granger Causality Network',
    'n_representative_stocks': len(selected),
    'max_lag': MAX_LAG,
    'significance_level': SIG_LEVEL,
    'n_significant_edges': n_sig_edges,
    'directed_industry_consistency': float(dir_ic),
    'undirected_industry_consistency': float(undir_ic),
    'nmi_vs_industry': float(nmi),
    'prediction_r2_granger': float(np.mean(pred_r2)) if pred_r2 else None,
    'prediction_r2_ar_baseline': float(np.mean(baseline_r2)) if baseline_r2 else None,
}

with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

pd.DataFrame(p_matrix, index=selected, columns=selected).to_csv(f'{OUT_DIR}/granger_pvalues.csv')

print("\n实验3完成!")
