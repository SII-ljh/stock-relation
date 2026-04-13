"""
实验5: 综合关系模型
1. 融合Pearson相关 + 偏相关 + Granger因果信号
2. 通过加权邻接矩阵评估
3. 下游任务验证:
   a. 收益率预测（用邻居信息预测）
   b. 最小方差投资组合优化
   c. 风险预测精度
"""
import pandas as pd
import numpy as np
from sklearn.covariance import GraphicalLasso, LedoitWolf
from sklearn.metrics import normalized_mutual_info_score, mean_squared_error, r2_score
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/exp5"
os.makedirs(OUT_DIR, exist_ok=True)

returns = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns.columns.tolist()
n_stocks = len(stocks)

print(f"股票数: {n_stocks}, 交易日: {len(returns)}")

# ===== 计算三种关系信号 =====
print("\n=== 计算三种关系信号 ===")

# 1) Pearson相关
pearson_corr = returns.corr().values
print(f"Pearson相关: shape={pearson_corr.shape}")

# 2) 偏相关 (从GLasso)
scaler = StandardScaler()
returns_std = scaler.fit_transform(returns.values)
gl = GraphicalLasso(alpha=0.27, max_iter=200)
gl.fit(returns_std)
precision = gl.precision_
d = np.sqrt(np.diag(precision))
d[d == 0] = 1e-10
partial_corr = -precision / np.outer(d, d)
np.fill_diagonal(partial_corr, 1)
partial_corr = np.clip(partial_corr, -1, 1)
print(f"偏相关: non-zero={(np.abs(partial_corr) > 0.01).sum()}")

# 3) 简化的因果信号: 滞后相关性（i的t-1 vs j的t）
print("计算滞后互相关...")
lag_corr = np.zeros((n_stocks, n_stocks))
ret_vals = returns.values
for i in range(n_stocks):
    for j in range(n_stocks):
        if i != j:
            # corr(i_{t-1}, j_t)
            lag_corr[i, j] = np.corrcoef(ret_vals[:-1, i], ret_vals[1:, j])[0, 1]
lag_corr = np.nan_to_num(lag_corr)
print(f"滞后相关: mean_abs={np.abs(lag_corr).mean():.6f}")

# ===== 融合策略 =====
print("\n=== 融合关系矩阵 ===")

def normalize_matrix(mat):
    """归一化到[0,1]"""
    abs_mat = np.abs(mat)
    np.fill_diagonal(abs_mat, 0)
    max_val = abs_mat.max()
    if max_val > 0:
        return abs_mat / max_val
    return abs_mat

pearson_norm = normalize_matrix(pearson_corr)
partial_norm = normalize_matrix(partial_corr)
lag_norm = normalize_matrix(lag_corr)

def fuse_matrices(w_pearson, w_partial, w_lag):
    """加权融合"""
    fused = w_pearson * pearson_norm + w_partial * partial_norm + w_lag * lag_norm
    np.fill_diagonal(fused, 0)
    return fused

def topk_adj(mat, k=5):
    n = mat.shape[0]
    adj = np.zeros_like(mat)
    for i in range(n):
        row = mat[i].copy()
        row[i] = -np.inf
        top = np.argsort(row)[-k:]
        adj[i, top] = 1
    return ((adj + adj.T) > 0).astype(float)

def eval_nmi(adj, stocks, code_to_industry):
    try:
        abs_adj = np.abs(adj).astype(float)
        np.fill_diagonal(abs_adj, abs_adj.max())
        sc = SpectralClustering(n_clusters=10, affinity='precomputed',
                               random_state=42, n_init=5)
        pred = sc.fit_predict(abs_adj)
        unique_inds = sorted(set(code_to_industry.values()))
        ind_map = {ind: i for i, ind in enumerate(unique_inds)}
        true_labels = [ind_map.get(code_to_industry.get(s, 'Unknown'), 0) for s in stocks]
        return normalized_mutual_info_score(true_labels, pred)
    except:
        return 0.0

# 测试不同权重组合
weight_combos = [
    (1.0, 0.0, 0.0, 'Pearson_only'),
    (0.0, 1.0, 0.0, 'Partial_only'),
    (0.0, 0.0, 1.0, 'Lag_only'),
    (0.5, 0.5, 0.0, 'Pearson+Partial'),
    (0.5, 0.0, 0.5, 'Pearson+Lag'),
    (0.0, 0.5, 0.5, 'Partial+Lag'),
    (0.4, 0.4, 0.2, 'Balanced'),
    (0.5, 0.3, 0.2, 'Pearson_heavy'),
    (0.3, 0.5, 0.2, 'Partial_heavy'),
    (0.3, 0.3, 0.4, 'Lag_heavy'),
]

fusion_results = []
for wp, wpc, wl, name in weight_combos:
    fused = fuse_matrices(wp, wpc, wl)
    adj = topk_adj(fused, k=5)
    np.fill_diagonal(adj, 0)
    
    # NMI
    nmi = eval_nmi(adj, stocks, code_to_industry)
    
    # 行业一致性
    same = total = 0
    for i in range(n_stocks):
        for j in range(i+1, n_stocks):
            if adj[i, j] > 0:
                total += 1
                if code_to_industry.get(stocks[i], 'X') == code_to_industry.get(stocks[j], 'Y'):
                    same += 1
    ic = same / max(total, 1)
    
    fusion_results.append({
        'name': name, 'w_pearson': wp, 'w_partial': wpc, 'w_lag': wl,
        'nmi': nmi, 'industry_consistency': ic, 'n_edges': int(adj.sum()/2),
    })
    print(f"{name}: NMI={nmi:.4f}, IC={ic:.4f}")

fusion_df = pd.DataFrame(fusion_results)
best_fusion = fusion_df.loc[fusion_df['nmi'].idxmax()]
print(f"\n最佳融合: {best_fusion['name']} (NMI={best_fusion['nmi']:.4f})")

# ===== 下游任务1: 收益率预测 =====
print("\n=== 下游任务1: 收益率预测 ===")

# 使用最佳融合矩阵
best_wp = best_fusion['w_pearson']
best_wpc = best_fusion['w_partial']
best_wl = best_fusion['w_lag']

TRAIN_RATIO = 0.7
n_train = int(len(returns) * TRAIN_RATIO)

# 分割
train_ret = returns.iloc[:n_train]
test_ret = returns.iloc[n_train:]

# 在训练集上计算关系矩阵
train_pearson = normalize_matrix(train_ret.corr().values)
scaler2 = StandardScaler()
train_std = scaler2.fit_transform(train_ret.values)
try:
    gl2 = GraphicalLasso(alpha=0.27, max_iter=200)
    gl2.fit(train_std)
    prec2 = gl2.precision_
    d2 = np.sqrt(np.diag(prec2))
    d2[d2 == 0] = 1e-10
    train_partial = -prec2 / np.outer(d2, d2)
    np.fill_diagonal(train_partial, 1)
    train_partial = np.clip(train_partial, -1, 1)
    train_partial_norm = normalize_matrix(train_partial)
except:
    train_partial_norm = np.zeros_like(train_pearson)

train_lag = np.zeros((n_stocks, n_stocks))
train_vals = train_ret.values
for i in range(n_stocks):
    for j in range(n_stocks):
        if i != j:
            train_lag[i, j] = np.corrcoef(train_vals[:-1, i], train_vals[1:, j])[0, 1]
train_lag = np.nan_to_num(train_lag)
train_lag_norm = normalize_matrix(train_lag)

fused_train = best_wp * train_pearson + best_wpc * train_partial_norm + best_wl * normalize_matrix(train_lag)
np.fill_diagonal(fused_train, 0)
adj_train = topk_adj(fused_train, k=5)

# 预测：用邻居的t-1收益预测t收益
test_vals = test_ret.values
predictions = {'ensemble': [], 'pearson_only': [], 'no_neighbor': []}
true_vals_list = []

# TopK=5 邻接矩阵 (Pearson only 对比)
adj_pearson = topk_adj(train_pearson, k=5)

for t in range(1, len(test_vals)):
    for method, adj_m in [('ensemble', adj_train), ('pearson_only', adj_pearson)]:
        preds = []
        for i in range(n_stocks):
            neighbors = np.where(adj_m[i] > 0)[0]
            if len(neighbors) > 0:
                # 邻居加权平均（用关系强度加权）
                weights = fused_train[i, neighbors] if method == 'ensemble' else train_pearson[i, neighbors]
                weights = weights / (weights.sum() + 1e-10)
                pred_i = np.dot(weights, test_vals[t-1, neighbors])
            else:
                pred_i = 0
            preds.append(pred_i)
        predictions[method].append(preds)
    
    # 自回归基准
    predictions['no_neighbor'].append(test_vals[t-1].tolist())
    true_vals_list.append(test_vals[t].tolist())

# 计算R²和MSE
true_arr = np.array(true_vals_list)
for method in ['ensemble', 'pearson_only', 'no_neighbor']:
    pred_arr = np.array(predictions[method])
    # 每只股票的R²
    r2_per_stock = []
    for i in range(n_stocks):
        r2 = r2_score(true_arr[:, i], pred_arr[:, i])
        r2_per_stock.append(r2)
    
    avg_r2 = np.mean(r2_per_stock)
    med_r2 = np.median(r2_per_stock)
    mse = mean_squared_error(true_arr.flatten(), pred_arr.flatten())
    print(f"{method}: avg_R²={avg_r2:.6f}, median_R²={med_r2:.6f}, MSE={mse:.8f}")

# ===== 下游任务2: 最小方差投资组合 =====
print("\n=== 下游任务2: 最小方差投资组合 ===")

def minimum_variance_portfolio(cov_matrix):
    """求解最小方差组合"""
    n = cov_matrix.shape[0]
    
    def objective(w):
        return w @ cov_matrix @ w
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 0.05)] * n  # 单只最多5%
    w0 = np.ones(n) / n
    
    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 500})
    return result.x if result.success else w0

# 分成多个时段评估
REBALANCE_PERIOD = 60  # 60天再平衡
LOOKBACK = 250

portfolio_results = {}

for method_name in ['sample_cov', 'ledoit_wolf', 'glasso']:
    realized_returns = []
    realized_vols = []
    
    for t_start in range(LOOKBACK, len(returns) - REBALANCE_PERIOD, REBALANCE_PERIOD):
        train_window = returns.iloc[t_start-LOOKBACK:t_start].values
        test_window = returns.iloc[t_start:t_start+REBALANCE_PERIOD].values
        
        if method_name == 'sample_cov':
            cov = np.cov(train_window.T)
        elif method_name == 'ledoit_wolf':
            lw = LedoitWolf()
            lw.fit(train_window)
            cov = lw.covariance_
        elif method_name == 'glasso':
            try:
                sc = StandardScaler()
                train_s = sc.fit_transform(train_window)
                gl = GraphicalLasso(alpha=0.1, max_iter=100)
                gl.fit(train_s)
                cov = gl.covariance_ * np.outer(sc.scale_, sc.scale_)
            except:
                lw = LedoitWolf()
                lw.fit(train_window)
                cov = lw.covariance_
        
        # 确保正定
        cov = cov + np.eye(n_stocks) * 1e-6
        
        weights = minimum_variance_portfolio(cov)
        
        # 实现收益
        port_returns = test_window @ weights
        realized_returns.extend(port_returns.tolist())
        realized_vols.append(np.std(port_returns) * np.sqrt(252))
    
    r = np.array(realized_returns)
    ann_ret = np.mean(r) * 252
    ann_vol = np.std(r) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = np.min(np.minimum.accumulate(np.cumsum(r)) - np.cumsum(r))
    
    portfolio_results[method_name] = {
        'annual_return': float(ann_ret),
        'annual_volatility': float(ann_vol),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_dd),
        'avg_rebalance_vol': float(np.mean(realized_vols)),
    }
    print(f"{method_name}: ret={ann_ret:.4f}, vol={ann_vol:.4f}, sharpe={sharpe:.4f}")

# 等权基准
eq_returns = returns.iloc[LOOKBACK:].values @ (np.ones(n_stocks) / n_stocks)
eq_ann_ret = np.mean(eq_returns) * 252
eq_ann_vol = np.std(eq_returns) * np.sqrt(252)
eq_sharpe = eq_ann_ret / eq_ann_vol
print(f"equal_weight: ret={eq_ann_ret:.4f}, vol={eq_ann_vol:.4f}, sharpe={eq_sharpe:.4f}")

# ===== 保存 =====
summary = {
    'experiment': 'Exp5: Ensemble Model + Downstream Validation',
    'fusion_results': fusion_results,
    'best_fusion': {
        'name': best_fusion['name'],
        'nmi': float(best_fusion['nmi']),
        'weights': {'pearson': float(best_wp), 'partial': float(best_wpc), 'lag': float(best_wl)},
    },
    'portfolio_results': portfolio_results,
}

with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

fusion_df.to_csv(f'{OUT_DIR}/fusion_results.csv', index=False)

print("\n实验5完成!")
