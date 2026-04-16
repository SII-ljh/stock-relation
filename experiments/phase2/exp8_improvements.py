"""
实验8: 改进方向探索

改进1: 调整聚类数 - 原来用10个，但实际有42个行业
改进2: 扩展先验权重范围 - 测试0.35-0.6
改进3: 行业规模非均匀先验 - 大行业给更高权重
"""
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf, GraphicalLasso
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/exp8"
os.makedirs(OUT_DIR, exist_ok=True)

returns = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns.columns.tolist()
n_stocks = len(stocks)
ret_vals = returns.values

print(f"股票数: {n_stocks}, 行业数: {len(industry['industry'].unique())}")

# ===== 构建行业先验 =====
def build_industry_prior(stocks, code_to_industry, weight_mode='uniform', industry_sizes=None):
    """
    weight_mode: 'uniform' - 均匀权重
                 'size' - 基于行业规模的权重
                 'sqrt_size' - 基于行业规模平方根的权重
    """
    n = len(stocks)
    prior = np.zeros((n, n))
    
    # 计算行业规模
    ind_map = {}
    for i, s in enumerate(stocks):
        ind = code_to_industry.get(s, 'Unknown')
        if ind not in ind_map:
            ind_map[ind] = []
        ind_map[ind].append(i)
    
    if industry_sizes is None:
        industry_sizes = {ind: len(members) for ind, members in ind_map.items()}
    
    for i in range(n):
        for j in range(n):
            ind_i = code_to_industry.get(stocks[i], 'X')
            ind_j = code_to_industry.get(stocks[j], 'Y')
            
            if ind_i == ind_j and ind_i != 'Unknown':
                if weight_mode == 'uniform':
                    prior[i, j] = 1.0
                elif weight_mode == 'size':
                    # 大行业给更高权重
                    size_i = industry_sizes.get(ind_i, 1)
                    prior[i, j] = np.log(1 + size_i)  # 对数尺度避免极端值
                elif weight_mode == 'sqrt_size':
                    size_i = industry_sizes.get(ind_i, 1)
                    prior[i, j] = np.sqrt(size_i)
    
    np.fill_diagonal(prior, 0)
    # 归一化到[0,1]
    if prior.max() > 0:
        prior = prior / prior.max()
    return prior, ind_map

industry_prior_uniform, ind_map = build_industry_prior(stocks, code_to_industry, 'uniform')
industry_prior_size, _ = build_industry_prior(stocks, code_to_industry, 'size')
industry_prior_sqrt, _ = build_industry_prior(stocks, code_to_industry, 'sqrt_size')

# 计算每个行业的大小
industry_sizes = {ind: len(members) for ind, members in ind_map.items()}
print(f"行业大小范围: {min(industry_sizes.values())} - {max(industry_sizes.values())}")

# ===== 评估函数 =====
def topk_adj(corr, k=5):
    n = corr.shape[0]
    adj = np.zeros_like(corr)
    for i in range(n):
        row = np.abs(corr[i].copy())
        row[i] = -np.inf
        top = np.argsort(row)[-k:]
        adj[i, top] = 1
    return ((adj + adj.T) > 0).astype(float)

def eval_nmi(adj, stocks, code_to_industry, n_clusters=10):
    """评估NMI，可指定聚类数"""
    try:
        abs_adj = np.abs(adj).astype(float)
        np.fill_diagonal(abs_adj, abs_adj.max())
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                               random_state=42, n_init=3)
        pred = sc.fit_predict(abs_adj)
        unique_inds = sorted(set(code_to_industry.values()))
        ind_map_label = {ind: i for i, ind in enumerate(unique_inds)}
        true_labels = [ind_map_label.get(code_to_industry.get(s, 'Unknown'), 0) for s in stocks]
        return normalized_mutual_info_score(true_labels, pred)
    except Exception as e:
        return 0.0

def industry_consistency(adj, stocks, code_to_industry):
    same = total = 0
    for i in range(adj.shape[0]):
        for j in range(i+1, adj.shape[1]):
            if adj[i, j] > 0:
                total += 1
                if code_to_industry.get(stocks[i], 'X') == code_to_industry.get(stocks[j], 'Y'):
                    same += 1
    return same / max(total, 1)

# ===== 自适应协方差估计器 =====
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

# ===== 实验设置 =====
WARMUP = 250
UPDATE_FREQ = 20
EVAL_FREQ = 60
FORECAST = 60

eval_points = list(range(WARMUP, len(ret_vals) - FORECAST, EVAL_FREQ))
print(f"评估点: {len(eval_points)}")

# ===== 实验1: 不同聚类数的基准 =====
print("\n" + "="*60)
print("实验1: 不同聚类数的影响 (hl=252, pw=0.3)")
print("="*60)

cluster_counts = [8, 10, 15, 20, 25, 30, 35, 40, 42]
cluster_results = []

# 使用全样本静态相关性
corr_full = np.corrcoef(ret_vals.T)
adj_full = topk_adj(corr_full, k=5)
np.fill_diagonal(adj_full, 0)

for n_clusters in cluster_counts:
    nmi = eval_nmi(adj_full, stocks, code_to_industry, n_clusters=n_clusters)
    ic = industry_consistency(adj_full, stocks, code_to_industry)
    cluster_results.append({
        'n_clusters': n_clusters,
        'nmi': nmi,
        'ic': ic,
    })
    print(f"n_clusters={n_clusters:2d}: NMI={nmi:.4f}, IC={ic:.4f}")

# ===== 实验2: 扩展先验权重范围 =====
print("\n" + "="*60)
print("实验2: 扩展先验权重范围 (hl=252, n_clusters=42)")
print("="*60)

prior_weights_ext = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
extended_results = []

for pw in prior_weights_ext:
    est = AdaptiveCovEstimator(n_stocks, half_life=252, prior_weight=pw, prior_matrix=industry_prior_uniform)
    
    # Warmup
    for t in range(0, WARMUP, UPDATE_FREQ):
        est.update(ret_vals[t:t+UPDATE_FREQ])
    
    cov_errors = []
    nmis = []
    
    for et in eval_points:
        for t in range(max(WARMUP, et - UPDATE_FREQ * 5), et, UPDATE_FREQ):
            batch = ret_vals[t:t+UPDATE_FREQ]
            if len(batch) > 0:
                est.update(batch)
        
        future = ret_vals[et:et+FORECAST]
        cov_true = np.cov(future.T)
        cov_pred = est.get_cov()
        rel_err = np.linalg.norm(cov_pred - cov_true, 'fro') / np.linalg.norm(cov_true, 'fro')
        cov_errors.append(rel_err)
        
        corr = est.get_corr()
        adj = topk_adj(corr, k=5)
        np.fill_diagonal(adj, 0)
        nmi = eval_nmi(adj, stocks, code_to_industry, n_clusters=42)
        nmis.append(nmi)
    
    avg_err = np.mean(cov_errors)
    avg_nmi = np.mean(nmis)
    extended_results.append({
        'prior_weight': pw,
        'avg_cov_error': avg_err,
        'avg_nmi': avg_nmi,
    })
    print(f"pw={pw:.2f}: cov_err={avg_err:.4f}, NMI={avg_nmi:.4f}")

# ===== 实验3: 行业规模非均匀先验 =====
print("\n" + "="*60)
print("实验3: 行业规模非均匀先验 (hl=252, pw=0.40)")
print("="*60)

prior_modes = ['uniform', 'size', 'sqrt_size']
mode_results = []
best_pw = 0.40  # 从实验2选最佳

for mode in prior_modes:
    if mode == 'uniform':
        prior_mat = industry_prior_uniform
    elif mode == 'size':
        prior_mat = industry_prior_size
    elif mode == 'sqrt_size':
        prior_mat = industry_prior_sqrt
    
    est = AdaptiveCovEstimator(n_stocks, half_life=252, prior_weight=best_pw, prior_matrix=prior_mat)
    
    # Warmup
    for t in range(0, WARMUP, UPDATE_FREQ):
        est.update(ret_vals[t:t+UPDATE_FREQ])
    
    cov_errors = []
    nmis = []
    
    for et in eval_points:
        for t in range(max(WARMUP, et - UPDATE_FREQ * 5), et, UPDATE_FREQ):
            batch = ret_vals[t:t+UPDATE_FREQ]
            if len(batch) > 0:
                est.update(batch)
        
        future = ret_vals[et:et+FORECAST]
        cov_true = np.cov(future.T)
        cov_pred = est.get_cov()
        rel_err = np.linalg.norm(cov_pred - cov_true, 'fro') / np.linalg.norm(cov_true, 'fro')
        cov_errors.append(rel_err)
        
        corr = est.get_corr()
        adj = topk_adj(corr, k=5)
        np.fill_diagonal(adj, 0)
        nmi = eval_nmi(adj, stocks, code_to_industry, n_clusters=42)
        nmis.append(nmi)
    
    avg_err = np.mean(cov_errors)
    avg_nmi = np.mean(nmis)
    mode_results.append({
        'prior_mode': mode,
        'prior_weight': best_pw,
        'avg_cov_error': avg_err,
        'avg_nmi': avg_nmi,
    })
    print(f"{mode}: cov_err={avg_err:.4f}, NMI={avg_nmi:.4f}")

# ===== 实验4: 最优组合搜索 =====
print("\n" + "="*60)
print("实验4: 最优组合 (聚类数 + 先验模式 + 先验权重)")
print("="*60)

best_combination_results = []

# 测试关键组合
combinations = [
    # (half_life, prior_weight, prior_mode, n_clusters)
    (252, 0.40, 'uniform', 42),
    (252, 0.45, 'uniform', 42),
    (252, 0.50, 'uniform', 42),
    (252, 0.40, 'size', 42),
    (252, 0.40, 'sqrt_size', 42),
    (189, 0.45, 'uniform', 42),
    (315, 0.40, 'uniform', 42),  # 更长的半衰期
]

for hl, pw, mode, nc in combinations:
    if mode == 'uniform':
        prior_mat = industry_prior_uniform
    elif mode == 'size':
        prior_mat = industry_prior_size
    elif mode == 'sqrt_size':
        prior_mat = industry_prior_sqrt
    
    est = AdaptiveCovEstimator(n_stocks, half_life=hl, prior_weight=pw, prior_matrix=prior_mat)
    
    # Warmup
    for t in range(0, WARMUP, UPDATE_FREQ):
        est.update(ret_vals[t:t+UPDATE_FREQ])
    
    cov_errors = []
    nmis = []
    
    for et in eval_points:
        for t in range(max(WARMUP, et - UPDATE_FREQ * 5), et, UPDATE_FREQ):
            batch = ret_vals[t:t+UPDATE_FREQ]
            if len(batch) > 0:
                est.update(batch)
        
        future = ret_vals[et:et+FORECAST]
        cov_true = np.cov(future.T)
        cov_pred = est.get_cov()
        rel_err = np.linalg.norm(cov_pred - cov_true, 'fro') / np.linalg.norm(cov_true, 'fro')
        cov_errors.append(rel_err)
        
        corr = est.get_corr()
        adj = topk_adj(corr, k=5)
        np.fill_diagonal(adj, 0)
        nmi = eval_nmi(adj, stocks, code_to_industry, n_clusters=nc)
        nmis.append(nmi)
    
    avg_err = np.mean(cov_errors)
    avg_nmi = np.mean(nmis)
    best_combination_results.append({
        'half_life': hl,
        'prior_weight': pw,
        'prior_mode': mode,
        'n_clusters': nc,
        'avg_cov_error': avg_err,
        'avg_nmi': avg_nmi,
    })
    print(f"hl={hl}, pw={pw:.2f}, mode={mode}, nc={nc}: cov_err={avg_err:.4f}, NMI={avg_nmi:.4f}")

# ===== 保存结果 =====
all_results = {
    'experiment': 'Exp8: 改进方向探索',
    'cluster_experiment': cluster_results,
    'extended_prior_weight': extended_results,
    'prior_mode_experiment': mode_results,
    'best_combinations': best_combination_results,
}

with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

pd.DataFrame(cluster_results).to_csv(f'{OUT_DIR}/cluster_experiment.csv', index=False)
pd.DataFrame(extended_results).to_csv(f'{OUT_DIR}/extended_prior_weight.csv', index=False)
pd.DataFrame(mode_results).to_csv(f'{OUT_DIR}/prior_mode_experiment.csv', index=False)
pd.DataFrame(best_combination_results).to_csv(f'{OUT_DIR}/best_combinations.csv', index=False)

# 找出最佳组合
best_row = max(best_combination_results, key=lambda x: x['avg_nmi'])
print(f"\n=== 最佳组合 ===")
print(f"半衰期: {best_row['half_life']}")
print(f"先验权重: {best_row['prior_weight']}")
print(f"先验模式: {best_row['prior_mode']}")
print(f"聚类数: {best_row['n_clusters']}")
print(f"NMI: {best_row['avg_nmi']:.4f}")
print(f"Cov Error: {best_row['avg_cov_error']:.4f}")

print("\n实验8完成!")
print(f"基准NMI: 0.7478 (hl=252, pw=0.3, nc=10)")
print(f"当前最佳: {best_row['avg_nmi']:.4f}")