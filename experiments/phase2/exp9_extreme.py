"""
实验9: 进一步探索更大先验权重和其他改进方向
"""
import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/exp9"
os.makedirs(OUT_DIR, exist_ok=True)

returns = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns.columns.tolist()
n_stocks = len(stocks)
ret_vals = returns.values

print(f"股票数: {n_stocks}")

# 构建行业先验
def build_industry_prior(stocks, code_to_industry):
    n = len(stocks)
    prior = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if code_to_industry.get(stocks[i], 'X') == code_to_industry.get(stocks[j], 'Y') and \
               code_to_industry.get(stocks[i], 'X') != 'Unknown':
                prior[i, j] = 1.0
    np.fill_diagonal(prior, 0)
    return prior

industry_prior = build_industry_prior(stocks, code_to_industry)

def topk_adj(corr, k=5):
    n = corr.shape[0]
    adj = np.zeros_like(corr)
    for i in range(n):
        row = np.abs(corr[i].copy())
        row[i] = -np.inf
        top = np.argsort(row)[-k:]
        adj[i, top] = 1
    return ((adj + adj.T) > 0).astype(float)

def eval_nmi(adj, stocks, code_to_industry, n_clusters=42):
    try:
        abs_adj = np.abs(adj).astype(float)
        np.fill_diagonal(abs_adj, abs_adj.max())
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                               random_state=42, n_init=3)
        pred = sc.fit_predict(abs_adj)
        unique_inds = sorted(set(code_to_industry.values()))
        ind_map = {ind: i for i, ind in enumerate(unique_inds)}
        true_labels = [ind_map.get(code_to_industry.get(s, 'Unknown'), 0) for s in stocks]
        return normalized_mutual_info_score(true_labels, pred)
    except:
        return 0.0

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

# 实验设置
WARMUP = 250
UPDATE_FREQ = 20
EVAL_FREQ = 60
FORECAST = 60

eval_points = list(range(WARMUP, len(ret_vals) - FORECAST, EVAL_FREQ))
print(f"评估点: {len(eval_points)}")

# ===== 实验1: 扩展先验权重到0.6-1.0 =====
print("\n" + "="*60)
print("实验1: 扩展先验权重范围 0.6-1.0 (hl=252, nc=42)")
print("="*60)

prior_weights_ext = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]
extreme_results = []

for pw in prior_weights_ext:
    est = AdaptiveCovEstimator(n_stocks, half_life=252, prior_weight=pw, prior_matrix=industry_prior)
    
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
    extreme_results.append({
        'prior_weight': pw,
        'avg_cov_error': avg_err,
        'avg_nmi': avg_nmi,
    })
    print(f"pw={pw:.2f}: cov_err={avg_err:.4f}, NMI={avg_nmi:.4f}")

# ===== 实验2: 探索更长的半衰期 =====
print("\n" + "="*60)
print("实验2: 更长的半衰期 (pw=0.5, nc=42)")
print("="*60)

half_lives = [252, 300, 350, 400, 450, 500, 600]
hl_results = []

for hl in half_lives:
    est = AdaptiveCovEstimator(n_stocks, half_life=hl, prior_weight=0.5, prior_matrix=industry_prior)
    
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
    hl_results.append({
        'half_life': hl,
        'avg_cov_error': avg_err,
        'avg_nmi': avg_nmi,
    })
    print(f"hl={hl}: cov_err={avg_err:.4f}, NMI={avg_nmi:.4f}")

# ===== 实验3: 更小的TopK =====
print("\n" + "="*60)
print("实验3: 更小的TopK (hl=252, pw=0.5, nc=42)")
print("="*60)

topk_values = [3, 4, 5, 6, 7, 8, 10]
topk_results = []

for k in topk_values:
    est = AdaptiveCovEstimator(n_stocks, half_life=252, prior_weight=0.5, prior_matrix=industry_prior)
    
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
        adj = topk_adj(corr, k=k)
        np.fill_diagonal(adj, 0)
        nmi = eval_nmi(adj, stocks, code_to_industry, n_clusters=42)
        nmis.append(nmi)
    
    avg_err = np.mean(cov_errors)
    avg_nmi = np.mean(nmis)
    topk_results.append({
        'topk': k,
        'avg_cov_error': avg_err,
        'avg_nmi': avg_nmi,
    })
    print(f"k={k}: cov_err={avg_err:.4f}, NMI={avg_nmi:.4f}")

# ===== 实验4: 不同聚类数精细搜索 =====
print("\n" + "="*60)
print("实验4: 聚类数精细搜索 (hl=252, pw=0.5)")
print("="*60)

cluster_counts = [30, 35, 38, 40, 42, 44, 46, 50]
cluster_results = []

for nc in cluster_counts:
    est = AdaptiveCovEstimator(n_stocks, half_life=252, prior_weight=0.5, prior_matrix=industry_prior)
    
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
    cluster_results.append({
        'n_clusters': nc,
        'avg_cov_error': avg_err,
        'avg_nmi': avg_nmi,
    })
    print(f"nc={nc}: cov_err={avg_err:.4f}, NMI={avg_nmi:.4f}")

# ===== 网格搜索最优组合 =====
print("\n" + "="*60)
print("实验5: 精细网格搜索 (pw=0.4-0.7, hl=200-350)")
print("="*60)

grid_results = []
half_lives_grid = [200, 252, 300, 350]
prior_weights_grid = [0.40, 0.50, 0.60, 0.70]

for hl in half_lives_grid:
    for pw in prior_weights_grid:
        est = AdaptiveCovEstimator(n_stocks, half_life=hl, prior_weight=pw, prior_matrix=industry_prior)
        
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
        grid_results.append({
            'half_life': hl,
            'prior_weight': pw,
            'avg_cov_error': avg_err,
            'avg_nmi': avg_nmi,
        })
        print(f"hl={hl}, pw={pw:.2f}: cov_err={avg_err:.4f}, NMI={avg_nmi:.4f}")

# 保存结果
all_results = {
    'experiment': 'Exp9: 进一步探索',
    'extreme_prior_weights': extreme_results,
    'half_life_experiment': hl_results,
    'topk_experiment': topk_results,
    'cluster_fine_search': cluster_results,
    'grid_search': grid_results,
}

with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

pd.DataFrame(extreme_results).to_csv(f'{OUT_DIR}/extreme_prior_weights.csv', index=False)
pd.DataFrame(hl_results).to_csv(f'{OUT_DIR}/half_life_experiment.csv', index=False)
pd.DataFrame(topk_results).to_csv(f'{OUT_DIR}/topk_experiment.csv', index=False)
pd.DataFrame(cluster_results).to_csv(f'{OUT_DIR}/cluster_fine_search.csv', index=False)
pd.DataFrame(grid_results).to_csv(f'{OUT_DIR}/grid_search.csv', index=False)

# 找出最佳
best_row = max(grid_results, key=lambda x: x['avg_nmi'])
print(f"\n=== 网格搜索最佳 ===")
print(f"半衰期: {best_row['half_life']}")
print(f"先验权重: {best_row['prior_weight']}")
print(f"NMI: {best_row['avg_nmi']:.4f}")
print(f"Cov Error: {best_row['avg_cov_error']:.4f}")

best_extreme = max(extreme_results, key=lambda x: x['avg_nmi'])
print(f"\n=== 扩展先验权重最佳 ===")
print(f"先验权重: {best_extreme['prior_weight']}")
print(f"NMI: {best_extreme['avg_nmi']:.4f}")

print("\n实验9完成!")
print(f"基准NMI: 0.7478 (旧: hl=252, pw=0.3, nc=10)")
print(f"Exp8最佳: 0.8801 (hl=252, pw=0.5, nc=42)")
print(f"当前最佳: {max(best_row['avg_nmi'], best_extreme['avg_nmi']):.4f}")