"""
实验10: 验证最优参数并探索新方向
"""
import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, r2_score
from sklearn.cluster import SpectralClustering
from scipy.optimize import minimize
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/exp10"
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

# ===== 实验1: 纯先验 vs 数据+先验 =====
print("\n" + "="*60)
print("实验1: 纯先验 vs 数据+先验对比")
print("="*60)

# 纯先验基准
corr_prior = industry_prior.copy()
np.fill_diagonal(corr_prior, 1)  # 对角线设为1
adj_prior = topk_adj(corr_prior, k=5)
np.fill_diagonal(adj_prior, 0)
nmi_prior = eval_nmi(adj_prior, stocks, code_to_industry, n_clusters=42)
print(f"纯先验(TopK=5): NMI={nmi_prior:.4f}")

# 不同TopK
for k in [3, 4, 5, 6, 8]:
    adj = topk_adj(corr_prior, k=k)
    np.fill_diagonal(adj, 0)
    nmi = eval_nmi(adj, stocks, code_to_industry, n_clusters=42)
    print(f"纯先验(TopK={k}): NMI={nmi:.4f}")

# ===== 实验2: 验证pw=1.0 vs pw=0.7 =====
print("\n" + "="*60)
print("实验2: 验证pw=1.0 vs pw=0.7 (不同聚类数)")
print("="*60)

results = []
prior_weights = [0.7, 0.8, 0.9, 1.0]
cluster_counts = [30, 35, 40, 42]

for pw in prior_weights:
    for nc in cluster_counts:
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
            nmi = eval_nmi(adj, stocks, code_to_industry, n_clusters=nc)
            nmis.append(nmi)
        
        avg_err = np.mean(cov_errors)
        avg_nmi = np.mean(nmis)
        results.append({
            'prior_weight': pw,
            'n_clusters': nc,
            'avg_cov_error': avg_err,
            'avg_nmi': avg_nmi,
        })
        print(f"pw={pw:.1f}, nc={nc}: cov_err={avg_err:.4f}, NMI={avg_nmi:.4f}")

# ===== 实验3: 更精细的TopK搜索 =====
print("\n" + "="*60)
print("实验3: 更精细的TopK搜索 (pw=0.7, nc=35)")
print("="*60)

topk_values = [2, 3, 4, 5, 6]
topk_results = []
pw = 0.7
nc = 35

for k in topk_values:
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
        adj = topk_adj(corr, k=k)
        np.fill_diagonal(adj, 0)
        nmi = eval_nmi(adj, stocks, code_to_industry, n_clusters=nc)
        nmis.append(nmi)
    
    avg_err = np.mean(cov_errors)
    avg_nmi = np.mean(nmis)
    topk_results.append({
        'topk': k,
        'avg_cov_error': avg_err,
        'avg_nmi': avg_nmi,
    })
    print(f"k={k}: cov_err={avg_err:.4f}, NMI={avg_nmi:.4f}")

# ===== 实验4: 投资组合验证 =====
print("\n" + "="*60)
print("实验4: 投资组合验证 (最佳参数)")
print("="*60)

def min_var_weights(cov, max_weight=0.05):
    n = cov.shape[0]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n
    w0 = np.ones(n) / n
    result = minimize(lambda w: w @ cov @ w, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints, options={'maxiter': 300})
    return result.x if result.success else w0

REBALANCE = 60
best_params = {'half_life': 252, 'prior_weight': 0.7, 'n_clusters': 35, 'topk': 3}

portfolios = {}
for method_name, params in [
    ('adaptive_best', {'half_life': 252, 'prior_weight': 0.7}),
    ('adaptive_pw1', {'half_life': 252, 'prior_weight': 1.0}),
    ('static_prior', None),
]:
    realized_rets = []
    
    for t in range(WARMUP, len(ret_vals) - REBALANCE, REBALANCE):
        if method_name == 'adaptive_best':
            est = AdaptiveCovEstimator(n_stocks, half_life=252, prior_weight=0.7, prior_matrix=industry_prior)
            for tt in range(0, t, UPDATE_FREQ):
                est.update(ret_vals[tt:tt+UPDATE_FREQ])
            cov = est.get_cov()
        elif method_name == 'adaptive_pw1':
            est = AdaptiveCovEstimator(n_stocks, half_life=252, prior_weight=1.0, prior_matrix=industry_prior)
            for tt in range(0, t, UPDATE_FREQ):
                est.update(ret_vals[tt:tt+UPDATE_FREQ])
            cov = est.get_cov()
        elif method_name == 'static_prior':
            cov = industry_prior.copy()
            np.fill_diagonal(cov, np.var(ret_vals[:t], axis=0).mean())
        
        cov += np.eye(n_stocks) * 1e-6
        w = min_var_weights(cov)
        
        period_rets = ret_vals[t:t+REBALANCE] @ w
        realized_rets.extend(period_rets.tolist())
    
    r = np.array(realized_rets)
    ann_ret = np.mean(r) * 252
    ann_vol = np.std(r) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    portfolios[method_name] = {
        'return': float(ann_ret), 
        'volatility': float(ann_vol), 
        'sharpe': float(sharpe)
    }
    print(f"{method_name}: ret={ann_ret:.4f}, vol={ann_vol:.4f}, sharpe={sharpe:.4f}")

# ===== 保存结果 =====
all_results = {
    'experiment': 'Exp10: 验证最优参数',
    'validation_results': results,
    'topk_results': topk_results,
    'portfolios': portfolios,
    'pure_prior_nmi': float(nmi_prior),
}

with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

pd.DataFrame(results).to_csv(f'{OUT_DIR}/validation_results.csv', index=False)
pd.DataFrame(topk_results).to_csv(f'{OUT_DIR}/topk_results.csv', index=False)

# 找出最佳
best_row = max(results, key=lambda x: x['avg_nmi'])
best_topk = max(topk_results, key=lambda x: x['avg_nmi'])

print(f"\n=== 最佳验证结果 ===")
print(f"先验权重: {best_row['prior_weight']}")
print(f"聚类数: {best_row['n_clusters']}")
print(f"NMI: {best_row['avg_nmi']:.4f}")
print(f"Cov Error: {best_row['avg_cov_error']:.4f}")

print(f"\n=== 最佳TopK ===")
print(f"TopK: {best_topk['topk']}")
print(f"NMI: {best_topk['avg_nmi']:.4f}")

print("\n实验10完成!")
print(f"基准NMI: 0.7478 (旧: hl=252, pw=0.3, nc=10)")
print(f"Exp8最佳: 0.8801 (hl=252, pw=0.5, nc=42)")
print(f"Exp9最佳: 0.8889 (hl=252, pw=1.0, nc=42)")
print(f"当前最佳: {max(best_row['avg_nmi'], best_topk['avg_nmi']):.4f}")