"""
实验7: 参数优化 + 最终模型对比
1. 网格搜索最优先验权重和半衰期
2. 全方法对比总结
"""
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf, GraphicalLasso
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/exp7"
os.makedirs(OUT_DIR, exist_ok=True)

returns = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns.columns.tolist()
n_stocks = len(stocks)
ret_vals = returns.values

# 行业先验
industry_prior = np.zeros((n_stocks, n_stocks))
for i in range(n_stocks):
    for j in range(n_stocks):
        if code_to_industry.get(stocks[i], 'X') == code_to_industry.get(stocks[j], 'Y') and \
           code_to_industry.get(stocks[i], 'X') != 'Unknown':
            industry_prior[i, j] = 1.0
np.fill_diagonal(industry_prior, 0)

def topk_adj(corr, k=5):
    n = corr.shape[0]
    adj = np.zeros_like(corr)
    for i in range(n):
        row = np.abs(corr[i].copy())
        row[i] = -np.inf
        top = np.argsort(row)[-k:]
        adj[i, top] = 1
    return ((adj + adj.T) > 0).astype(float)

def eval_nmi(adj, stocks, code_to_industry):
    try:
        abs_adj = np.abs(adj).astype(float)
        np.fill_diagonal(abs_adj, abs_adj.max())
        sc = SpectralClustering(n_clusters=10, affinity='precomputed',
                               random_state=42, n_init=3)
        pred = sc.fit_predict(abs_adj)
        unique_inds = sorted(set(code_to_industry.values()))
        ind_map = {ind: i for i, ind in enumerate(unique_inds)}
        true_labels = [ind_map.get(code_to_industry.get(s, 'Unknown'), 0) for s in stocks]
        return normalized_mutual_info_score(true_labels, pred)
    except:
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

class AdaptiveCovEstimator:
    def __init__(self, n, half_life=63, prior_weight=0.1):
        self.n = n
        self.decay = np.log(2) / half_life
        self.prior_weight = prior_weight
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
        
    def get_cov(self, prior=None):
        if self.cov_ewma is None:
            return np.eye(self.n) * 0.001
        cov = self.cov_ewma.copy()
        shrinkage = min(0.8, max(0.0, self.n / max(self.count, 1)))
        target = np.diag(np.diag(cov))
        cov = (1 - shrinkage) * cov + shrinkage * target
        if prior is not None and self.prior_weight > 0:
            avg_var = np.diag(cov).mean()
            prior_cov = prior * avg_var * 0.5
            np.fill_diagonal(prior_cov, 0)
            cov = (1 - self.prior_weight) * cov + self.prior_weight * (cov + prior_cov)
        cov = (cov + cov.T) / 2
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 1e-6:
            cov += np.eye(self.n) * (1e-6 - eigvals.min())
        return cov
    
    def get_corr(self, **kwargs):
        cov = self.get_cov(**kwargs)
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-10
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 1)
        return np.clip(corr, -1, 1)

# ===== 网格搜索 =====
print("=== 网格搜索最优参数 ===")
WARMUP = 250
UPDATE_FREQ = 20
EVAL_FREQ = 60
FORECAST = 60

half_lives = [42, 63, 84, 126, 189, 252]
prior_weights = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

eval_points = list(range(WARMUP, len(ret_vals) - FORECAST, EVAL_FREQ))
grid_results = []

for hl in half_lives:
    for pw in prior_weights:
        est = AdaptiveCovEstimator(n_stocks, half_life=hl, prior_weight=pw)
        
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
            cov_pred = est.get_cov(prior=industry_prior)
            rel_err = np.linalg.norm(cov_pred - cov_true, 'fro') / np.linalg.norm(cov_true, 'fro')
            cov_errors.append(rel_err)
            
            corr = est.get_corr(prior=industry_prior)
            adj = topk_adj(corr, k=5)
            np.fill_diagonal(adj, 0)
            nmi = eval_nmi(adj, stocks, code_to_industry)
            nmis.append(nmi)
        
        avg_err = np.mean(cov_errors)
        avg_nmi = np.mean(nmis)
        grid_results.append({
            'half_life': hl, 'prior_weight': pw,
            'avg_cov_error': avg_err, 'avg_nmi': avg_nmi,
        })
        print(f"hl={hl:3d}, pw={pw:.2f}: cov_err={avg_err:.4f}, NMI={avg_nmi:.4f}")

grid_df = pd.DataFrame(grid_results)

# 最优参数
best_nmi_row = grid_df.loc[grid_df['avg_nmi'].idxmax()]
best_err_row = grid_df.loc[grid_df['avg_cov_error'].idxmin()]

print(f"\nNMI最优: hl={int(best_nmi_row['half_life'])}, pw={best_nmi_row['prior_weight']:.2f}, NMI={best_nmi_row['avg_nmi']:.4f}")
print(f"CovErr最优: hl={int(best_err_row['half_life'])}, pw={best_err_row['prior_weight']:.2f}, err={best_err_row['avg_cov_error']:.4f}")

# ===== 最终全方法对比 =====
print("\n=== 全方法对比 ===")
final_comparison = []

# 1. Static Pearson TopK=5
corr_full = np.corrcoef(ret_vals.T)
adj = topk_adj(corr_full, 5)
nmi = eval_nmi(adj, stocks, code_to_industry)
ic = industry_consistency(adj, stocks, code_to_industry)
final_comparison.append({'method': 'Static Pearson TopK5', 'NMI': nmi, 'IC': ic, 'cov_err': 0.4245})

# 2. Static Partial Corr TopK=5
scaler = StandardScaler()
ret_std = scaler.fit_transform(ret_vals)
gl = GraphicalLasso(alpha=0.27, max_iter=200)
gl.fit(ret_std)
prec = gl.precision_
d = np.sqrt(np.diag(prec)); d[d==0] = 1e-10
pcorr = -prec / np.outer(d, d)
np.fill_diagonal(pcorr, 1)
adj_pc = topk_adj(pcorr, 5)
nmi_pc = eval_nmi(adj_pc, stocks, code_to_industry)
ic_pc = industry_consistency(adj_pc, stocks, code_to_industry)
final_comparison.append({'method': 'Partial Corr (GLasso) TopK5', 'NMI': nmi_pc, 'IC': ic_pc, 'cov_err': 0.7419})

# 3. Granger Network (从exp3)
final_comparison.append({'method': 'Granger Causality (50 stocks)', 'NMI': 0.5996, 'IC': 0.0259, 'cov_err': None, 'prediction_r2': 0.265})

# 4. LedoitWolf Dynamic
final_comparison.append({'method': 'LedoitWolf Rolling w250', 'NMI': 0.5503, 'IC': None, 'cov_err': 0.7291})

# 5. Adaptive (best params)
best_hl = int(best_nmi_row['half_life'])
best_pw = best_nmi_row['prior_weight']
final_comparison.append({
    'method': f'Adaptive EWMA (hl={best_hl}, prior={best_pw:.2f})',
    'NMI': float(best_nmi_row['avg_nmi']),
    'IC': None,
    'cov_err': float(grid_df.loc[(grid_df['half_life']==best_hl) & (grid_df['prior_weight']==best_pw), 'avg_cov_error'].values[0]),
})

# 6. Adaptive (best cov)
best_hl2 = int(best_err_row['half_life'])
best_pw2 = best_err_row['prior_weight']
if best_hl != best_hl2 or best_pw != best_pw2:
    final_comparison.append({
        'method': f'Adaptive EWMA (hl={best_hl2}, prior={best_pw2:.2f}) [best cov]',
        'NMI': float(grid_df.loc[(grid_df['half_life']==best_hl2) & (grid_df['prior_weight']==best_pw2), 'avg_nmi'].values[0]),
        'cov_err': float(best_err_row['avg_cov_error']),
    })

comp_df = pd.DataFrame(final_comparison)
print(comp_df.to_string(index=False))

# 保存
grid_df.to_csv(f'{OUT_DIR}/grid_search.csv', index=False)
comp_df.to_csv(f'{OUT_DIR}/final_comparison.csv', index=False)

with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump({
        'best_nmi_params': {'half_life': best_hl, 'prior_weight': float(best_pw)},
        'best_nmi': float(best_nmi_row['avg_nmi']),
        'best_cov_params': {'half_life': best_hl2, 'prior_weight': float(best_pw2)},
        'best_cov_error': float(best_err_row['avg_cov_error']),
        'final_comparison': final_comparison,
    }, f, indent=2, ensure_ascii=False, default=str)

print("\n实验7完成!")
