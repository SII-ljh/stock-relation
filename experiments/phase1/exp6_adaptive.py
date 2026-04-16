"""
实验6: 自适应动态关系网络
核心思想: 用行业先验正则化的动态协方差估计
1. 行业先验: 同行业股票给更高初始关系权重
2. 动态更新: EWMA方式持续更新协方差
3. 收缩: Ledoit-Wolf收缩提高估计稳健性
4. 评估: 滚动预测精度、关系稳定性、下游任务
"""
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import normalized_mutual_info_score, r2_score
from sklearn.cluster import SpectralClustering
from scipy.optimize import minimize
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/exp6"
os.makedirs(OUT_DIR, exist_ok=True)

returns = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns.columns.tolist()
n_stocks = len(stocks)

print(f"股票数: {n_stocks}, 交易日: {len(returns)}")

# ===== 构建行业先验矩阵 =====
print("\n=== 构建行业先验 ===")
industry_prior = np.zeros((n_stocks, n_stocks))
for i in range(n_stocks):
    for j in range(n_stocks):
        ind_i = code_to_industry.get(stocks[i], 'X')
        ind_j = code_to_industry.get(stocks[j], 'Y')
        if ind_i == ind_j and ind_i != 'Unknown':
            industry_prior[i, j] = 1.0
        else:
            industry_prior[i, j] = 0.0
np.fill_diagonal(industry_prior, 0)
print(f"行业先验非零元素: {(industry_prior > 0).sum()}")

# ===== 自适应协方差估计 =====
class AdaptiveCovEstimator:
    """
    结合EWMA + Ledoit-Wolf收缩 + 行业先验的自适应协方差估计器
    """
    def __init__(self, n_features, half_life=63, prior_weight=0.1, shrinkage_target='lw'):
        self.n = n_features
        self.half_life = half_life
        self.decay = np.log(2) / half_life
        self.prior_weight = prior_weight
        self.cov_ewma = None
        self.mean_ewma = None
        self.count = 0
        
    def update(self, returns_batch):
        """用一批新收益率更新协方差估计"""
        batch_cov = np.cov(returns_batch.T)
        batch_mean = returns_batch.mean(axis=0)
        
        if self.cov_ewma is None:
            self.cov_ewma = batch_cov
            self.mean_ewma = batch_mean
        else:
            alpha = 1 - np.exp(-self.decay * len(returns_batch))
            self.cov_ewma = (1 - alpha) * self.cov_ewma + alpha * batch_cov
            self.mean_ewma = (1 - alpha) * self.mean_ewma + alpha * batch_mean
        self.count += len(returns_batch)
        
    def get_covariance(self, industry_prior=None):
        """获取正则化协方差估计"""
        if self.cov_ewma is None:
            return np.eye(self.n) * 0.001
        
        cov = self.cov_ewma.copy()
        
        # Ledoit-Wolf收缩
        target = np.diag(np.diag(cov))  # 对角目标
        n = self.count if self.count > 0 else 100
        # 简化的收缩系数
        shrinkage = min(1.0, max(0.0, self.n / n))
        cov = (1 - shrinkage) * cov + shrinkage * target
        
        # 行业先验正则化
        if industry_prior is not None and self.prior_weight > 0:
            # 同行业的方差均值作为先验协方差
            avg_var = np.diag(cov).mean()
            prior_cov = industry_prior * avg_var * 0.5  # 同行业给50%平均方差的协方差
            np.fill_diagonal(prior_cov, 0)
            cov = (1 - self.prior_weight) * cov + self.prior_weight * (cov + prior_cov)
        
        # 确保正定
        cov = (cov + cov.T) / 2
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 1e-6:
            cov += np.eye(self.n) * (1e-6 - eigvals.min())
        
        return cov
    
    def get_correlation(self, **kwargs):
        cov = self.get_covariance(**kwargs)
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-10
        corr = cov / np.outer(std, std)
        np.fill_diagonal(corr, 1)
        return np.clip(corr, -1, 1)

# ===== 滚动评估 =====
print("\n=== 滚动评估 ===")

WARMUP = 250
UPDATE_FREQ = 20  # 每20天更新一次
EVAL_FREQ = 60    # 每60天评估一次
FORECAST_WINDOW = 60

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

# 比较多种方法
methods = {
    'adaptive_hl63_prior0.1': AdaptiveCovEstimator(n_stocks, half_life=63, prior_weight=0.1),
    'adaptive_hl63_prior0.0': AdaptiveCovEstimator(n_stocks, half_life=63, prior_weight=0.0),
    'adaptive_hl126_prior0.1': AdaptiveCovEstimator(n_stocks, half_life=126, prior_weight=0.1),
    'adaptive_hl21_prior0.1': AdaptiveCovEstimator(n_stocks, half_life=21, prior_weight=0.1),
}

ret_vals = returns.values
eval_results = {m: {'cov_errors': [], 'nmis': [], 'ics': []} for m in methods}
eval_results['static_pearson'] = {'cov_errors': [], 'nmis': [], 'ics': []}
eval_results['ledoit_wolf'] = {'cov_errors': [], 'nmis': [], 'ics': []}

# 初始化
for t in range(0, WARMUP, UPDATE_FREQ):
    batch = ret_vals[t:t+UPDATE_FREQ]
    for m in methods.values():
        m.update(batch)

# 滚动评估
eval_points = list(range(WARMUP, len(ret_vals) - FORECAST_WINDOW, EVAL_FREQ))
print(f"评估点: {len(eval_points)}")

for eval_t in eval_points:
    # 更新到当前时刻
    for t in range(max(WARMUP, eval_t - UPDATE_FREQ * 3), eval_t, UPDATE_FREQ):
        batch = ret_vals[t:t+UPDATE_FREQ]
        if len(batch) > 0:
            for m in methods.values():
                m.update(batch)
    
    # 真实未来协方差
    future = ret_vals[eval_t:eval_t+FORECAST_WINDOW]
    cov_true = np.cov(future.T)
    frob_true = np.linalg.norm(cov_true, 'fro')
    
    for name, estimator in methods.items():
        cov_pred = estimator.get_covariance(industry_prior=industry_prior)
        rel_err = np.linalg.norm(cov_pred - cov_true, 'fro') / frob_true
        eval_results[name]['cov_errors'].append(rel_err)
        
        corr = estimator.get_correlation(industry_prior=industry_prior)
        adj = topk_adj(corr, k=5)
        np.fill_diagonal(adj, 0)
        nmi = eval_nmi(adj, stocks, code_to_industry)
        eval_results[name]['nmis'].append(nmi)
    
    # 静态基准
    full_cov = np.cov(ret_vals[:eval_t].T)
    rel_err_static = np.linalg.norm(full_cov - cov_true, 'fro') / frob_true
    eval_results['static_pearson']['cov_errors'].append(rel_err_static)
    
    full_corr = np.corrcoef(ret_vals[:eval_t].T)
    adj_static = topk_adj(full_corr, k=5)
    nmi_static = eval_nmi(adj_static, stocks, code_to_industry)
    eval_results['static_pearson']['nmis'].append(nmi_static)
    
    # LedoitWolf基准
    window = ret_vals[max(0, eval_t-250):eval_t]
    lw = LedoitWolf()
    lw.fit(window)
    rel_err_lw = np.linalg.norm(lw.covariance_ - cov_true, 'fro') / frob_true
    eval_results['ledoit_wolf']['cov_errors'].append(rel_err_lw)

# 打印结果
print("\n=== 结果汇总 ===")
summary_table = []
for name, res in eval_results.items():
    row = {
        'method': name,
        'avg_cov_error': np.mean(res['cov_errors']) if res['cov_errors'] else None,
        'std_cov_error': np.std(res['cov_errors']) if res['cov_errors'] else None,
        'avg_nmi': np.mean(res['nmis']) if res['nmis'] else None,
    }
    summary_table.append(row)
    if row['avg_cov_error'] is not None:
        print(f"{name}: cov_err={row['avg_cov_error']:.4f}±{row['std_cov_error']:.4f}, nmi={row['avg_nmi']:.4f}" if row['avg_nmi'] else f"{name}: cov_err={row['avg_cov_error']:.4f}")

# ===== 最小方差投资组合对比 =====
print("\n=== 投资组合对比 ===")

def min_var_weights(cov, max_weight=0.05):
    n = cov.shape[0]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n
    w0 = np.ones(n) / n
    result = minimize(lambda w: w @ cov @ w, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints, options={'maxiter': 300})
    return result.x if result.success else w0

# 重新计算投资组合
REBALANCE = 60
port_results = {}

for method_name in ['adaptive_hl63_prior0.1', 'static_pearson', 'ledoit_wolf']:
    estimator = methods.get(method_name)
    realized_rets = []
    
    for t in range(WARMUP, len(ret_vals) - REBALANCE, REBALANCE):
        if method_name == 'adaptive_hl63_prior0.1':
            cov = estimator.get_covariance(industry_prior=industry_prior)
        elif method_name == 'static_pearson':
            cov = np.cov(ret_vals[:t].T)
        elif method_name == 'ledoit_wolf':
            lw = LedoitWolf()
            lw.fit(ret_vals[max(0,t-250):t])
            cov = lw.covariance_
        
        cov += np.eye(n_stocks) * 1e-6
        w = min_var_weights(cov)
        
        period_rets = ret_vals[t:t+REBALANCE] @ w
        realized_rets.extend(period_rets.tolist())
    
    r = np.array(realized_rets)
    ann_ret = np.mean(r) * 252
    ann_vol = np.std(r) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    port_results[method_name] = {
        'return': float(ann_ret), 'volatility': float(ann_vol), 'sharpe': float(sharpe)
    }
    print(f"{method_name}: ret={ann_ret:.4f}, vol={ann_vol:.4f}, sharpe={sharpe:.4f}")

# 等权
eq_ret = ret_vals[WARMUP:].mean(axis=1)
print(f"equal_weight: ret={eq_ret.mean()*252:.4f}, vol={eq_ret.std()*np.sqrt(252):.4f}, sharpe={eq_ret.mean()/eq_ret.std()*np.sqrt(252):.4f}")

# 保存
summary = {
    'experiment': 'Exp6: Adaptive Dynamic Relation Network',
    'summary_table': summary_table,
    'portfolio_results': port_results,
}
with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

pd.DataFrame(summary_table).to_csv(f'{OUT_DIR}/results.csv', index=False)

print("\n实验6完成!")
