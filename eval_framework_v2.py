"""
评估框架 V2 - 修复数据泄露问题
核心原则: 所有方法统一使用 walk-forward 评估
  - 在时刻t做评估时, 只能使用 [0, t) 的数据估计模型
  - 用 [t, t+forecast) 的数据做样本外验证
  - 没有任何静态/动态的区分, 所有方法都是rolling的

4个维度, 14个指标 (同V1):
  1. 网络结构: NMI, ARI, Modularity, IC
  2. 协方差质量: CovError, LogLik, RankIC
  3. 金融表现: Sharpe, Sortino, MaxDD, Calmar
  4. 鲁棒性: NMI_Std, CovErr_Std
  5. 综合: CompositeScore
"""
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering
from scipy.optimize import minimize
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 基础工具函数
# ============================================================

def topk_adj(corr, k=5):
    """TopK稀疏化邻接矩阵"""
    n = corr.shape[0]
    adj = np.zeros_like(corr)
    for i in range(n):
        row = np.abs(corr[i].copy())
        row[i] = -np.inf
        top = np.argsort(row)[-k:]
        adj[i, top] = 1
    return ((adj + adj.T) > 0).astype(float)


def cov_to_corr(cov):
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1e-10
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1)
    return np.clip(corr, -1, 1)


def ensure_psd(cov, eps=1e-6):
    cov = (cov + cov.T) / 2
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < eps:
        cov += np.eye(cov.shape[0]) * (eps - eigvals.min())
    return cov


def build_industry_prior(stocks, code_to_industry):
    """构建行业先验矩阵"""
    n = len(stocks)
    prior = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ind_i = code_to_industry.get(stocks[i], 'X')
            ind_j = code_to_industry.get(stocks[j], 'Y')
            if ind_i == ind_j and ind_i != 'Unknown':
                prior[i, j] = 1.0
    np.fill_diagonal(prior, 0)
    return prior


# ============================================================
# 单点评估函数
# ============================================================

def _get_labels(stocks, code_to_industry):
    unique_inds = sorted(set(code_to_industry.values()))
    ind_map = {ind: i for i, ind in enumerate(unique_inds)}
    true_labels = [ind_map.get(code_to_industry.get(s, 'Unknown'), 0) for s in stocks]
    return true_labels, ind_map


def eval_nmi(adj, stocks, code_to_industry, n_clusters=35):
    try:
        abs_adj = np.abs(adj).astype(float)
        np.fill_diagonal(abs_adj, abs_adj.max())
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                               random_state=42, n_init=3)
        pred = sc.fit_predict(abs_adj)
        true_labels, _ = _get_labels(stocks, code_to_industry)
        return normalized_mutual_info_score(true_labels, pred)
    except Exception:
        return 0.0


def eval_ari(adj, stocks, code_to_industry, n_clusters=35):
    try:
        abs_adj = np.abs(adj).astype(float)
        np.fill_diagonal(abs_adj, abs_adj.max())
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                               random_state=42, n_init=3)
        pred = sc.fit_predict(abs_adj)
        true_labels, _ = _get_labels(stocks, code_to_industry)
        return adjusted_rand_score(true_labels, pred)
    except Exception:
        return 0.0


def eval_modularity(adj, stocks, code_to_industry):
    try:
        G = nx.from_numpy_array(adj)
        if G.number_of_edges() == 0:
            return 0.0
        communities = {}
        for idx, s in enumerate(stocks):
            ind = code_to_industry.get(s, 'Unknown')
            if ind not in communities:
                communities[ind] = set()
            communities[ind].add(idx)
        return nx.community.modularity(G, list(communities.values()))
    except Exception:
        return 0.0


def eval_ic(adj, stocks, code_to_industry):
    n = adj.shape[0]
    same_count = 0
    total_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                total_count += 1
                if code_to_industry.get(stocks[i], 'X') == code_to_industry.get(stocks[j], 'Y'):
                    same_count += 1
    return same_count / max(total_count, 1)


def eval_cov_error(cov_pred, cov_true):
    return np.linalg.norm(cov_pred - cov_true, 'fro') / np.linalg.norm(cov_true, 'fro')


def eval_log_likelihood(cov_pred, returns_test):
    n = cov_pred.shape[0]
    try:
        cov = ensure_psd(cov_pred.copy(), 1e-8)
        sign, log_det = np.linalg.slogdet(cov)
        if sign <= 0:
            return -np.inf
        cov_inv = np.linalg.inv(cov)
        S = np.cov(returns_test.T)
        return -0.5 * (n * np.log(2 * np.pi) + log_det + np.trace(cov_inv @ S))
    except Exception:
        return -np.inf


def eval_rank_ic(cov_pred, cov_true):
    from scipy.stats import spearmanr
    n = cov_pred.shape[0]
    idx = np.triu_indices(n, k=1)
    pred_flat = cov_pred[idx]
    true_flat = cov_true[idx]
    rho, _ = spearmanr(pred_flat, true_flat)
    return rho if not np.isnan(rho) else 0.0


def min_var_weights(cov, max_weight=0.05):
    n = cov.shape[0]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n
    w0 = np.ones(n) / n
    result = minimize(lambda w: w @ cov @ w, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 300})
    return result.x if result.success else w0


def eval_portfolio_metrics(realized_returns):
    r = np.array(realized_returns)
    if len(r) < 10:
        return {'sharpe': 0.0, 'sortino': 0.0, 'max_drawdown': 0.0, 'calmar': 0.0}
    ann_ret = np.mean(r) * 252
    ann_vol = np.std(r) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    downside = r[r < 0]
    downside_std = np.std(downside) * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = ann_ret / downside_std if downside_std > 0 else 0.0
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    drawdowns = (cum - peak) / peak
    max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
    calmar = ann_ret / max_dd if max_dd > 0 else 0.0
    return {'sharpe': float(sharpe), 'sortino': float(sortino),
            'max_drawdown': float(max_dd), 'calmar': float(calmar)}


# ============================================================
# Walk-Forward 评估器 (V2核心)
# ============================================================

class WalkForwardEvaluator:
    """
    Walk-Forward 评估器
    
    所有方法统一使用同一套评估流程:
      1. 在每个评估时刻 t, 用 [0, t) 的数据估计协方差/相关矩阵
      2. 用 [t, t+forecast) 的数据计算样本外指标
      3. 投资组合回测也严格只用过去数据
    
    estimator_fn(returns_history) -> (corr_matrix, cov_matrix)
      接收历史数据, 返回估计的相关矩阵和协方差矩阵
      这是唯一的模型接口, 不区分静态/动态
    """
    
    def __init__(self, returns, stocks, code_to_industry, industry_prior,
                 warmup=500, eval_freq=60, forecast=60,
                 rebalance=60, n_clusters=35, topk=4):
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
        self.topk = topk
        
        # 评估时刻: 从warmup开始, 每eval_freq天评估一次
        self.eval_points = list(range(
            warmup, len(returns) - forecast, eval_freq
        ))
    
    def evaluate(self, name, estimator_fn):
        """
        统一评估接口
        
        estimator_fn: callable(returns_history: np.ndarray) -> (corr, cov)
            输入: 到当前时刻为止的历史收益率 (T_hist x N)
            输出: (相关矩阵 NxN, 协方差矩阵 NxN)
        """
        results = {'method': name}
        
        nmis, aris, mods, ics = [], [], [], []
        cov_errors, log_liks, rank_ics = [], [], []
        
        for t in self.eval_points:
            # ===== 严格只用 [0, t) 的数据 =====
            history = self.returns[:t]
            corr_est, cov_est = estimator_fn(history)
            
            # --- 网络指标 ---
            adj = topk_adj(corr_est, k=self.topk)
            np.fill_diagonal(adj, 0)
            nmis.append(eval_nmi(adj, self.stocks, self.code_to_industry, self.n_clusters))
            aris.append(eval_ari(adj, self.stocks, self.code_to_industry, self.n_clusters))
            # Modularity 和 IC 计算较慢, 只在部分点计算
            if t == self.eval_points[-1]:
                mods.append(eval_modularity(adj, self.stocks, self.code_to_industry))
                ics.append(eval_ic(adj, self.stocks, self.code_to_industry))
            
            # --- 协方差指标 (样本外) ---
            future = self.returns[t:t + self.forecast]
            if len(future) >= self.forecast:
                cov_true = np.cov(future.T)
                cov_errors.append(eval_cov_error(cov_est, cov_true))
                log_liks.append(eval_log_likelihood(cov_est, future))
                rank_ics.append(eval_rank_ic(cov_est, cov_true))
        
        # 聚合
        results['NMI'] = np.mean(nmis) if nmis else 0.0
        results['ARI'] = np.mean(aris) if aris else 0.0
        results['Modularity'] = np.mean(mods) if mods else 0.0
        results['IC'] = np.mean(ics) if ics else 0.0
        results['NMI_Std'] = np.std(nmis) if len(nmis) > 1 else 0.0
        
        results['CovError'] = np.mean(cov_errors) if cov_errors else np.nan
        results['CovErr_Std'] = np.std(cov_errors) if len(cov_errors) > 1 else 0.0
        results['LogLik'] = np.mean(log_liks) if log_liks else np.nan
        results['RankIC'] = np.mean(rank_ics) if rank_ics else np.nan
        
        # --- 投资组合回测 (严格walk-forward) ---
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
# 辅助: 排名和综合得分
# ============================================================

def format_results_table(results_list, sort_by='NMI', ascending=False):
    df = pd.DataFrame(results_list)
    cols_order = ['method', 'NMI', 'ARI', 'Modularity', 'IC',
                  'CovError', 'LogLik', 'RankIC',
                  'Sharpe', 'Sortino', 'MaxDD', 'Calmar',
                  'NMI_Std', 'CovErr_Std']
    existing_cols = [c for c in cols_order if c in df.columns]
    df = df[existing_cols]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    return df


def compute_composite_score(df):
    """
    综合得分 (0-1归一化后加权平均)
    权重: NMI(25%), ARI(10%), Modularity(5%), IC(5%),
          CovError(10%↓), RankIC(10%),
          Sharpe(15%), Sortino(10%), MaxDD(5%↓), Calmar(5%)
    """
    weights = {
        'NMI': 0.25, 'ARI': 0.10, 'Modularity': 0.05, 'IC': 0.05,
        'CovError': -0.10, 'RankIC': 0.10,
        'Sharpe': 0.15, 'Sortino': 0.10, 'MaxDD': -0.05, 'Calmar': 0.05,
    }
    score = np.zeros(len(df))
    for col, w in weights.items():
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        valid = ~np.isnan(vals) & ~np.isinf(vals)
        if valid.sum() == 0:
            continue
        vmin, vmax = vals[valid].min(), vals[valid].max()
        if vmax - vmin < 1e-10:
            normalized = np.where(valid, 0.5, 0.0)
        else:
            normalized = np.where(valid, (vals - vmin) / (vmax - vmin), 0.0)
        if w < 0:
            normalized = np.where(valid, 1 - normalized, 0.0)
            score += abs(w) * normalized
        else:
            score += w * normalized
    return score


if __name__ == '__main__':
    print("评估框架 V2 (Walk-Forward)")
    print("修复: 所有方法统一使用 walk-forward, 无数据泄露")
    print("接口: estimator_fn(returns_history) -> (corr, cov)")
