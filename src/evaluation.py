"""Walk-Forward 评估框架 (V2)

核心原则:
  - 在时刻 t 做评估时，只能使用 [0, t) 的历史数据
  - 用 [t, t+forecast) 的数据做样本外验证
  - 所有方法统一接口，无数据泄露
"""
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering
import networkx as nx

from src.network import topk_adj
from src.portfolio import min_var_weights, eval_portfolio_metrics
from src.utils import ensure_psd


# ============================================================
# 单点评估函数
# ============================================================

def _get_true_labels(stocks, code_to_industry):
    """获取真实行业标签"""
    unique = sorted(set(code_to_industry.values()))
    label_map = {ind: i for i, ind in enumerate(unique)}
    return [label_map.get(code_to_industry.get(s, 'Unknown'), 0) for s in stocks]


def spectral_cluster(adj, n_clusters=35):
    """对邻接矩阵做谱聚类

    Args:
        adj: N x N 邻接矩阵
        n_clusters: 聚类数

    Returns:
        labels: N 维聚类标签
    """
    abs_adj = np.abs(adj).astype(float)
    diag_val = abs_adj.max() if abs_adj.max() > 0 else 1.0
    np.fill_diagonal(abs_adj, diag_val)
    sc = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed',
        random_state=42, n_init=3
    )
    return sc.fit_predict(abs_adj)


def eval_nmi(adj, stocks, code_to_industry, n_clusters=35):
    """NMI: 聚类结果与真实行业的匹配度 (0~1, 越高越好)"""
    try:
        pred = spectral_cluster(adj, n_clusters)
        true_labels = _get_true_labels(stocks, code_to_industry)
        return normalized_mutual_info_score(true_labels, pred)
    except Exception:
        return 0.0


def eval_ari(adj, stocks, code_to_industry, n_clusters=35):
    """ARI: 调整兰德指数 (越高越好)"""
    try:
        pred = spectral_cluster(adj, n_clusters)
        true_labels = _get_true_labels(stocks, code_to_industry)
        return adjusted_rand_score(true_labels, pred)
    except Exception:
        return 0.0


def eval_modularity(adj, stocks, code_to_industry):
    """模块度: 网络社区结构质量"""
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
    """IC: 行业一致性 (同行业边 / 总边数)"""
    n = adj.shape[0]
    same_count, total_count = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                total_count += 1
                ind_i = code_to_industry.get(stocks[i], f'_x{i}')
                ind_j = code_to_industry.get(stocks[j], f'_y{j}')
                if ind_i == ind_j:
                    same_count += 1
    return same_count / max(total_count, 1)


def eval_cov_error(cov_pred, cov_true):
    """协方差预测误差 (相对 Frobenius 范数, 越低越好)"""
    return np.linalg.norm(cov_pred - cov_true, 'fro') / np.linalg.norm(cov_true, 'fro')


def eval_log_likelihood(cov_pred, returns_test):
    """样本外对数似然 (越高越好)"""
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
    """协方差元素的 Spearman 秩相关 (越高越好)"""
    from scipy.stats import spearmanr
    n = cov_pred.shape[0]
    idx = np.triu_indices(n, k=1)
    rho, _ = spearmanr(cov_pred[idx], cov_true[idx])
    return rho if not np.isnan(rho) else 0.0


# ============================================================
# Walk-Forward 评估器
# ============================================================

class FlexibleEvaluator:
    """Walk-Forward 评估器

    统一评估流程:
      1. 在每个评估时刻 t，用 [0, t) 的数据估计模型
      2. 用 [t, t+forecast) 的数据做样本外验证
      3. 投资组合回测严格只用历史数据

    Args:
        returns: T x N 收益率矩阵 (numpy array)
        stocks: 股票代码列表
        code_to_industry: {股票代码: 行业} 字典
        industry_prior: N x N 行业先验矩阵
        warmup: 最少训练天数（默认500天，约2年）
        eval_freq: 评估频率（默认60天）
        forecast: 预测窗口（默认60天）
        rebalance: 调仓频率（默认60天）
        n_clusters: 聚类数（默认35）
        adj_fn: 邻接矩阵构建函数 (corr -> adj)
    """

    def __init__(self, returns, stocks, code_to_industry, industry_prior,
                 warmup=500, eval_freq=60, forecast=60, rebalance=60,
                 n_clusters=35, adj_fn=None):
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
        self.adj_fn = adj_fn or (lambda c: topk_adj(c, k=5))
        self.eval_points = list(range(
            warmup, len(returns) - forecast, eval_freq
        ))

    def evaluate(self, name, estimator_fn):
        """评估一个方法

        Args:
            name: 方法名称
            estimator_fn: 估计器函数 (history -> (corr, cov))

        Returns:
            dict: 包含所有指标的字典
        """
        results = {'method': name}
        nmis, aris = [], []
        cov_errors, log_liks, rank_ics = [], [], []

        for t in self.eval_points:
            history = self.returns[:t]
            corr_est, cov_est = estimator_fn(history)

            # 网络聚类指标
            adj = self.adj_fn(corr_est)
            np.fill_diagonal(adj, 0)
            nmis.append(eval_nmi(adj, self.stocks, self.code_to_industry, self.n_clusters))
            aris.append(eval_ari(adj, self.stocks, self.code_to_industry, self.n_clusters))

            # 协方差质量（样本外）
            future = self.returns[t:t + self.forecast]
            if len(future) >= self.forecast:
                cov_true = np.cov(future.T)
                cov_errors.append(eval_cov_error(cov_est, cov_true))
                log_liks.append(eval_log_likelihood(cov_est, future))
                rank_ics.append(eval_rank_ic(cov_est, cov_true))

        results['NMI'] = np.mean(nmis) if nmis else 0.0
        results['ARI'] = np.mean(aris) if aris else 0.0
        results['NMI_Std'] = np.std(nmis) if len(nmis) > 1 else 0.0
        results['CovError'] = np.mean(cov_errors) if cov_errors else np.nan
        results['CovErr_Std'] = np.std(cov_errors) if len(cov_errors) > 1 else 0.0
        results['LogLik'] = np.mean(log_liks) if log_liks else np.nan
        results['RankIC'] = np.mean(rank_ics) if rank_ics else np.nan

        # 组合回测（严格 walk-forward）
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
# 辅助函数
# ============================================================

def compute_composite_score(df):
    """计算综合得分 (0-1 归一化后加权平均)

    权重: NMI 25%, Sharpe 15%, ARI 10%, Sortino 10%, RankIC 10%,
          CovError 10%(越低越好), IC 5%, Modularity 5%,
          MaxDD 5%(越低越好), Calmar 5%
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
            score += abs(w) * np.where(valid, 1 - normalized, 0.0)
        else:
            score += w * normalized
    return score


def format_results_table(results_list, sort_by='NMI'):
    """格式化结果表格"""
    df = pd.DataFrame(results_list)
    cols_order = ['method', 'NMI', 'ARI', 'Modularity', 'IC',
                  'CovError', 'LogLik', 'RankIC',
                  'Sharpe', 'Sortino', 'MaxDD', 'Calmar',
                  'NMI_Std', 'CovErr_Std']
    existing = [c for c in cols_order if c in df.columns]
    df = df[existing]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    return df
