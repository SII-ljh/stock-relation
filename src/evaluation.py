"""Walk-Forward 评估框架 (V2)

核心原则:
  - 在时刻 t 做评估时，只能使用 [0, t) 的历史数据
  - 用 [t, t+forecast) 的数据做样本外验证
  - 所有方法统一接口；参数冻结与点时标签由调用方负责
"""
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering
import networkx as nx

from src.data import normalize_industry
from src.network import topk_adj
from src.portfolio import min_var_weights, eval_portfolio_metrics
from src.utils import ensure_psd, validate_square_matrix


# ============================================================
# 单点评估函数
# ============================================================

def _get_true_labels(stocks, code_to_industry):
    """获取真实行业标签"""
    normalized = [normalize_industry(code_to_industry.get(stock)) for stock in stocks]
    unique = sorted({label for label in normalized if label is not None})
    label_map = {ind: i for i, ind in enumerate(unique)}
    # -1 explicitly marks an unknown label.  It must not be conflated with
    # the first alphabetically sorted industry.
    return [label_map.get(label, -1) for label in normalized]


def spectral_cluster(adj, n_clusters=35):
    """对邻接矩阵做谱聚类

    Args:
        adj: N x N 邻接矩阵
        n_clusters: 聚类数

    Returns:
        labels: N 维聚类标签
    """
    adj = validate_square_matrix(adj, "adjacency")
    n = adj.shape[0]
    if not isinstance(n_clusters, (int, np.integer)) or isinstance(n_clusters, bool):
        raise TypeError("n_clusters must be an integer")
    if n_clusters < 2 or n_clusters > n:
        raise ValueError(f"n_clusters must satisfy 2 <= n_clusters <= {n}")
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
    pred = spectral_cluster(adj, n_clusters)
    true_labels = np.asarray(_get_true_labels(stocks, code_to_industry))
    valid = true_labels >= 0
    if valid.sum() == 0:
        raise ValueError("no known industry labels available for NMI")
    return float(normalized_mutual_info_score(true_labels[valid], pred[valid]))


def eval_ari(adj, stocks, code_to_industry, n_clusters=35):
    """ARI: 调整兰德指数 (越高越好)"""
    pred = spectral_cluster(adj, n_clusters)
    true_labels = np.asarray(_get_true_labels(stocks, code_to_industry))
    valid = true_labels >= 0
    if valid.sum() == 0:
        raise ValueError("no known industry labels available for ARI")
    return float(adjusted_rand_score(true_labels[valid], pred[valid]))


def eval_modularity(adj, stocks, code_to_industry):
    """模块度: 网络社区结构质量"""
    adj = validate_square_matrix(adj, "adjacency")
    if adj.shape[0] != len(stocks):
        raise ValueError("adjacency size must match stocks")
    G = nx.from_numpy_array(adj)
    if G.number_of_edges() == 0:
        return 0.0
    communities = {}
    for idx, stock in enumerate(stocks):
        ind = normalize_industry(code_to_industry.get(stock))
        # Keep unknown labels out of a shared pseudo-industry; otherwise all
        # missing labels would manufacture a strong community by themselves.
        if ind is None:
            ind = f'__unknown_{idx}'
        communities.setdefault(ind, set()).add(idx)
    return float(nx.community.modularity(G, list(communities.values())))


def eval_ic(adj, stocks, code_to_industry):
    """IC: 行业一致性 (同行业边 / 总边数)"""
    adj = validate_square_matrix(adj, "adjacency")
    if adj.shape[0] != len(stocks):
        raise ValueError("adjacency size must match stocks")
    n = adj.shape[0]
    same_count, total_count = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                ind_i = normalize_industry(code_to_industry.get(stocks[i]))
                ind_j = normalize_industry(code_to_industry.get(stocks[j]))
                if ind_i is None or ind_j is None:
                    continue
                total_count += 1
                if ind_i == ind_j:
                    same_count += 1
    return same_count / max(total_count, 1)


def eval_cov_error(cov_pred, cov_true):
    """协方差预测误差 (相对 Frobenius 范数, 越低越好)"""
    cov_pred = validate_square_matrix(cov_pred, "cov_pred")
    cov_true = validate_square_matrix(cov_true, "cov_true")
    if cov_pred.shape != cov_true.shape:
        raise ValueError("cov_pred and cov_true must have the same shape")
    denominator = np.linalg.norm(cov_true, 'fro')
    if denominator == 0:
        raise ValueError("cov_true has zero Frobenius norm")
    return float(np.linalg.norm(cov_pred - cov_true, 'fro') / denominator)


def eval_log_likelihood(cov_pred, returns_test):
    """样本外对数似然 (越高越好)"""
    cov_pred = validate_square_matrix(cov_pred, "cov_pred")
    returns_test = np.asarray(returns_test, dtype=float)
    if returns_test.ndim != 2 or returns_test.shape[1] != cov_pred.shape[0]:
        raise ValueError("returns_test must be a T x N array matching cov_pred")
    if len(returns_test) < 2 or not np.isfinite(returns_test).all():
        raise ValueError("returns_test must contain at least 2 finite observations")
    n = cov_pred.shape[0]
    cov = ensure_psd(cov_pred, 1e-8)
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("cov_pred is not positive definite")
    centered = returns_test - returns_test.mean(axis=0, keepdims=True)
    S = np.atleast_2d(np.cov(centered, rowvar=False, ddof=1))
    solved = np.linalg.solve(cov, S)
    # Return the mean per-observation Gaussian log-likelihood.
    return float(-0.5 * (n * np.log(2 * np.pi) + log_det + np.trace(solved)))


def eval_rank_ic(cov_pred, cov_true):
    """协方差元素的 Spearman 秩相关 (越高越好)"""
    from scipy.stats import spearmanr
    cov_pred = validate_square_matrix(cov_pred, "cov_pred")
    cov_true = validate_square_matrix(cov_true, "cov_true")
    if cov_pred.shape != cov_true.shape:
        raise ValueError("cov_pred and cov_true must have the same shape")
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
                 n_clusters=35, adj_fn=None, max_weight=0.05):
        self.returns = np.asarray(returns, dtype=float)
        if self.returns.ndim != 2 or not np.isfinite(self.returns).all():
            raise ValueError("returns must be a finite T x N array")
        self.stocks = stocks
        if self.returns.shape[1] != len(stocks):
            raise ValueError("returns columns must match stocks")
        self.code_to_industry = code_to_industry
        self.industry_prior = validate_square_matrix(industry_prior, "industry_prior")
        if self.industry_prior.shape != (len(stocks), len(stocks)):
            raise ValueError("industry_prior shape must match stocks")
        self.n_stocks = len(stocks)
        self.warmup = warmup
        self.eval_freq = eval_freq
        self.forecast = forecast
        self.rebalance = rebalance
        self.n_clusters = n_clusters
        self.adj_fn = adj_fn or (lambda c: topk_adj(c, k=5))
        if not np.isfinite(max_weight) or max_weight <= 0 or max_weight > 1:
            raise ValueError("max_weight must be in (0, 1]")
        if self.n_stocks * max_weight < 1 - 1e-12:
            raise ValueError("max_weight is infeasible for the number of stocks")
        self.max_weight = max_weight
        if warmup < 2 or eval_freq <= 0 or forecast < 2 or rebalance <= 0:
            raise ValueError("warmup, eval_freq, forecast and rebalance must be positive")
        if len(self.returns) <= warmup + forecast:
            raise ValueError("returns does not contain a complete evaluation window")
        self.eval_points = list(range(
            warmup, len(self.returns) - forecast, eval_freq
        ))

    def evaluate(self, name, estimator_fn):
        """评估一个方法

        Args:
            name: 方法名称
            estimator_fn: 估计器函数 (history -> (corr, cov))

        Returns:
            dict: 包含所有指标的字典
        """
        results = {
            'method': name,
            # A prior-assisted clustering score is useful for diagnostics, but
            # it must not be read as an independent discovery of industries.
            'NMI_Label_Assisted': bool(
                getattr(estimator_fn, 'uses_industry_prior', False)
            ),
        }
        nmis, aris, modularities, ics = [], [], [], []
        cov_errors, log_liks, rank_ics = [], [], []

        for t in self.eval_points:
            history = self.returns[:t]
            corr_est, cov_est = estimator_fn(history)
            corr_est = validate_square_matrix(corr_est, "estimated correlation")
            cov_est = validate_square_matrix(cov_est, "estimated covariance")
            expected_shape = (self.n_stocks, self.n_stocks)
            if corr_est.shape != expected_shape or cov_est.shape != expected_shape:
                raise ValueError("estimator outputs must match the stock universe")

            # 网络聚类指标
            adj = self.adj_fn(corr_est)
            np.fill_diagonal(adj, 0)
            nmis.append(eval_nmi(adj, self.stocks, self.code_to_industry, self.n_clusters))
            aris.append(eval_ari(adj, self.stocks, self.code_to_industry, self.n_clusters))
            modularities.append(eval_modularity(adj, self.stocks, self.code_to_industry))
            ics.append(eval_ic(adj, self.stocks, self.code_to_industry))

            # 协方差质量（样本外）
            future = self.returns[t:t + self.forecast]
            if len(future) >= self.forecast:
                cov_true = np.cov(future.T)
                cov_errors.append(eval_cov_error(cov_est, cov_true))
                log_liks.append(eval_log_likelihood(cov_est, future))
                rank_ics.append(eval_rank_ic(cov_est, cov_true))

        results['NMI'] = np.mean(nmis) if nmis else 0.0
        results['ARI'] = np.mean(aris) if aris else 0.0
        results['Modularity'] = np.mean(modularities) if modularities else np.nan
        results['IC'] = np.mean(ics) if ics else np.nan
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
            cov_est = validate_square_matrix(cov_est, "estimated covariance")
            if cov_est.shape != (self.n_stocks, self.n_stocks):
                raise ValueError("estimator covariance does not match the stock universe")
            cov_reg = ensure_psd(cov_est) + np.eye(self.n_stocks) * 1e-6
            w = min_var_weights(cov_reg, max_weight=self.max_weight)
            period_rets = self.returns[t:t + self.rebalance] @ w
            portfolio_rets.extend(period_rets.tolist())

        if not portfolio_rets:
            raise ValueError("returns does not contain a complete portfolio window")
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
    weight_sum = np.zeros(len(df))
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
        contribution = (1 - normalized) if w < 0 else normalized
        score += abs(w) * np.where(valid, contribution, 0.0)
        weight_sum += abs(w) * valid.astype(float)
    return np.divide(score, weight_sum, out=np.full(len(df), np.nan), where=weight_sum > 0)


def format_results_table(results_list, sort_by='NMI'):
    """格式化结果表格"""
    df = pd.DataFrame(results_list)
    cols_order = ['method', 'NMI', 'NMI_Label_Assisted', 'ARI', 'Modularity', 'IC',
                  'CovError', 'LogLik', 'RankIC',
                  'Sharpe', 'Sortino', 'MaxDD', 'Calmar',
                  'NMI_Std', 'CovErr_Std']
    existing = [c for c in cols_order if c in df.columns]
    df = df[existing]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    return df
