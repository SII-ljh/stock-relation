"""
综合评估框架 - 多维度评估股票相关性模型
4个维度, 12+指标:
  1. 网络结构质量: NMI, ARI, Modularity, IC
  2. 协方差估计质量: Cov Error, Log-Likelihood, Rank IC
  3. 金融表现: Sharpe, Sortino, Max Drawdown, Calmar
  4. 鲁棒性: NMI Stability, Cov Error Stability
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
# 1. 网络结构质量指标
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


def _get_labels(stocks, code_to_industry):
    """获取行业标签映射"""
    unique_inds = sorted(set(code_to_industry.values()))
    ind_map = {ind: i for i, ind in enumerate(unique_inds)}
    true_labels = [ind_map.get(code_to_industry.get(s, 'Unknown'), 0) for s in stocks]
    return true_labels, ind_map


def eval_nmi(adj, stocks, code_to_industry, n_clusters=35):
    """NMI: 谱聚类 vs 行业分类"""
    try:
        abs_adj = np.abs(adj).astype(float)
        np.fill_diagonal(abs_adj, abs_adj.max())
        sc = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed',
            random_state=42, n_init=3
        )
        pred = sc.fit_predict(abs_adj)
        true_labels, _ = _get_labels(stocks, code_to_industry)
        return normalized_mutual_info_score(true_labels, pred)
    except Exception:
        return 0.0


def eval_ari(adj, stocks, code_to_industry, n_clusters=35):
    """ARI: 调整兰德指数, 衡量聚类一致性 (校正随机)"""
    try:
        abs_adj = np.abs(adj).astype(float)
        np.fill_diagonal(abs_adj, abs_adj.max())
        sc = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed',
            random_state=42, n_init=3
        )
        pred = sc.fit_predict(abs_adj)
        true_labels, _ = _get_labels(stocks, code_to_industry)
        return adjusted_rand_score(true_labels, pred)
    except Exception:
        return 0.0


def eval_modularity(adj, stocks, code_to_industry):
    """Modularity: 基于行业分类的模块度 (Q值)"""
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
        community_list = list(communities.values())
        return nx.community.modularity(G, community_list)
    except Exception:
        return 0.0


def eval_ic(adj, stocks, code_to_industry):
    """IC: 同行业连边占比 (Industry Consistency)"""
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


# ============================================================
# 2. 协方差估计质量指标
# ============================================================

def eval_cov_error(cov_pred, cov_true):
    """相对Frobenius误差"""
    return np.linalg.norm(cov_pred - cov_true, 'fro') / np.linalg.norm(cov_true, 'fro')


def eval_log_likelihood(cov_pred, returns_test):
    """
    样本外高斯对数似然 (越高越好)
    L = -0.5 * (n*log(2pi) + log|Sigma| + trace(Sigma^{-1} * S))
    返回每个样本的平均对数似然
    """
    n = cov_pred.shape[0]
    T = returns_test.shape[0]
    try:
        # 确保正定
        cov = cov_pred.copy()
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < 1e-8:
            cov += np.eye(n) * (1e-8 - eigvals.min())
        
        sign, log_det = np.linalg.slogdet(cov)
        if sign <= 0:
            return -np.inf
        
        cov_inv = np.linalg.inv(cov)
        S = np.cov(returns_test.T)
        
        avg_ll = -0.5 * (n * np.log(2 * np.pi) + log_det + np.trace(cov_inv @ S))
        return avg_ll
    except Exception:
        return -np.inf


def eval_rank_ic(cov_pred, cov_true):
    """
    Rank IC: 预测协方差 vs 真实协方差的Spearman秩相关
    衡量相对排序的准确性, 不受绝对值偏差影响
    """
    from scipy.stats import spearmanr
    n = cov_pred.shape[0]
    # 取上三角元素
    idx = np.triu_indices(n, k=1)
    pred_flat = cov_pred[idx]
    true_flat = cov_true[idx]
    rho, _ = spearmanr(pred_flat, true_flat)
    return rho if not np.isnan(rho) else 0.0


# ============================================================
# 3. 金融表现指标
# ============================================================

def min_var_weights(cov, max_weight=0.05):
    """最小方差组合权重"""
    n = cov.shape[0]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n
    w0 = np.ones(n) / n
    result = minimize(
        lambda w: w @ cov @ w, w0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': 300}
    )
    return result.x if result.success else w0


def eval_portfolio_metrics(realized_returns):
    """
    从已实现收益率序列计算多个金融指标
    返回: sharpe, sortino, max_drawdown, calmar
    """
    r = np.array(realized_returns)
    if len(r) < 10:
        return {'sharpe': 0.0, 'sortino': 0.0, 'max_drawdown': 0.0, 'calmar': 0.0,
                'ann_return': 0.0, 'ann_vol': 0.0}
    
    ann_ret = np.mean(r) * 252
    ann_vol = np.std(r) * np.sqrt(252)
    
    # Sharpe
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    
    # Sortino (下行标准差)
    downside = r[r < 0]
    downside_std = np.std(downside) * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = ann_ret / downside_std if downside_std > 0 else 0.0
    
    # Max Drawdown
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    drawdowns = (cum - peak) / peak
    max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
    
    # Calmar
    calmar = ann_ret / max_dd if max_dd > 0 else 0.0
    
    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_dd),
        'calmar': float(calmar),
        'ann_return': float(ann_ret),
        'ann_vol': float(ann_vol),
    }


# ============================================================
# 4. 鲁棒性指标 (时间序列统计量的稳定性)
# ============================================================

def eval_stability(values):
    """计算序列的稳定性 (标准差/均值, 越低越稳定)"""
    if len(values) < 2:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    return float(std)


# ============================================================
# 综合评估管线
# ============================================================

class ComprehensiveEvaluator:
    """
    综合评估器: 给定数据和模型, 计算全部12+指标
    """
    
    def __init__(self, returns, stocks, code_to_industry, industry_prior,
                 warmup=250, update_freq=20, eval_freq=60, forecast=60,
                 rebalance=60, n_clusters=35, topk=4):
        self.returns = returns  # np.ndarray
        self.stocks = stocks
        self.code_to_industry = code_to_industry
        self.industry_prior = industry_prior
        self.n_stocks = len(stocks)
        self.warmup = warmup
        self.update_freq = update_freq
        self.eval_freq = eval_freq
        self.forecast = forecast
        self.rebalance = rebalance
        self.n_clusters = n_clusters
        self.topk = topk
        
        self.eval_points = list(range(
            warmup, len(returns) - forecast, eval_freq
        ))
    
    def evaluate_static_method(self, name, corr_matrix, cov_matrix=None):
        """
        评估静态方法 (给定一个固定的相关/协方差矩阵)
        """
        results = {'method': name}
        
        # 网络指标
        adj = topk_adj(corr_matrix, k=self.topk)
        np.fill_diagonal(adj, 0)
        
        results['NMI'] = eval_nmi(adj, self.stocks, self.code_to_industry, self.n_clusters)
        results['ARI'] = eval_ari(adj, self.stocks, self.code_to_industry, self.n_clusters)
        results['Modularity'] = eval_modularity(adj, self.stocks, self.code_to_industry)
        results['IC'] = eval_ic(adj, self.stocks, self.code_to_industry)
        
        # 协方差指标 (静态: train/test split)
        if cov_matrix is not None:
            split = int(len(self.returns) * 0.7)
            cov_true = np.cov(self.returns[split:].T)
            results['CovError'] = eval_cov_error(cov_matrix, cov_true)
            results['LogLik'] = eval_log_likelihood(cov_matrix, self.returns[split:])
            results['RankIC'] = eval_rank_ic(cov_matrix, cov_true)
        else:
            results['CovError'] = np.nan
            results['LogLik'] = np.nan
            results['RankIC'] = np.nan
        
        # 投资组合指标 (静态方法用全样本协方差)
        if cov_matrix is not None:
            portfolio_rets = self._run_portfolio(
                lambda t: cov_matrix
            )
            pm = eval_portfolio_metrics(portfolio_rets)
            results.update({
                'Sharpe': pm['sharpe'],
                'Sortino': pm['sortino'],
                'MaxDD': pm['max_drawdown'],
                'Calmar': pm['calmar'],
            })
        else:
            results.update({'Sharpe': np.nan, 'Sortino': np.nan,
                           'MaxDD': np.nan, 'Calmar': np.nan})
        
        # 鲁棒性 (静态方法无时间变化, 设为0)
        results['NMI_Std'] = 0.0
        results['CovErr_Std'] = 0.0
        
        return results
    
    def evaluate_dynamic_method(self, name, estimator_factory, topk_override=None):
        """
        评估动态方法
        estimator_factory: 返回一个对象, 需要有:
          - update(batch): 更新
          - get_cov(): 返回协方差矩阵
          - get_corr(): 返回相关矩阵
        """
        results = {'method': name}
        tk = topk_override if topk_override is not None else self.topk
        
        estimator = estimator_factory()
        
        # 预热
        for t in range(0, self.warmup, self.update_freq):
            batch = self.returns[t:t + self.update_freq]
            if len(batch) > 0:
                estimator.update(batch)
        
        nmis, aris, cov_errors, log_liks, rank_ics = [], [], [], [], []
        
        for et in self.eval_points:
            # 更新到当前
            for t in range(max(self.warmup, et - self.update_freq * 5), et, self.update_freq):
                batch = self.returns[t:t + self.update_freq]
                if len(batch) > 0:
                    estimator.update(batch)
            
            # 网络指标
            corr = estimator.get_corr()
            adj = topk_adj(corr, k=tk)
            np.fill_diagonal(adj, 0)
            nmis.append(eval_nmi(adj, self.stocks, self.code_to_industry, self.n_clusters))
            aris.append(eval_ari(adj, self.stocks, self.code_to_industry, self.n_clusters))
            
            # 协方差指标
            future = self.returns[et:et + self.forecast]
            if len(future) >= self.forecast:
                cov_pred = estimator.get_cov()
                cov_true = np.cov(future.T)
                cov_errors.append(eval_cov_error(cov_pred, cov_true))
                log_liks.append(eval_log_likelihood(cov_pred, future))
                rank_ics.append(eval_rank_ic(cov_pred, cov_true))
        
        results['NMI'] = np.mean(nmis) if nmis else 0.0
        results['ARI'] = np.mean(aris) if aris else 0.0
        results['NMI_Std'] = eval_stability(nmis)
        
        # Modularity 和 IC 取最后一个时间点
        if len(nmis) > 0:
            corr = estimator.get_corr()
            adj = topk_adj(corr, k=tk)
            np.fill_diagonal(adj, 0)
            results['Modularity'] = eval_modularity(adj, self.stocks, self.code_to_industry)
            results['IC'] = eval_ic(adj, self.stocks, self.code_to_industry)
        else:
            results['Modularity'] = 0.0
            results['IC'] = 0.0
        
        results['CovError'] = np.mean(cov_errors) if cov_errors else np.nan
        results['CovErr_Std'] = eval_stability(cov_errors) if len(cov_errors) > 1 else 0.0
        results['LogLik'] = np.mean(log_liks) if log_liks else np.nan
        results['RankIC'] = np.mean(rank_ics) if rank_ics else np.nan
        
        # 投资组合
        portfolio_rets = self._run_portfolio_dynamic(estimator_factory)
        pm = eval_portfolio_metrics(portfolio_rets)
        results.update({
            'Sharpe': pm['sharpe'],
            'Sortino': pm['sortino'],
            'MaxDD': pm['max_drawdown'],
            'Calmar': pm['calmar'],
        })
        
        return results
    
    def _run_portfolio(self, cov_fn):
        """运行投资组合回测 (静态)"""
        realized_rets = []
        for t in range(self.warmup, len(self.returns) - self.rebalance, self.rebalance):
            cov = cov_fn(t)
            cov_reg = cov + np.eye(self.n_stocks) * 1e-6
            w = min_var_weights(cov_reg)
            period_rets = self.returns[t:t + self.rebalance] @ w
            realized_rets.extend(period_rets.tolist())
        return realized_rets
    
    def _run_portfolio_dynamic(self, estimator_factory):
        """运行投资组合回测 (动态)"""
        estimator = estimator_factory()
        realized_rets = []
        
        for t in range(0, self.warmup, self.update_freq):
            batch = self.returns[t:t + self.update_freq]
            if len(batch) > 0:
                estimator.update(batch)
        
        for t in range(self.warmup, len(self.returns) - self.rebalance, self.rebalance):
            # 更新到当前
            for tt in range(max(self.warmup, t - self.update_freq * 5), t, self.update_freq):
                batch = self.returns[tt:tt + self.update_freq]
                if len(batch) > 0:
                    estimator.update(batch)
            
            cov = estimator.get_cov()
            cov_reg = cov + np.eye(self.n_stocks) * 1e-6
            w = min_var_weights(cov_reg)
            period_rets = self.returns[t:t + self.rebalance] @ w
            realized_rets.extend(period_rets.tolist())
        
        return realized_rets


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


def format_results_table(results_list, sort_by='NMI', ascending=False):
    """格式化结果为DataFrame并排序"""
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
    计算综合得分 (0-1归一化后加权平均)
    权重: NMI(25%), ARI(10%), Modularity(5%), IC(5%),
          CovError(10%, 反向), RankIC(10%),
          Sharpe(15%), Sortino(10%), MaxDD(5%, 反向), Calmar(5%)
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
        vmin = vals[valid].min()
        vmax = vals[valid].max()
        if vmax - vmin < 1e-10:
            normalized = np.where(valid, 0.5, 0.0)
        else:
            normalized = np.where(valid, (vals - vmin) / (vmax - vmin), 0.0)
        if w < 0:
            # 反向指标: 越小越好
            normalized = np.where(valid, 1 - normalized, 0.0)
            score += abs(w) * normalized
        else:
            score += w * normalized
    
    return score


if __name__ == '__main__':
    print("评估框架已加载, 包含以下指标:")
    print("1. 网络结构: NMI, ARI, Modularity, IC")
    print("2. 协方差质量: CovError, LogLik, RankIC")
    print("3. 金融表现: Sharpe, Sortino, MaxDD, Calmar")
    print("4. 鲁棒性: NMI_Std, CovErr_Std")
    print("5. 综合得分: CompositeScore")
