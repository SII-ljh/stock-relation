"""组合优化"""
import numpy as np
from scipy.optimize import minimize


def min_var_weights(cov, max_weight=0.05):
    """最小方差组合权重

    在权重非负、总和为1、单只上限 max_weight 的约束下，
    最小化组合方差 w'Cov w。

    Args:
        cov: N x N 协方差矩阵
        max_weight: 单只股票最大权重（默认5%）

    Returns:
        weights: N 维权重向量
    """
    n = cov.shape[0]
    w0 = np.ones(n) / n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n

    result = minimize(
        lambda w: w @ cov @ w, w0,
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 300}
    )
    return result.x if result.success else w0


def eval_portfolio_metrics(realized_returns):
    """计算组合绩效指标

    Args:
        realized_returns: 每日收益率序列

    Returns:
        dict: {sharpe, sortino, max_drawdown, calmar}
    """
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

    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_dd),
        'calmar': float(calmar),
    }
