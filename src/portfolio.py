"""组合优化"""
import numpy as np
from scipy.optimize import minimize
from src.utils import ensure_psd, validate_square_matrix


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
    cov = validate_square_matrix(cov, "covariance")
    if not np.isfinite(max_weight) or max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be in (0, 1]")
    n = cov.shape[0]
    if n * max_weight < 1 - 1e-12:
        raise ValueError(
            f"infeasible weight cap: {n} assets * max_weight={max_weight} < 1"
        )
    cov = ensure_psd(cov)
    w0 = np.ones(n) / n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight)] * n

    result = minimize(
        lambda w: w @ cov @ w, w0,
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 300, 'ftol': 1e-12}
    )
    if not result.success:
        raise RuntimeError(f"minimum-variance optimization failed: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    if (not np.isfinite(weights).all() or
            abs(weights.sum() - 1) > 1e-7 or
            weights.min() < -1e-7 or
            weights.max() > max_weight + 1e-7):
        raise RuntimeError("optimizer returned weights violating portfolio constraints")
    weights = np.clip(weights, 0, max_weight)
    weights /= weights.sum()
    if weights.max() > max_weight + 1e-7:
        raise RuntimeError("normalization pushed weights over max_weight")
    return weights


def eval_portfolio_metrics(realized_returns, risk_free=0.0, periods_per_year=252):
    """计算组合绩效指标

    Args:
        realized_returns: 每日收益率序列

    Returns:
        dict: {sharpe, sortino, max_drawdown, calmar}
    """
    r = np.asarray(realized_returns, dtype=float)
    if r.ndim != 1 or not np.isfinite(r).all():
        raise ValueError("realized_returns must be a finite 1-D array")
    if not np.isfinite(risk_free) or periods_per_year <= 0:
        raise ValueError("risk_free must be finite and periods_per_year positive")
    if np.any(1 + r <= 0):
        raise ValueError("returns must be greater than -100%")
    if len(r) < 10:
        raise ValueError("at least 10 realized returns are required")
    excess = r - risk_free
    ann_ret = float(np.expm1(np.log1p(r).mean() * periods_per_year))
    ann_vol = float(np.std(excess, ddof=1) * np.sqrt(periods_per_year))
    sharpe = (float(np.mean(excess)) * np.sqrt(periods_per_year) / ann_vol
              if ann_vol > 0 else 0.0)

    downside = np.minimum(excess, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(periods_per_year))
    sortino = (float(np.mean(excess)) * np.sqrt(periods_per_year) / downside_dev
               if downside_dev > 0 else 0.0)

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
