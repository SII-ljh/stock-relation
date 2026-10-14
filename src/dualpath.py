"""DualPath 双路径框架

核心思想: 聚类和组合优化使用不同的矩阵
  - 聚类路径: 注入行业先验，放大行业差异 -> 更高 NMI
  - 组合路径: 纯去噪，不注入先验 -> 更高 Sharpe

NMI 在聚类路径注入先验时属于标签辅助诊断；它不能单独证明模型从收益率中
独立发现了行业结构。组合指标仍需在冻结参数的独立区间上报告。
"""
import numpy as np
from src.utils import ensure_psd, validate_square_matrix


def make_dual_path_estimator(base_estimator, industry_prior, cp=0.6, pp=0.0):
    """创建 DualPath 估计器

    Args:
        base_estimator: 基础估计器函数，接口 (history -> (corr, cov))
        industry_prior: N x N 行业先验矩阵（同行业=1，异行业=0）
        cp: 聚类路径的先验权重 (0~1)
            - 越大越依赖行业先验，推荐 0.8（加权TopK下最优）
        pp: 组合路径的先验权重
            - 推荐 0.0（纯去噪，不注入先验）

    Returns:
        estimator_fn: (history -> (corr_for_clustering, cov_for_portfolio))

    Example:
        estimator = make_dual_path_estimator(rmt_denoise, prior, cp=0.8)
        corr, cov = estimator(returns_array)
    """
    prior = validate_square_matrix(industry_prior, "industry_prior")
    if not np.isfinite(cp) or not 0 <= cp <= 1:
        raise ValueError("cp must be in [0, 1]")
    if not np.isfinite(pp) or not 0 <= pp <= 1:
        raise ValueError("pp must be in [0, 1]")
    if not np.allclose(prior, prior.T):
        raise ValueError("industry_prior must be symmetric")
    if not np.allclose(np.diag(prior), 0):
        raise ValueError("industry_prior diagonal must be zero")

    def estimator_fn(history):
        corr_base, cov_base = base_estimator(history)
        corr_base = validate_square_matrix(corr_base, "base correlation")
        cov_base = validate_square_matrix(cov_base, "base covariance")
        if corr_base.shape != prior.shape or cov_base.shape != prior.shape:
            raise ValueError("base estimator output shape does not match industry_prior")

        # 聚类路径: 融合行业先验
        corr_cluster = (1 - cp) * corr_base + cp * prior
        np.fill_diagonal(corr_cluster, 1.0)

        # 组合路径: 纯去噪（默认 pp=0）
        if pp > 0:
            avg_var = np.diag(cov_base).mean()
            prior_cov = prior * avg_var * 0.5
            np.fill_diagonal(prior_cov, np.diag(cov_base))
            cov_portfolio = ensure_psd((1 - pp) * cov_base + pp * prior_cov)
        else:
            cov_portfolio = cov_base

        return corr_cluster, cov_portfolio

    # Keep provenance with the callable so evaluators can label NMI/ARI as
    # label-assisted when the same industry prior is injected into clustering.
    estimator_fn.uses_industry_prior = bool(cp > 0)
    estimator_fn.industry_prior_weight = float(cp)
    estimator_fn.portfolio_prior_weight = float(pp)
    return estimator_fn


def make_ensemble_estimator(base_estimators, industry_prior, cp=0.6):
    """创建集成 DualPath 估计器

    将多个基础估计器的结果取平均，再融合行业先验。

    Args:
        base_estimators: 基础估计器列表
        industry_prior: N x N 行业先验矩阵
        cp: 聚类路径的先验权重

    Returns:
        estimator_fn: (history -> (corr_for_clustering, cov_for_portfolio))
    """
    prior = validate_square_matrix(industry_prior, "industry_prior")
    if not base_estimators:
        raise ValueError("base_estimators must not be empty")
    if not np.isfinite(cp) or not 0 <= cp <= 1:
        raise ValueError("cp must be in [0, 1]")

    def estimator_fn(history):
        corrs, covs = [], []
        for base_fn in base_estimators:
            c, cv = base_fn(history)
            c = validate_square_matrix(c, "base correlation")
            cv = validate_square_matrix(cv, "base covariance")
            if c.shape != prior.shape or cv.shape != prior.shape:
                raise ValueError("base estimator output shape does not match industry_prior")
            corrs.append(c)
            covs.append(cv)

        corr_avg = np.mean(corrs, axis=0)
        corr_cluster = (1 - cp) * corr_avg + cp * prior
        np.fill_diagonal(corr_cluster, 1.0)

        cov_portfolio = ensure_psd(np.mean(covs, axis=0))
        return corr_cluster, cov_portfolio

    estimator_fn.uses_industry_prior = bool(cp > 0)
    estimator_fn.industry_prior_weight = float(cp)
    return estimator_fn
