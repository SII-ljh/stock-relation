"""DualPath 双路径框架

核心思想: 聚类和组合优化使用不同的矩阵
  - 聚类路径: 注入行业先验，放大行业差异 -> 更高 NMI
  - 组合路径: 纯去噪，不注入先验 -> 更高 Sharpe

这一设计打破了 NMI-Sharpe 之间的权衡，两者可以同时达到最优。
"""
import numpy as np
from src.utils import ensure_psd


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
    def estimator_fn(history):
        corr_base, cov_base = base_estimator(history)

        # 聚类路径: 融合行业先验
        corr_cluster = (1 - cp) * corr_base + cp * industry_prior
        np.fill_diagonal(corr_cluster, 1.0)

        # 组合路径: 纯去噪（默认 pp=0）
        if pp > 0:
            avg_var = np.diag(cov_base).mean()
            prior_cov = industry_prior * avg_var * 0.5
            np.fill_diagonal(prior_cov, np.diag(cov_base))
            cov_portfolio = ensure_psd((1 - pp) * cov_base + pp * prior_cov)
        else:
            cov_portfolio = cov_base

        return corr_cluster, cov_portfolio

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
    def estimator_fn(history):
        corrs, covs = [], []
        for base_fn in base_estimators:
            c, cv = base_fn(history)
            corrs.append(c)
            covs.append(cv)

        corr_avg = np.mean(corrs, axis=0)
        corr_cluster = (1 - cp) * corr_avg + cp * industry_prior
        np.fill_diagonal(corr_cluster, 1.0)

        cov_portfolio = ensure_psd(np.mean(covs, axis=0))
        return corr_cluster, cov_portfolio

    return estimator_fn
