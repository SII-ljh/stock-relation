"""通用工具函数"""
import numpy as np


def cov_to_corr(cov):
    """协方差矩阵 -> 相关矩阵（返回新矩阵）"""
    d = np.sqrt(np.diag(cov).copy())
    d[d < 1e-10] = 1e-10
    result = cov / np.outer(d, d)
    result = np.clip(result, -1, 1)
    np.fill_diagonal(result, 1.0)
    return result


def ensure_psd(cov, eps=1e-6):
    """确保矩阵正半定（返回新矩阵）"""
    sym = (cov + cov.T) / 2
    eigvals = np.linalg.eigvalsh(sym)
    if eigvals.min() < eps:
        return sym + np.eye(sym.shape[0]) * (eps - eigvals.min())
    return sym.copy()
