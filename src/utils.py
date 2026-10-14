"""通用工具函数"""
import numpy as np


def validate_square_matrix(matrix, name="matrix", finite=True):
    """Validate and return a floating-point square matrix.

    Public functions in this project accept matrices from user code and from
    CSV files.  Failing early here is preferable to producing a misleading
    zero score or an optimizer fallback later in the pipeline.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if finite and not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def cov_to_corr(cov):
    """协方差矩阵 -> 相关矩阵（返回新矩阵）"""
    cov = validate_square_matrix(cov, "covariance")
    diag = np.diag(cov)
    if np.any(diag < 0):
        raise ValueError("covariance diagonal must be non-negative")
    d = np.sqrt(diag.copy())
    d[d < 1e-10] = 1e-10
    result = cov / np.outer(d, d)
    result = np.clip(result, -1, 1)
    np.fill_diagonal(result, 1.0)
    return result


def ensure_psd(cov, eps=1e-6):
    """确保矩阵正半定（返回新矩阵）"""
    cov = validate_square_matrix(cov, "covariance")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite non-negative number")
    sym = (cov + cov.T) / 2
    eigvals = np.linalg.eigvalsh(sym)
    if eigvals.min() < eps:
        return sym + np.eye(sym.shape[0]) * (eps - eigvals.min())
    return sym.copy()
