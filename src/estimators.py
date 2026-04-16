"""协方差/相关矩阵估计器

每个估计器接口统一: estimator(history: np.ndarray) -> (corr, cov)
  - history: T x N 的收益率矩阵（T个交易日，N只股票）
  - corr: N x N 相关矩阵
  - cov: N x N 协方差矩阵
"""
import numpy as np
from sklearn.covariance import LedoitWolf as _LedoitWolf
from src.utils import cov_to_corr, ensure_psd


def sample_covariance(history):
    """样本协方差估计（最基础的方法）"""
    cov = np.cov(history.T)
    return cov_to_corr(cov), cov


def ledoit_wolf(history):
    """Ledoit-Wolf 收缩估计

    通过线性收缩降低样本协方差的估计误差。
    """
    cov = _LedoitWolf().fit(history).covariance_
    return cov_to_corr(cov), cov


def rmt_denoise(history):
    """随机矩阵理论 (RMT) 去噪

    利用 Marchenko-Pastur 分布确定噪声特征值上界，
    将低于阈值的特征值替换为均值，保留信号特征值。
    这是本项目表现最好的基础估计器。
    """
    T, N = history.shape
    q = N / T

    cov = np.cov(history.T)
    std = np.sqrt(np.diag(cov).copy())
    std[std < 1e-10] = 1e-10
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1)

    eigvals, eigvecs = np.linalg.eigh(corr)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # Marchenko-Pastur 上界
    lam_plus = (1 + np.sqrt(q)) ** 2
    n_signal = max(int(np.sum(eigvals > lam_plus)), 1)

    # 去噪：噪声特征值替换为均值
    denoised = eigvals.copy()
    if n_signal < N:
        denoised[n_signal:] = np.mean(eigvals[n_signal:])

    corr_d = eigvecs @ np.diag(denoised) @ eigvecs.T
    np.fill_diagonal(corr_d, 1)
    cov_d = corr_d * np.outer(std, std)

    return cov_to_corr(ensure_psd(cov_d)), ensure_psd(cov_d)


def pca_factor(history, n_factors=20):
    """PCA 因子模型

    用前 n_factors 个主成分构建低秩协方差估计 + 对角残差。
    n_factors=10 时 Sharpe 最高（更强正则化 = 更稳定组合）。

    Args:
        history: T x N 收益率矩阵
        n_factors: 因子数量（默认20，推荐10用于组合优化）
    """
    cov_sample = np.cov(history.T)
    eigvals, eigvecs = np.linalg.eigh(cov_sample)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    B = eigvecs[:, :n_factors]
    factor_var = eigvals[:n_factors]

    cov_factor = B @ np.diag(factor_var) @ B.T
    residual_var = np.maximum(
        np.diag(cov_sample) - np.sum(B ** 2 * factor_var, axis=1), 1e-8
    )
    cov_est = ensure_psd(cov_factor + np.diag(residual_var))
    return cov_to_corr(cov_est), cov_est


def poet(history, n_factors=15, threshold_const=0.5):
    """POET 估计器 (Principal Orthogonal complEment Thresholding)

    因子模型 + 残差矩阵软阈值处理，适合高维场景。

    Args:
        history: T x N 收益率矩阵
        n_factors: 因子数量
        threshold_const: 阈值常数
    """
    T, N = history.shape
    cov_sample = np.cov(history.T)

    eigvals, eigvecs = np.linalg.eigh(cov_sample)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    B = eigvecs[:, :n_factors]
    factor_var = eigvals[:n_factors]
    cov_factor = B @ np.diag(factor_var) @ B.T

    residual = cov_sample - cov_factor
    threshold = threshold_const * np.sqrt(np.log(N) / T)

    thresholded = residual.copy()
    for i in range(N):
        for j in range(N):
            if i != j:
                scale = np.sqrt(abs(residual[i, i] * residual[j, j]))
                thresholded[i, j] = np.sign(residual[i, j]) * max(
                    abs(residual[i, j]) - threshold * scale, 0
                )

    cov_est = ensure_psd(cov_factor + thresholded)
    return cov_to_corr(cov_est), cov_est


def nonlinear_shrinkage(history):
    """非线性收缩估计器

    对每个特征值施加不同的收缩系数（与特征值大小相关）。
    """
    T, N = history.shape
    cov_sample = np.cov(history.T)

    eigvals, eigvecs = np.linalg.eigh(cov_sample)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    shrunk = np.zeros_like(eigvals)
    for i in range(N):
        diffs = eigvals - eigvals[i]
        diffs[i] = 1
        h = max(eigvals[i] * (N / T) ** 0.5 * 0.1, 1e-8)
        kernel_val = np.sum(h / (diffs ** 2 + h ** 2)) / (N * np.pi)
        shrink_factor = 1.0 / (1 + (N / T) * kernel_val * np.pi * eigvals[i])
        shrunk[i] = eigvals[i] * max(shrink_factor, 0.01)

    shrunk = np.maximum(shrunk, 1e-8)
    cov_est = ensure_psd(eigvecs @ np.diag(shrunk) @ eigvecs.T)
    return cov_to_corr(cov_est), cov_est
