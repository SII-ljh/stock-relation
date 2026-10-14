"""网络邻接矩阵构建

将相关矩阵稀疏化为邻接矩阵，用于后续的谱聚类。
"""
import numpy as np


def _validate_corr_and_k(corr, k):
    corr = np.asarray(corr, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError(f"corr must be a square matrix, got {corr.shape}")
    n = corr.shape[0]
    if n == 0:
        raise ValueError("corr must not be empty")
    if not np.isfinite(corr).all():
        raise ValueError("corr contains NaN or infinite values")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if k < 0 or k >= n:
        raise ValueError(f"k must satisfy 0 <= k < n ({n}), got {k}")
    return corr, int(k)


def topk_adj(corr, k=5):
    """二值 TopK 邻接矩阵

    对每个节点保留相关系数绝对值最大的 k 个邻居，
    边权重为 1（二值），使用 OR 对称化。

    Args:
        corr: N x N 相关矩阵
        k: 每个节点保留的邻居数

    Returns:
        adj: N x N 对称二值邻接矩阵
    """
    corr, k = _validate_corr_and_k(corr, k)
    n = corr.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        if k == 0:
            continue
        row = np.abs(corr[i]).copy()
        row[i] = -np.inf
        top_k = np.argsort(row)[-k:]
        adj[i, top_k] = 1.0
    return np.maximum(adj, adj.T)


def weighted_topk_adj(corr, k=5):
    """加权 TopK 邻接矩阵（本项目核心创新）

    对每个节点保留相关系数绝对值最大的 k 个邻居，
    边权重为 |相关系数|，使用 MAX 对称化。

相比二值 TopK，保留了关联强度的梯度信息，使谱聚类能区分"强关联"和"弱关联"。
历史实验中曾观察到 NMI 提升 6.3%，新数据集仍需独立协议验证。

    Args:
        corr: N x N 相关矩阵
        k: 每个节点保留的邻居数

    Returns:
        adj: N x N 对称加权邻接矩阵
    """
    corr, k = _validate_corr_and_k(corr, k)
    n = corr.shape[0]
    adj = np.zeros((n, n))
    for i in range(n):
        if k == 0:
            continue
        row = np.abs(corr[i]).copy()
        row[i] = -np.inf
        top_k = np.argsort(row)[-k:]
        for j in top_k:
            adj[i, j] = abs(corr[i, j])
    return np.maximum(adj, adj.T)
