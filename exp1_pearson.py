"""
实验1: 基于Pearson相关性的静态关系模型
- 全样本相关矩阵
- 稀疏化相关网络（阈值法 + Top-K法）
- 行业一致性评估
- 稳定性评估（不同时间窗口的Frobenius范数差异）
- 协方差预测评估（下游任务）
"""
import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering
import networkx as nx
import json
import os
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/exp1"
os.makedirs(OUT_DIR, exist_ok=True)

# 加载数据
returns = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))

stocks = returns.columns.tolist()
n_stocks = len(stocks)
print(f"股票数: {n_stocks}, 交易日: {len(returns)}")

# ===== 1. 全样本Pearson相关矩阵 =====
print("\n=== 1. 全样本Pearson相关矩阵 ===")
corr_full = returns.corr().values

# ===== 2. 稀疏化策略 =====
print("\n=== 2. 稀疏化策略 ===")

def threshold_network(corr, threshold):
    """阈值法构建邻接矩阵"""
    adj = (corr > threshold).astype(float)
    np.fill_diagonal(adj, 0)
    return adj

def topk_network(corr, k):
    """每个节点保留Top-K个最强连接"""
    n = corr.shape[0]
    adj = np.zeros_like(corr)
    for i in range(n):
        row = corr[i].copy()
        row[i] = -np.inf
        top_indices = np.argsort(row)[-k:]
        adj[i, top_indices] = 1
    # 对称化
    adj = ((adj + adj.T) > 0).astype(float)
    np.fill_diagonal(adj, 0)
    return adj

# 测试不同阈值
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
topk_values = [5, 10, 15, 20, 30]

# ===== 3. 评估指标 =====

def evaluate_network(adj, corr, stocks, code_to_industry, method_name):
    """评估网络质量"""
    n = adj.shape[0]
    results = {'method': method_name}
    
    # 基本统计
    n_edges = int(adj.sum() / 2)
    density = n_edges / (n * (n - 1) / 2)
    results['n_edges'] = n_edges
    results['density'] = density
    
    # 度分布
    degrees = adj.sum(axis=1)
    results['avg_degree'] = float(degrees.mean())
    results['max_degree'] = int(degrees.max())
    
    # 行业一致性: 连边中同行业比例
    same_industry_edges = 0
    total_edges = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0:
                total_edges += 1
                ind_i = code_to_industry.get(stocks[i], 'X')
                ind_j = code_to_industry.get(stocks[j], 'Y')
                if ind_i == ind_j:
                    same_industry_edges += 1
    
    industry_consistency = same_industry_edges / max(total_edges, 1)
    results['industry_consistency'] = industry_consistency
    
    # 随机基准的行业一致性
    industry_labels = [code_to_industry.get(s, 'Unknown') for s in stocks]
    unique_industries = list(set(industry_labels))
    industry_sizes = {}
    for ind in unique_industries:
        industry_sizes[ind] = sum(1 for l in industry_labels if l == ind)
    random_consistency = sum(s * (s - 1) for s in industry_sizes.values()) / (n * (n - 1))
    results['random_industry_consistency'] = random_consistency
    results['industry_lift'] = industry_consistency / max(random_consistency, 1e-6)
    
    # 模块度 (用networkx)
    G = nx.from_numpy_array(adj)
    if n_edges > 0:
        try:
            # 基于行业的社区划分
            communities = {}
            for idx, s in enumerate(stocks):
                ind = code_to_industry.get(s, 'Unknown')
                if ind not in communities:
                    communities[ind] = set()
                communities[ind].add(idx)
            community_list = list(communities.values())
            modularity = nx.community.modularity(G, community_list)
            results['modularity_industry'] = modularity
        except:
            results['modularity_industry'] = 0.0
        
        # 谱聚类
        try:
            n_clusters = min(10, len(set(industry_labels)))
            sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                   random_state=42, n_init=10)
            adj_for_sc = adj.copy()
            np.fill_diagonal(adj_for_sc, 1)
            pred_labels = sc.fit_predict(adj_for_sc)
            
            # 将行业标签转为数字
            ind_label_map = {ind: i for i, ind in enumerate(unique_industries)}
            true_labels = [ind_label_map[code_to_industry.get(s, 'Unknown')] for s in stocks]
            
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            ari = adjusted_rand_score(true_labels, pred_labels)
            results['nmi_vs_industry'] = nmi
            results['ari_vs_industry'] = ari
        except Exception as e:
            results['nmi_vs_industry'] = 0.0
            results['ari_vs_industry'] = 0.0
    else:
        results['modularity_industry'] = 0.0
        results['nmi_vs_industry'] = 0.0
        results['ari_vs_industry'] = 0.0
    
    return results

# ===== 4. 稳定性评估 =====
print("\n=== 3. 稳定性评估 ===")

def compute_stability(returns, window_size=120, step=60):
    """计算滚动窗口相关矩阵的稳定性"""
    dates = returns.index
    corr_matrices = []
    
    for start in range(0, len(dates) - window_size, step):
        end = start + window_size
        window_returns = returns.iloc[start:end]
        corr = window_returns.corr().values
        corr_matrices.append(corr)
    
    # 计算相邻窗口的Frobenius范数差异
    frobenius_diffs = []
    for i in range(1, len(corr_matrices)):
        diff = np.linalg.norm(corr_matrices[i] - corr_matrices[i-1], 'fro')
        frobenius_diffs.append(diff)
    
    return {
        'n_windows': len(corr_matrices),
        'mean_frobenius_diff': float(np.mean(frobenius_diffs)),
        'std_frobenius_diff': float(np.std(frobenius_diffs)),
        'max_frobenius_diff': float(np.max(frobenius_diffs)),
    }

stability = compute_stability(returns)
print(f"窗口数: {stability['n_windows']}")
print(f"平均Frobenius差异: {stability['mean_frobenius_diff']:.4f}")
print(f"最大Frobenius差异: {stability['max_frobenius_diff']:.4f}")

# ===== 5. 协方差预测评估 =====
print("\n=== 4. 协方差预测评估 ===")

def covariance_prediction_eval(returns, train_ratio=0.7):
    """用前70%数据估计协方差，评估在后30%上的预测误差"""
    n_train = int(len(returns) * train_ratio)
    train_returns = returns.iloc[:n_train]
    test_returns = returns.iloc[n_train:]
    
    # 样本协方差（训练集估计）
    cov_train = train_returns.cov().values
    
    # 真实测试集协方差
    cov_test = test_returns.cov().values
    
    # 预测误差
    frobenius_error = np.linalg.norm(cov_train - cov_test, 'fro')
    
    # 相对误差
    relative_error = frobenius_error / np.linalg.norm(cov_test, 'fro')
    
    # 单位方差下的MAE
    mae = np.mean(np.abs(cov_train - cov_test))
    
    return {
        'frobenius_error': float(frobenius_error),
        'relative_error': float(relative_error),
        'mae': float(mae),
        'train_days': n_train,
        'test_days': len(test_returns),
    }

cov_pred = covariance_prediction_eval(returns)
print(f"训练/测试: {cov_pred['train_days']}/{cov_pred['test_days']} 天")
print(f"协方差预测 Frobenius误差: {cov_pred['frobenius_error']:.6f}")
print(f"相对误差: {cov_pred['relative_error']:.4f}")
print(f"MAE: {cov_pred['mae']:.8f}")

# ===== 6. 评估所有网络 =====
print("\n=== 5. 评估不同稀疏化策略 ===")
all_results = []

for thresh in thresholds:
    adj = threshold_network(corr_full, thresh)
    res = evaluate_network(adj, corr_full, stocks, code_to_industry, f'Threshold_{thresh}')
    all_results.append(res)
    print(f"Threshold={thresh}: edges={res['n_edges']}, density={res['density']:.4f}, "
          f"ind_consist={res['industry_consistency']:.4f}, lift={res['industry_lift']:.2f}, "
          f"NMI={res['nmi_vs_industry']:.4f}")

for k in topk_values:
    adj = topk_network(corr_full, k)
    res = evaluate_network(adj, corr_full, stocks, code_to_industry, f'TopK_{k}')
    all_results.append(res)
    print(f"TopK={k}: edges={res['n_edges']}, density={res['density']:.4f}, "
          f"ind_consist={res['industry_consistency']:.4f}, lift={res['industry_lift']:.2f}, "
          f"NMI={res['nmi_vs_industry']:.4f}")

# 保存结果
results_df = pd.DataFrame(all_results)
results_df.to_csv(f'{OUT_DIR}/network_evaluation.csv', index=False)

# 汇总
summary = {
    'experiment': 'Exp1: Static Pearson Correlation',
    'n_stocks': n_stocks,
    'n_trading_days': len(returns),
    'stability': stability,
    'covariance_prediction': cov_pred,
    'best_threshold': results_df.loc[results_df['nmi_vs_industry'].idxmax()].to_dict(),
    'network_results': all_results,
}

with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

print("\n实验1完成!")
print(f"最佳方法: {results_df.loc[results_df['nmi_vs_industry'].idxmax(), 'method']}")
print(f"最佳NMI: {results_df['nmi_vs_industry'].max():.4f}")
