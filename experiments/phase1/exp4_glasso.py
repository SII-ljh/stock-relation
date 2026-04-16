"""
实验4: Graphical Lasso + 谱聚类
Graphical Lasso估计稀疏精度矩阵（逆协方差），
非零元素表示条件依赖关系——更准确地捕捉直接关系
"""
import pandas as pd
import numpy as np
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso, LedoitWolf
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import networkx as nx
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results/exp4"
os.makedirs(OUT_DIR, exist_ok=True)

returns = pd.read_csv(f'{DATA_DIR}/returns_clean.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stocks = returns.columns.tolist()
n_stocks = len(stocks)

print(f"股票数: {n_stocks}, 交易日: {len(returns)}")

# 标准化
scaler = StandardScaler()
returns_std = pd.DataFrame(
    scaler.fit_transform(returns),
    index=returns.index, columns=returns.columns
)

# ===== 评估函数 =====
def eval_adj(adj, stocks, code_to_industry, name):
    n = adj.shape[0]
    n_edges = int((adj != 0).sum()) // 2
    
    # 行业一致性
    same = total = 0
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] != 0:
                total += 1
                if code_to_industry.get(stocks[i], 'X') == code_to_industry.get(stocks[j], 'Y'):
                    same += 1
    ic = same / max(total, 1)
    
    # NMI
    nmi = 0.0
    try:
        abs_adj = np.abs(adj)
        np.fill_diagonal(abs_adj, abs_adj.max())
        sc = SpectralClustering(n_clusters=min(10, n//5), affinity='precomputed',
                               random_state=42, n_init=5)
        pred = sc.fit_predict(abs_adj)
        unique_inds = sorted(set(code_to_industry.values()))
        ind_map = {ind: i for i, ind in enumerate(unique_inds)}
        true_labels = [ind_map.get(code_to_industry.get(s, 'Unknown'), 0) for s in stocks]
        nmi = normalized_mutual_info_score(true_labels, pred)
        ari = adjusted_rand_score(true_labels, pred)
    except:
        ari = 0.0
    
    return {
        'method': name,
        'n_edges': n_edges,
        'density': n_edges / (n * (n - 1) / 2),
        'industry_consistency': ic,
        'nmi': nmi,
        'ari': ari,
    }

# ===== 1. GraphicalLasso with CV =====
print("\n=== 1. GraphicalLasso CV ===")

# 使用Ledoit-Wolf初始化以加速
try:
    glasso_cv = GraphicalLassoCV(cv=3, max_iter=200, n_jobs=40, verbose=0)
    glasso_cv.fit(returns_std.values)
    
    precision = glasso_cv.precision_
    alpha_cv = glasso_cv.alpha_
    print(f"最优alpha: {alpha_cv:.6f}")
    
    # 精度矩阵的非零模式 = 条件依赖网络
    adj_glasso_cv = (np.abs(precision) > 1e-4).astype(float)
    np.fill_diagonal(adj_glasso_cv, 0)
    
    res_cv = eval_adj(adj_glasso_cv, stocks, code_to_industry, f'GLasso_CV_alpha{alpha_cv:.4f}')
    print(f"GLasso CV: edges={res_cv['n_edges']}, ic={res_cv['industry_consistency']:.4f}, NMI={res_cv['nmi']:.4f}")
except Exception as e:
    print(f"GLasso CV 失败: {e}")
    alpha_cv = 0.05
    res_cv = None

# ===== 2. 不同alpha值的GLasso =====
print("\n=== 2. 不同alpha值 ===")
alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
all_results = []

for alpha in alphas:
    try:
        glasso = GraphicalLasso(alpha=alpha, max_iter=200)
        glasso.fit(returns_std.values)
        
        precision = glasso.precision_
        adj = (np.abs(precision) > 1e-4).astype(float)
        np.fill_diagonal(adj, 0)
        
        res = eval_adj(adj, stocks, code_to_industry, f'GLasso_alpha{alpha}')
        all_results.append(res)
        print(f"alpha={alpha}: edges={res['n_edges']}, density={res['density']:.4f}, "
              f"ic={res['industry_consistency']:.4f}, NMI={res['nmi']:.4f}")
    except Exception as e:
        print(f"alpha={alpha}: 失败 ({str(e)[:50]})")

if res_cv:
    all_results.append(res_cv)

# ===== 3. 协方差预测评估 =====
print("\n=== 3. 协方差预测评估（滚动窗口）===")

WINDOW = 250
STEP = 60
test_starts = list(range(WINDOW, len(returns) - 60, STEP))

def glasso_cov_predict(returns, window, test_start, alpha):
    train = returns.iloc[test_start-window:test_start].values
    test = returns.iloc[test_start:test_start+window].values
    
    if len(train) < 50 or len(test) < 30:
        return None
    
    scaler = StandardScaler()
    train_std = scaler.fit_transform(train)
    
    try:
        gl = GraphicalLasso(alpha=alpha, max_iter=100)
        gl.fit(train_std)
        cov_pred = gl.covariance_
        # 反标准化
        stds = scaler.scale_
        cov_pred = cov_pred * np.outer(stds, stds)
    except:
        # fallback to LW
        lw = LedoitWolf()
        lw.fit(train)
        cov_pred = lw.covariance_
    
    cov_true = np.cov(test.T)
    frob = np.linalg.norm(cov_pred - cov_true, 'fro')
    rel_err = frob / np.linalg.norm(cov_true, 'fro')
    return rel_err

# 测试几个alpha
for alpha in [0.02, 0.05, 0.1]:
    errors = []
    for ts in test_starts:
        err = glasso_cov_predict(returns, WINDOW, ts, alpha)
        if err is not None:
            errors.append(err)
    if errors:
        print(f"GLasso alpha={alpha}: avg_rel_err={np.mean(errors):.4f} (std={np.std(errors):.4f})")

# LedoitWolf基准
lw_errors = []
for ts in test_starts:
    train = returns.iloc[ts-WINDOW:ts].values
    test = returns.iloc[ts:ts+WINDOW].values
    if len(test) < 30:
        continue
    lw = LedoitWolf()
    lw.fit(train)
    cov_pred = lw.covariance_
    cov_true = np.cov(test.T)
    frob = np.linalg.norm(cov_pred - cov_true, 'fro')
    rel_err = frob / np.linalg.norm(cov_true, 'fro')
    lw_errors.append(rel_err)
print(f"LedoitWolf baseline: avg_rel_err={np.mean(lw_errors):.4f}")

# ===== 4. 偏相关网络 vs 全相关网络对比 =====
print("\n=== 4. 偏相关 vs 全相关对比 ===")

# 从最佳GLasso得到偏相关网络
best_alpha = 0.05
try:
    gl = GraphicalLasso(alpha=best_alpha, max_iter=200)
    gl.fit(returns_std.values)
    precision = gl.precision_
    
    # 偏相关 = -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1)
    
    # Top-K偏相关网络
    def topk_adj(mat, k):
        n = mat.shape[0]
        adj = np.zeros_like(mat)
        for i in range(n):
            row = np.abs(mat[i].copy())
            row[i] = -np.inf
            top = np.argsort(row)[-k:]
            adj[i, top] = 1
        return ((adj + adj.T) > 0).astype(float)
    
    for k in [5, 10, 15]:
        adj_pc = topk_adj(partial_corr, k)
        np.fill_diagonal(adj_pc, 0)
        res_pc = eval_adj(adj_pc, stocks, code_to_industry, f'PartialCorr_TopK{k}')
        all_results.append(res_pc)
        print(f"PartialCorr TopK={k}: ic={res_pc['industry_consistency']:.4f}, NMI={res_pc['nmi']:.4f}")
    
    # 全相关Top-K（对比）
    full_corr = returns.corr().values
    for k in [5, 10, 15]:
        adj_fc = topk_adj(full_corr, k)
        np.fill_diagonal(adj_fc, 0)
        res_fc = eval_adj(adj_fc, stocks, code_to_industry, f'FullCorr_TopK{k}')
        all_results.append(res_fc)
        print(f"FullCorr TopK={k}: ic={res_fc['industry_consistency']:.4f}, NMI={res_fc['nmi']:.4f}")
        
except Exception as e:
    print(f"偏相关分析失败: {e}")

# 保存结果
results_df = pd.DataFrame(all_results)
results_df.to_csv(f'{OUT_DIR}/results.csv', index=False)

with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump({
        'experiment': 'Exp4: Graphical Lasso + Spectral Clustering',
        'results': all_results,
    }, f, indent=2, ensure_ascii=False, default=str)

print(f"\n=== 总结 ===")
print(results_df[['method', 'n_edges', 'density', 'industry_consistency', 'nmi']].to_string())
best_nmi = results_df.loc[results_df['nmi'].idxmax()]
print(f"\nNMI最优: {best_nmi['method']} ({best_nmi['nmi']:.4f})")

print("\n实验4完成!")
