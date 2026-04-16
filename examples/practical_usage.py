"""
实际应用示例: 如何将模型结果用于投资决策

本脚本展示三个实际场景:
  场景1: 查看某只股票的相似股票和所属聚类
  场景2: 构建最小方差组合并与等权组合对比回测
  场景3: 对比不同方法的表现 (Walk-Forward评估)

使用方法:
  cd stock-relation
  python examples/practical_usage.py

前置条件:
  需要先运行 quick_start.py 或确保 data/ 目录下有数据
"""
import sys
import os
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.data import load_data, build_industry_prior
from src.estimators import rmt_denoise, pca_factor
from src.network import weighted_topk_adj
from src.dualpath import make_dual_path_estimator
from src.portfolio import min_var_weights, eval_portfolio_metrics
from src.evaluation import (
    FlexibleEvaluator, compute_composite_score,
    format_results_table, spectral_cluster
)
from src.utils import ensure_psd

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')


def scenario_1():
    """场景1: 给定一只股票, 找到与它最相关的股票"""
    print("=" * 60)
    print("场景1: 查找相似股票")
    print("=" * 60)

    returns_df, stocks, code_to_industry = load_data(DATA_DIR)
    industry_prior = build_industry_prior(stocks, code_to_industry)

    estimator = make_dual_path_estimator(rmt_denoise, industry_prior, cp=0.8)
    corr_cluster, _ = estimator(returns_df.values)
    adj = weighted_topk_adj(corr_cluster, k=5)
    labels = spectral_cluster(adj, n_clusters=35)

    # 选第一只股票作为示例
    query_stock = stocks[0]
    query_idx = 0
    query_industry = code_to_industry.get(query_stock, '未知')
    query_cluster = labels[query_idx]

    print(f"\n  查询: {query_stock} (行业: {query_industry}, 聚类: {query_cluster})")

    # 同聚类股票
    cluster_members = [
        (stocks[i], code_to_industry.get(stocks[i], '未知'))
        for i in range(len(stocks))
        if labels[i] == query_cluster and i != query_idx
    ]
    print(f"\n  同聚类股票 ({len(cluster_members)}只):")
    for code, ind in cluster_members[:8]:
        j = stocks.index(code)
        corr_val = corr_cluster[query_idx, j]
        print(f"    {code} ({ind}) 相关系数: {corr_val:.3f}")

    # 全市场关联最强的5只
    row = np.abs(corr_cluster[query_idx]).copy()
    row[query_idx] = -np.inf
    top5 = np.argsort(row)[-5:][::-1]
    print(f"\n  全市场关联最强的5只:")
    for idx in top5:
        ind = code_to_industry.get(stocks[idx], '未知')
        print(f"    {stocks[idx]} ({ind}) 相关系数: {corr_cluster[query_idx, idx]:.3f}")


def scenario_2():
    """场景2: 构建最小方差组合并回测"""
    print("\n" + "=" * 60)
    print("场景2: 组合回测 (模型组合 vs 等权组合)")
    print("=" * 60)

    returns_df, stocks, code_to_industry = load_data(DATA_DIR)
    industry_prior = build_industry_prior(stocks, code_to_industry)
    ret_vals = returns_df.values
    n_stocks = len(stocks)

    lookback = 500   # 约2年训练数据
    rebalance = 60   # 每60个交易日调仓

    print(f"\n  训练窗口: {lookback}天, 调仓频率: {rebalance}天")

    estimator = make_dual_path_estimator(rmt_denoise, industry_prior, cp=0.8)

    # Walk-forward 回测
    model_rets = []
    equal_rets = []

    n_periods = 0
    for t in range(lookback, len(ret_vals) - rebalance, rebalance):
        history = ret_vals[:t]
        _, cov_est = estimator(history)
        cov_reg = ensure_psd(cov_est) + np.eye(n_stocks) * 1e-6
        w = min_var_weights(cov_reg, max_weight=0.05)

        period = ret_vals[t:t + rebalance]
        model_rets.extend((period @ w).tolist())
        equal_rets.extend(period.mean(axis=1).tolist())
        n_periods += 1

    model_m = eval_portfolio_metrics(model_rets)
    equal_m = eval_portfolio_metrics(equal_rets)

    print(f"  调仓次数: {n_periods}")
    print(f"\n  {'指标':<12} {'模型组合':>10} {'等权组合':>10} {'超额':>10}")
    print(f"  {'-' * 46}")
    for label, key in [('Sharpe', 'sharpe'), ('Sortino', 'sortino'),
                       ('最大回撤', 'max_drawdown'), ('Calmar', 'calmar')]:
        mv = model_m[key]
        ev = equal_m[key]
        diff = mv - ev
        sign = '+' if diff > 0 else ''
        if key == 'max_drawdown':
            sign = '' if diff > 0 else '+'
            diff = -diff if key == 'max_drawdown' else diff
        print(f"  {label:<12} {mv:>10.3f} {ev:>10.3f} {sign}{diff:>9.3f}")


def scenario_3():
    """场景3: 对比不同方法的表现"""
    print("\n" + "=" * 60)
    print("场景3: 方法对比 (Walk-Forward 评估)")
    print("=" * 60)

    returns_df, stocks, code_to_industry = load_data(DATA_DIR)
    industry_prior = build_industry_prior(stocks, code_to_industry)
    ret_vals = returns_df.values

    evaluator = FlexibleEvaluator(
        ret_vals, stocks, code_to_industry, industry_prior,
        n_clusters=35,
        adj_fn=lambda c: weighted_topk_adj(c, k=5),
    )

    methods = {
        'DualPath_RMT_cp0.8': make_dual_path_estimator(
            rmt_denoise, industry_prior, cp=0.8
        ),
        'DualPath_Factor_cp0.6': make_dual_path_estimator(
            lambda h: pca_factor(h, 10), industry_prior, cp=0.6
        ),
        'RMT(无先验)': rmt_denoise,
    }

    print(f"\n  评估 {len(methods)} 种方法 (Walk-Forward, 约需几分钟)...\n")

    results = []
    for name, fn in methods.items():
        print(f"  评估中: {name}...", end=' ', flush=True)
        r = evaluator.evaluate(name, fn)
        print(f"NMI={r['NMI']:.4f}, Sharpe={r['Sharpe']:.4f}")
        results.append(r)

    df = format_results_table(results)
    df['Composite'] = compute_composite_score(df)
    df = df.sort_values('Composite', ascending=False).reset_index(drop=True)

    print(f"\n  排名:")
    print(f"  {'#':<3} {'方法':<28} {'NMI':>6} {'Sharpe':>8} {'综合':>8}")
    print(f"  {'-' * 58}")
    for i, row in df.iterrows():
        print(f"  {i+1:<3} {row['method']:<28} "
              f"{row['NMI']:>6.4f} {row['Sharpe']:>8.4f} {row['Composite']:>8.4f}")


if __name__ == '__main__':
    scenario_1()
    scenario_2()
    # 取消下面的注释可运行方法对比 (耗时较长)
    # scenario_3()
