"""
数据预处理和探索性分析
1. 清洗收益率数据
2. 基础统计描述
3. 初步相关性热力图
"""
import pandas as pd
import numpy as np
import os
import json

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
OUT_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/results"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. 加载数据
print("=== 加载数据 ===")
close = pd.read_csv(f'{DATA_DIR}/close_prices_valid.csv', index_col=0, parse_dates=True)
returns = pd.read_csv(f'{DATA_DIR}/returns.csv', index_col=0, parse_dates=True)
industry = pd.read_csv(f'{DATA_DIR}/industry_info.csv')

print(f"收盘价: {close.shape}")
print(f"收益率: {returns.shape}")

# 2. 清洗
# 用前值填充少量缺失
returns = returns.fillna(method='ffill').fillna(0)
close = close.fillna(method='ffill').fillna(method='bfill')

# 去掉异常收益率 (绝对值 > 20% 的日收益率)
extreme_mask = returns.abs() > 0.20
print(f"极端收益率数量: {extreme_mask.sum().sum()} ({extreme_mask.mean().mean():.4%})")
returns = returns.clip(-0.20, 0.20)

# 保存清洗后的数据
returns.to_csv(f'{DATA_DIR}/returns_clean.csv')
close.to_csv(f'{DATA_DIR}/close_prices_clean.csv')

# 3. 基础统计
print("\n=== 基础统计 ===")
stats = pd.DataFrame({
    'mean_return': returns.mean(),
    'std_return': returns.std(),
    'sharpe': returns.mean() / returns.std() * np.sqrt(252),
    'skew': returns.skew(),
    'kurtosis': returns.kurtosis(),
    'max_drawdown': ((close / close.cummax()) - 1).min(),
})

# 合并行业
code_to_industry = dict(zip(industry['code'].astype(str).str.zfill(6), industry['industry']))
stats['industry'] = stats.index.map(lambda x: code_to_industry.get(x, 'Unknown'))

print(f"\n平均日收益率: {returns.mean().mean():.6f}")
print(f"平均日波动率: {returns.std().mean():.6f}")
print(f"平均年化Sharpe: {stats['sharpe'].mean():.4f}")
print(f"\n各行业平均Sharpe:")
print(stats.groupby('industry')['sharpe'].mean().sort_values(ascending=False).head(10))

stats.to_csv(f'{OUT_DIR}/stock_statistics.csv')

# 4. 计算全样本相关矩阵
print("\n=== 相关性分析 ===")
corr_matrix = returns.corr()

# 相关性统计
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
all_corrs = upper_tri.stack().values

print(f"相关性统计:")
print(f"  均值: {np.mean(all_corrs):.4f}")
print(f"  中位数: {np.median(all_corrs):.4f}")
print(f"  标准差: {np.std(all_corrs):.4f}")
print(f"  最大值: {np.max(all_corrs):.4f}")
print(f"  最小值: {np.min(all_corrs):.4f}")
print(f"  > 0.5 的比例: {(all_corrs > 0.5).mean():.4%}")
print(f"  > 0.7 的比例: {(all_corrs > 0.7).mean():.4%}")
print(f"  < 0 的比例: {(all_corrs < 0).mean():.4%}")

# 保存相关矩阵
corr_matrix.to_csv(f'{OUT_DIR}/correlation_matrix_full.csv')

# 5. 行业内 vs 行业间相关性
print("\n=== 行业内 vs 行业间相关性 ===")
intra_corrs = []
inter_corrs = []

for i in range(len(returns.columns)):
    for j in range(i+1, len(returns.columns)):
        c1, c2 = returns.columns[i], returns.columns[j]
        ind1 = code_to_industry.get(c1, 'X')
        ind2 = code_to_industry.get(c2, 'Y')
        corr_val = corr_matrix.iloc[i, j]
        if ind1 == ind2 and ind1 != 'Unknown':
            intra_corrs.append(corr_val)
        else:
            inter_corrs.append(corr_val)

print(f"行业内平均相关性: {np.mean(intra_corrs):.4f} (N={len(intra_corrs)})")
print(f"行业间平均相关性: {np.mean(inter_corrs):.4f} (N={len(inter_corrs)})")
print(f"差异: {np.mean(intra_corrs) - np.mean(inter_corrs):.4f}")

# 6. 时间段分析
print("\n=== 时间段相关性变化 ===")
yearly_corrs = {}
for year in range(2020, 2026):
    year_returns = returns[returns.index.year == year]
    if len(year_returns) > 50:
        year_corr = year_returns.corr()
        upper = year_corr.where(np.triu(np.ones(year_corr.shape), k=1).astype(bool))
        vals = upper.stack().values
        yearly_corrs[year] = {
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
            'std': float(np.std(vals)),
        }
        print(f"  {year}: mean={yearly_corrs[year]['mean']:.4f}, median={yearly_corrs[year]['median']:.4f}")

# 保存分析结果
analysis_results = {
    'n_stocks': int(close.shape[1]),
    'n_trading_days': int(close.shape[0]),
    'date_range': f"{close.index[0].strftime('%Y-%m-%d')} ~ {close.index[-1].strftime('%Y-%m-%d')}",
    'avg_daily_return': float(returns.mean().mean()),
    'avg_daily_vol': float(returns.std().mean()),
    'avg_correlation': float(np.mean(all_corrs)),
    'intra_industry_corr': float(np.mean(intra_corrs)),
    'inter_industry_corr': float(np.mean(inter_corrs)),
    'yearly_correlations': yearly_corrs,
}

with open(f'{OUT_DIR}/eda_summary.json', 'w') as f:
    json.dump(analysis_results, f, indent=2, ensure_ascii=False)

print("\n预处理和分析完成!")
print(f"结果保存在: {OUT_DIR}")
