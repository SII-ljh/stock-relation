"""数据下载、加载与预处理"""
import os
import time
import numpy as np
import pandas as pd


def download_data(data_dir='data', start='2020-01-01', end='2025-12-31'):
    """下载沪深300成分股日线数据（约5分钟）

    自动完成: 获取成分股列表 -> 下载日线数据 -> 合并清洗 -> 获取行业分类

    Args:
        data_dir: 数据保存目录
        start: 起始日期
        end: 结束日期
    """
    import baostock as bs
    from tqdm import tqdm

    os.makedirs(data_dir, exist_ok=True)

    lg = bs.login()
    print(f"baostock 登录: {lg.error_code} {lg.error_msg}")

    # 获取成分股列表
    print("获取沪深300成分股...")
    rs = bs.query_hs300_stocks()
    hs300_list = []
    while (rs.error_code == '0') and rs.next():
        hs300_list.append(rs.get_row_data())
    hs300_df = pd.DataFrame(hs300_list, columns=rs.fields)
    hs300_df.to_csv(os.path.join(data_dir, "hs300_constituents.csv"), index=False)
    print(f"成分股数量: {len(hs300_df)}")

    # 下载日线数据
    fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
    all_close = {}
    all_returns = {}
    numeric_cols = ['open', 'high', 'low', 'close', 'preclose',
                    'volume', 'amount', 'turn', 'pctChg']

    for _, row in tqdm(hs300_df.iterrows(), total=len(hs300_df), desc="下载数据"):
        code = row['code']
        code_short = code.replace("sh.", "").replace("sz.", "")
        try:
            rs = bs.query_history_k_data_plus(
                code, fields, start_date=start, end_date=end,
                frequency="d", adjustflag="2"
            )
            data_list = []
            while (rs.error_code == '0') and rs.next():
                data_list.append(rs.get_row_data())

            if len(data_list) > 100:
                df = pd.DataFrame(data_list, columns=rs.fields)
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.to_csv(os.path.join(data_dir, f"{code_short}.csv"), index=False)

                all_close[code_short] = df.set_index('date')['close']
                all_returns[code_short] = df.set_index('date')['pctChg'] / 100.0
        except Exception as e:
            tqdm.write(f"  {code}: {str(e)[:50]}")
        time.sleep(0.1)

    # 合并收盘价
    close_df = pd.DataFrame(all_close)
    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df.sort_index()

    # 过滤缺失超过10%的股票
    valid_mask = close_df.isnull().mean() < 0.1
    close_df.loc[:, valid_mask].to_csv(os.path.join(data_dir, "close_prices_valid.csv"))

    # 收益率
    ret_df = pd.DataFrame(all_returns)
    ret_df.index = pd.to_datetime(ret_df.index)
    ret_df = ret_df.sort_index().loc[:, valid_mask]
    ret_df.to_csv(os.path.join(data_dir, "returns.csv"))

    # 清洗收益率
    ret_clean = ret_df.ffill().fillna(0).clip(-0.20, 0.20)
    ret_clean.to_csv(os.path.join(data_dir, "returns_clean.csv"))

    # 行业分类
    print("获取行业分类...")
    industry_records = []
    for _, row in hs300_df.iterrows():
        code = row['code']
        code_short = code.replace("sh.", "").replace("sz.", "")
        try:
            rs = bs.query_stock_industry(code=code)
            while (rs.error_code == '0') and rs.next():
                data = rs.get_row_data()
                industry_records.append({
                    'code': code_short,
                    'industry': data[3] if len(data) > 3 else '',
                })
                break
        except Exception:
            pass

    if industry_records:
        pd.DataFrame(industry_records).to_csv(
            os.path.join(data_dir, "industry_info.csv"), index=False
        )

    bs.logout()
    n_valid = int(valid_mask.sum())
    n_days = len(ret_df)
    print(f"下载完成! 有效股票: {n_valid}, 交易日: {n_days}")


def load_data(data_dir='data'):
    """加载已下载的数据

    Returns:
        returns_df: 日收益率 DataFrame (日期 x 股票)
        stocks: 股票代码列表
        code_to_industry: {股票代码: 行业名称} 字典
    """
    returns = pd.read_csv(
        os.path.join(data_dir, 'returns_clean.csv'),
        index_col=0, parse_dates=True
    )
    industry = pd.read_csv(os.path.join(data_dir, 'industry_info.csv'))
    code_to_industry = dict(
        zip(industry['code'].astype(str).str.zfill(6), industry['industry'])
    )
    stocks = returns.columns.tolist()
    return returns, stocks, code_to_industry


def build_industry_prior(stocks, code_to_industry):
    """构建行业先验矩阵

    同行业股票对应位置为1，不同行业为0，对角线为0。

    Args:
        stocks: 股票代码列表
        code_to_industry: {股票代码: 行业名称} 字典

    Returns:
        prior: N x N 行业先验矩阵
    """
    n = len(stocks)
    prior = np.zeros((n, n))
    for i in range(n):
        ind_i = code_to_industry.get(stocks[i], f'_unknown_{i}')
        for j in range(i + 1, n):
            ind_j = code_to_industry.get(stocks[j], f'_unknown_{j}')
            if ind_i == ind_j and ind_i != 'Unknown':
                prior[i, j] = 1.0
                prior[j, i] = 1.0
    return prior
