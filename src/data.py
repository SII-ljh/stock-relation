"""数据下载、加载与预处理"""
import json
import os
import time
import warnings
import numpy as np
import pandas as pd


def normalize_industry(value):
    """Return a canonical industry name, or ``None`` for an unknown value."""
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {'unknown', 'nan', 'none', 'null'}:
        return None
    return value


def _read_constituents(constituents):
    """Load a constituent table supplied by the caller.

    A point-in-time constituent snapshot should be passed for historical
    studies.  The downloader still supports ``None`` for the quick-start
    flow, but marks that result as a current-universe snapshot.
    """
    if constituents is None:
        return None
    if isinstance(constituents, (str, os.PathLike)):
        frame = pd.read_csv(constituents)
    elif isinstance(constituents, pd.DataFrame):
        frame = constituents.copy()
    else:
        frame = pd.DataFrame(constituents)
    if 'code' not in frame.columns:
        raise ValueError("constituents must contain a 'code' column")
    frame['code'] = frame['code'].astype(str).str.strip()
    if frame['code'].isin({'', 'nan', 'None'}).any():
        raise ValueError("constituents contains an empty stock code")
    frame = frame.drop_duplicates('code').reset_index(drop=True)
    if frame.empty:
        raise ValueError("constituents must contain at least one stock")
    return frame


def download_data(data_dir='data', start='2020-01-01', end='2025-12-31',
                  constituents=None, clip_limit=None):
    """Download and clean daily HS300 data.

    ``constituents`` should be a point-in-time constituent table for a
    historical study.  Omitting it keeps the quick-start workflow available,
    but uses the provider's current snapshot and records that limitation in
    ``data_manifest.json``.
    """
    import baostock as bs
    from tqdm import tqdm

    os.makedirs(data_dir, exist_ok=True)
    if clip_limit is not None and (not np.isfinite(clip_limit) or clip_limit <= 0):
        raise ValueError("clip_limit must be a positive finite number or None")

    login = bs.login()
    print(f"baostock 登录: {login.error_code} {login.error_msg}")
    if str(login.error_code) != '0':
        raise RuntimeError(f"baostock login failed: {login.error_code} {login.error_msg}")

    try:
        print("获取沪深300成分股...")
        hs300_df = _read_constituents(constituents)
        if hs300_df is None:
            rs = bs.query_hs300_stocks()
            if str(rs.error_code) != '0':
                raise RuntimeError(f"HS300 query failed: {rs.error_code} {rs.error_msg}")
            hs300_list = []
            while rs.next():
                hs300_list.append(rs.get_row_data())
            hs300_df = pd.DataFrame(hs300_list, columns=rs.fields)
            universe_source = 'provider_current_snapshot'
            warnings.warn(
                "No point-in-time constituents supplied; the current HS300 "
                "universe is not unbiased for a historical backtest.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            universe_source = (
                'point_in_time_input'
                if 'industry' in hs300_df.columns
                else 'point_in_time_universe_current_industry'
            )
        if hs300_df.empty:
            raise RuntimeError("The constituent query returned no stocks")
        hs300_df.to_csv(os.path.join(data_dir, "hs300_constituents.csv"), index=False)
        print(f"成分股数量: {len(hs300_df)}")

        fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
        all_close, all_returns = {}, {}
        numeric_cols = ['open', 'high', 'low', 'close', 'preclose',
                        'volume', 'amount', 'turn', 'pctChg']
        for _, row in tqdm(hs300_df.iterrows(), total=len(hs300_df), desc="下载数据"):
            code = str(row['code'])
            code_short = code.replace("sh.", "").replace("sz.", "")
            try:
                rs = bs.query_history_k_data_plus(
                    code, fields, start_date=start, end_date=end,
                    frequency="d", adjustflag="2"
                )
                if str(rs.error_code) != '0':
                    tqdm.write(f"  {code}: {rs.error_code} {rs.error_msg}")
                    continue
                data_list = []
                while rs.next():
                    data_list.append(rs.get_row_data())
                if len(data_list) <= 100:
                    continue
                frame = pd.DataFrame(data_list, columns=rs.fields)
                for col in numeric_cols:
                    frame[col] = pd.to_numeric(frame[col], errors='coerce')
                frame.to_csv(os.path.join(data_dir, f"{code_short}.csv"), index=False)
                all_close[code_short] = frame.set_index('date')['close']
                all_returns[code_short] = frame.set_index('date')['pctChg'] / 100.0
            except Exception as exc:
                tqdm.write(f"  {code}: {str(exc)[:120]}")
            time.sleep(0.1)

        close_df = pd.DataFrame(all_close)
        if close_df.empty:
            raise RuntimeError("No stock price series were downloaded")
        close_df.index = pd.to_datetime(close_df.index)
        close_df = close_df.sort_index()
        valid_mask = close_df.isnull().mean() < 0.1
        if not bool(valid_mask.any()):
            raise RuntimeError("All downloaded stocks exceed the missing-data threshold")
        close_df.loc[:, valid_mask].to_csv(os.path.join(data_dir, "close_prices_valid.csv"))

        ret_df = pd.DataFrame(all_returns)
        ret_df.index = pd.to_datetime(ret_df.index)
        # Align return columns to the price universe before applying the mask;
        # a failed/short return query must not cause an indexing exception.
        ret_df = ret_df.sort_index().reindex(columns=close_df.columns).loc[:, valid_mask]
        ret_df.to_csv(os.path.join(data_dir, "returns.csv"))

        # A missing quote is not the previous day's return.  Filling with zero
        # makes the treatment explicit and avoids artificial serial correlation.
        ret_clean = ret_df.fillna(0.0)
        if clip_limit is not None:
            ret_clean = ret_clean.clip(-clip_limit, clip_limit)
        if not np.isfinite(ret_clean.to_numpy(dtype=float)).all():
            raise RuntimeError("Downloaded returns contain non-finite values")
        ret_clean.to_csv(os.path.join(data_dir, "returns_clean.csv"))

        print("获取行业分类...")
        industry_records = []
        if 'industry' in hs300_df.columns:
            for _, row in hs300_df.iterrows():
                code = str(row['code'])
                code_short = code.replace("sh.", "").replace("sz.", "")
                industry_records.append({
                    'code': code_short,
                    'industry': normalize_industry(row['industry']),
                })
        else:
            warnings.warn(
                "No point-in-time industry labels supplied; provider labels "
                "are a current snapshot.", RuntimeWarning, stacklevel=2,
            )
            for _, row in hs300_df.iterrows():
                code = str(row['code'])
                code_short = code.replace("sh.", "").replace("sz.", "")
                try:
                    rs = bs.query_stock_industry(code=code)
                    if str(rs.error_code) != '0':
                        continue
                    while rs.next():
                        data = rs.get_row_data()
                        industry_records.append({
                            'code': code_short,
                            'industry': normalize_industry(data[3] if len(data) > 3 else None),
                        })
                        break
                except Exception as exc:
                    print(f"  industry query failed for {code}: {exc}")
        if not industry_records:
            raise RuntimeError("No industry classifications were downloaded")
        pd.DataFrame(industry_records).to_csv(
            os.path.join(data_dir, "industry_info.csv"), index=False
        )

        manifest = {
            'start': str(start), 'end': str(end),
            'universe_source': universe_source, 'clip_limit': clip_limit,
            'n_stocks': int(valid_mask.sum()), 'n_days': int(len(ret_df)),
        }
        pd.Series(manifest, dtype=object).to_json(
            os.path.join(data_dir, 'data_manifest.json'), force_ascii=False
        )
        print(f"下载完成! 有效股票: {int(valid_mask.sum())}, 交易日: {len(ret_df)}")
    finally:
        bs.logout()


def load_data(data_dir='data', require_point_in_time=False):
    """加载已下载的数据

    Returns:
        returns_df: 日收益率 DataFrame (日期 x 股票)
        stocks: 股票代码列表
        code_to_industry: {股票代码: 行业名称} 字典
    """
    returns_path = os.path.join(data_dir, 'returns_clean.csv')
    industry_path = os.path.join(data_dir, 'industry_info.csv')
    manifest_path = os.path.join(data_dir, 'data_manifest.json')
    missing = [p for p in (returns_path, industry_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing generated data files: " + ", ".join(missing) +
            ". Run download_data() successfully first."
        )
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding='utf-8') as handle:
            manifest = json.load(handle)
        if manifest.get('universe_source') != 'point_in_time_input':
            message = (
                "data was downloaded from the provider's current universe; "
                "it is not suitable for an unbiased historical backtest"
            )
            if require_point_in_time:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    elif require_point_in_time:
        raise ValueError("data_manifest.json is required for point-in-time studies")
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    industry = pd.read_csv(industry_path)
    if returns.empty or returns.columns.duplicated().any():
        raise ValueError("returns_clean.csv must contain non-empty unique stock columns")
    if not returns.index.is_monotonic_increasing or returns.index.duplicated().any():
        raise ValueError("returns_clean.csv must have a sorted, unique date index")
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("returns_clean.csv contains NaN or infinite values")
    if not {'code', 'industry'}.issubset(industry.columns):
        raise ValueError("industry_info.csv must contain code and industry columns")
    code_to_industry = dict(
        zip(industry['code'].astype(str).str.zfill(6),
            industry['industry'].map(normalize_industry))
    )
    if require_point_in_time and not any(
            value is not None for value in code_to_industry.values()):
        raise ValueError(
            "point-in-time data requires at least one supplied industry label"
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
    if len(set(stocks)) != len(stocks):
        raise ValueError("stocks must contain unique identifiers")
    n = len(stocks)
    prior = np.zeros((n, n))
    labels = [normalize_industry(code_to_industry.get(stock)) for stock in stocks]
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] is not None and labels[i] == labels[j]:
                prior[i, j] = 1.0
                prior[j, i] = 1.0
    return prior
