"""
使用baostock下载沪深300成分股历史日线数据
baostock连接稳定，适合批量下载
"""
import baostock as bs
import pandas as pd
import os
import time
from tqdm import tqdm

DATA_DIR = "/inspire/hdd/project/global-event-perception-and-prediction/liaojianhan-CZXS24220039/stock-relation/data"
os.makedirs(DATA_DIR, exist_ok=True)

# 登录baostock
lg = bs.login()
print(f"Login: {lg.error_code} {lg.error_msg}")

# Step 1: 获取沪深300成分股（从baostock获取最新一期）
print("获取沪深300成分股...")
rs = bs.query_hs300_stocks()
hs300_list = []
while (rs.error_code == '0') and rs.next():
    hs300_list.append(rs.get_row_data())

hs300_df = pd.DataFrame(hs300_list, columns=rs.fields)
print(f"成分股数量: {len(hs300_df)}")
print(f"示例: {hs300_df['code'].head().tolist()}")
hs300_df.to_csv(os.path.join(DATA_DIR, "hs300_baostock.csv"), index=False)

# Step 2: 下载每只股票的日线数据
START = "2020-01-01"
END = "2025-12-31"
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"

success = 0
fail = 0
all_close = {}
all_volume = {}
all_returns = {}

for _, row in tqdm(hs300_df.iterrows(), total=len(hs300_df), desc="下载"):
    code = row['code']  # 格式: sh.600519 或 sz.000001
    code_short = code.replace("sh.", "").replace("sz.", "")
    
    filepath = os.path.join(DATA_DIR, f"{code_short}.csv")
    
    try:
        rs = bs.query_history_k_data_plus(
            code, FIELDS,
            start_date=START, end_date=END,
            frequency="d", adjustflag="2"  # 前复权
        )
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        
        if len(data_list) > 100:
            df = pd.DataFrame(data_list, columns=rs.fields)
            # 转数值
            for col in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.to_csv(filepath, index=False)
            
            # 收集收盘价和收益率
            series = df.set_index('date')['close']
            series.name = code_short
            all_close[code_short] = series
            
            ret_series = df.set_index('date')['pctChg'] / 100.0
            ret_series.name = code_short
            all_returns[code_short] = ret_series
            
            vol_series = df.set_index('date')['volume']
            vol_series.name = code_short
            all_volume[code_short] = vol_series
            
            success += 1
        else:
            fail += 1
            tqdm.write(f"  {code}: 数据不足 ({len(data_list)} 行)")
    except Exception as e:
        fail += 1
        tqdm.write(f"  {code}: 错误 {str(e)[:50]}")
    
    time.sleep(0.1)  # 小间隔

print(f"\n下载完成: 成功 {success}, 失败 {fail}")

# Step 3: 合并数据
print("\n合并收盘价...")
close_df = pd.DataFrame(all_close)
close_df.index = pd.to_datetime(close_df.index)
close_df = close_df.sort_index()

# 过滤缺失太多的
valid_mask = close_df.isnull().mean() < 0.1
close_valid = close_df.loc[:, valid_mask]

close_df.to_csv(os.path.join(DATA_DIR, "close_prices.csv"))
close_valid.to_csv(os.path.join(DATA_DIR, "close_prices_valid.csv"))

# 收益率
ret_df = pd.DataFrame(all_returns)
ret_df.index = pd.to_datetime(ret_df.index)
ret_df = ret_df.sort_index()
ret_df = ret_df.loc[:, valid_mask]
ret_df.to_csv(os.path.join(DATA_DIR, "returns.csv"))

# 成交量
vol_df = pd.DataFrame(all_volume)
vol_df.index = pd.to_datetime(vol_df.index)
vol_df = vol_df.sort_index()
vol_df.to_csv(os.path.join(DATA_DIR, "volume_data.csv"))

print(f"全部: {close_df.shape[1]} 只, 有效(<10%缺失): {close_valid.shape[1]} 只")
print(f"交易日: {close_df.shape[0]}, 范围: {close_df.index.min()} ~ {close_df.index.max()}")
print(f"收益率矩阵: {ret_df.shape}")

# 获取行业分类信息
print("\n获取行业分类...")
industry_map = {}
for _, row in hs300_df.iterrows():
    code = row['code']
    code_short = code.replace("sh.", "").replace("sz.", "")
    try:
        rs = bs.query_stock_industry(code=code)
        while (rs.error_code == '0') and rs.next():
            data = rs.get_row_data()
            industry_map[code_short] = {
                'code': code_short,
                'name': row.get('code_name', ''),
                'industry': data[3] if len(data) > 3 else '',
                'industryClassification': data[4] if len(data) > 4 else '',
            }
            break
    except:
        pass

if industry_map:
    ind_df = pd.DataFrame(industry_map.values())
    ind_df.to_csv(os.path.join(DATA_DIR, "industry_info.csv"), index=False)
    print(f"行业信息: {len(ind_df)} 只股票")
    print(f"行业分布: {ind_df['industry'].value_counts().head(10).to_dict()}")

bs.logout()
print("\n所有数据下载完成!")
