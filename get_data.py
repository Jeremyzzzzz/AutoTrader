# fetch_binance_data_final.py
import ccxt
import pandas as pd
import time
from datetime import datetime
import os
import re

def sanitize_filename(name):
    """清理非法文件名字符"""
    return re.sub(r'[\\/*?:"<>|:]', '_', name)

def fetch_ohlcv(exchange, symbol, timeframe, start_date, end_date):
    """获取指定时间范围的OHLCV数据"""
    since = exchange.parse8601(start_date + 'T00:00:00Z')
    end_timestamp = exchange.parse8601(end_date + 'T23:59:59Z')
    
    all_ohlcv = []
    current_since = since
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            if not ohlcv:
                break
            last_timestamp = ohlcv[-1][0]
            if last_timestamp > end_timestamp:
                ohlcv = [c for c in ohlcv if c[0] <= end_timestamp]
                if not ohlcv:
                    break
            all_ohlcv += ohlcv
            if last_timestamp >= end_timestamp:
                break
            current_since = last_timestamp + 1
            time.sleep(exchange.rateLimit / 1000)
        except ccxt.NetworkError:
            print("网络错误，5秒后重试...")
            time.sleep(5)
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)
    return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

def fetch_funding_rate(exchange, exchange_symbol, start_date, end_date):
    """获取资金费率历史数据"""
    since = exchange.parse8601(start_date + 'T00:00:00Z')
    end_timestamp = exchange.parse8601(end_date + 'T23:59:59Z')
    
    all_rates = []
    while True:
        try:
            rates = exchange.fapiPublicGetFundingRate({
                'symbol': exchange_symbol,
                'startTime': int(since),
                'limit': 1000
            })
            if isinstance(rates, dict) and 'code' in rates:
                print(f"API错误: {rates['msg']} (代码: {rates['code']})")
                break
            if not isinstance(rates, list):
                print(f"响应格式异常: {type(rates)}")
                break
            if not rates:
                break
            
            valid_rates = []
            for r in rates:
                try:
                    funding_time = int(r['fundingTime'])
                    if funding_time <= end_timestamp:
                        valid_rates.append(r)
                except KeyError:
                    continue
            all_rates += valid_rates
            if len(rates) < 1000:
                break
            since = int(rates[-1]['fundingTime']) + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"获取资金费率失败: {str(e)}")
            break
    
    if not all_rates:
        return pd.DataFrame()
    
    data = []
    for rate in all_rates:
        try:
            data.append({
                'datetime': pd.to_datetime(int(rate['fundingTime']), unit='ms', utc=True).tz_localize(None),
                'funding_rate': float(rate['fundingRate'])
            })
        except KeyError:
            continue
    return pd.DataFrame(data)

def main():
    # 初始化交易所
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    # 加载市场数据
    exchange.load_markets()
                                
    # 配置参数
    start_date = '2018-03-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 创建主目录
    base_dir = 'SOL回测结果训练集'
    os.makedirs(base_dir, exist_ok=True)
    
    # 获取所有USDT永续合约
    perpetual_markets = []
    for m in exchange.markets.values():
        info = m.get('info', {})
        if info.get('contractType') == 'PERPETUAL' and m['quote'] == 'USDT' and m['active']:
            perpetual_markets.append(m)
    
    print(f"发现{len(perpetual_markets)}个USDT永续合约")
    
    # 遍历所有永续合约
    for market in perpetual_markets:
        symbol_ccxt = market['symbol']
        if symbol_ccxt not in ["SOL/USDT:USDT"]:
            continue
        exchange_symbol = market['id']
        # 清理非法字符并标准化文件名
        symbol_folder = sanitize_filename(symbol_ccxt.replace('/', '_'))
        symbol_dir = os.path.join(base_dir, symbol_folder)
        os.makedirs(symbol_dir, exist_ok=True)
        
        print(f"\n{'='*30}")
        print(f"开始处理币种: {symbol_ccxt}")
        start_time = time.time()
        
        try:
            # 定义需要获取的时间框架
            timeframes = ['1h', '4h', '30m', '15m', '5m']
            
            # 获取K线数据
            for tf in timeframes:
                print(f"正在获取 {tf} K线数据...")
                df = fetch_ohlcv(exchange, symbol_ccxt, tf, start_date, end_date)
                file_name = f"{sanitize_filename(symbol_ccxt.replace('/', '_'))}_{tf}.xlsx"
                file_path = os.path.join(symbol_dir, file_name)
                df.to_excel(file_path, index=False)
                print(f"{tf}数据已保存，记录数：{len(df)}")
            
            # 获取资金费率
            print("正在获取资金费率数据...")
            funding_df = fetch_funding_rate(exchange, exchange_symbol, start_date, end_date)
            if not funding_df.empty:
                funding_name = f"{sanitize_filename(symbol_ccxt.replace('/', '_'))}_funding_rates.xlsx"
                funding_path = os.path.join(symbol_dir, funding_name)
                funding_df.to_excel(funding_path, index=False)
                print(f"资金费率数据已保存，记录数：{len(funding_df)}")
            else:
                print("警告：未获取到资金费率数据")
            
            # 显示时间范围
            print("\n数据时间范围验证:")
            for tf in timeframes:
                file_name = f"{sanitize_filename(symbol_ccxt.replace('/', '_'))}_{tf}.xlsx"
                df = pd.read_excel(os.path.join(symbol_dir, file_name))
                print(f"{tf.ljust(4)}数据: {df.datetime.min()} - {df.datetime.max()}")
            if not funding_df.empty:
                print(f"资金费率数据: {funding_df.datetime.min()} - {funding_df.datetime.max()}")
            
            # 计算耗时
            elapsed = time.time() - start_time
            print(f"完成处理 {symbol_ccxt}，耗时: {elapsed:.2f}秒")
            
        except Exception as e:
            print(f"处理 {symbol_ccxt} 时发生错误: {str(e)}")
            continue

if __name__ == "__main__":
    main()