import pandas as pd
import os
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import time

class DataAdapter:
    SOURCE_LOCAL = 'local'
    SOURCE_BINANCE = 'binance'
    
    def __init__(self, source, path=None, api_key=None, api_secret=None, mode='backtest'):
        self.source = source
        self.path = path
        if mode != 'backtest':
            self.binance_client = Client(api_key, api_secret) if api_key and api_secret else None
        
        # 创建本地存储目录
        if source == self.SOURCE_LOCAL and path and not os.path.exists(path):
            os.makedirs(path)

    def load_data(self, symbol, timeframe, start, end, btc_symbol=None, eth_symbol=None):

        """
        加载历史数据
        :param symbol: 交易对
        :param timeframe: K线周期
        :param start: 开始时间 (datetime)
        :param end: 结束时间 (datetime)
        :return: DataFrame with OHLCV data
        """
        print(f"self.source is ===>{self.source}")
        if self.source == self.SOURCE_LOCAL:
            main_df = self._load_from_local(symbol, timeframe, start, end)
            if btc_symbol:
                # 加载BTC数据并合并
                btc_df = self._load_from_local(btc_symbol, timeframe, start, end)
                main_df = main_df.join(btc_df[['close']].rename(columns={'close': 'btc_close'}), how='left')
                main_df['btc_close'].fillna(method='ffill', inplace=True)  # 前向填充
                main_df['btc_close'].fillna(method='bfill', inplace=True)  # 后向填充
            if eth_symbol:
                # 加载ETH数据并合并
                eth_df = self._load_from_local(eth_symbol, timeframe, start, end)
                main_df = main_df.join(eth_df[['close']].rename(columns={'close': 'eth_close'}), how='left')
                main_df['eth_close'].fillna(method='ffill', inplace=True)  # 前向填充
                main_df['eth_close'].fillna(method='bfill', inplace=True)  # 后向填充

            return main_df
        
        elif self.source == self.SOURCE_BINANCE:
            main_df = self._load_from_binance(symbol, timeframe, start, end)
            if btc_symbol:
                # 加载BTC数据并合并
                btc_df = self._load_from_binance('BTCUSDT', timeframe, start, end)
                main_df = main_df.join(btc_df[['close']].rename(columns={'close': 'btc_close'}), how='left')
                main_df['btc_close'].fillna(method='ffill', inplace=True)  # 前向填充
                main_df['btc_close'].fillna(method='bfill', inplace=True)  # 后向填充
            if eth_symbol:
                # 加载ETH数据并合并
                eth_df = self._load_from_binance('ETHUSDT', timeframe, start, end)
                main_df = main_df.join(eth_df[['close']].rename(columns={'close': 'eth_close'}), how='left')
                main_df['eth_close'].fillna(method='ffill', inplace=True)  # 前向填充
                main_df['eth_close'].fillna(method='bfill', inplace=True)  # 后向填充

            return main_df
        else:
            raise ValueError(f"Unsupported data source: {self.source}")
    
    def _load_from_local(self, symbol, timeframe, start, end):
        """从本地加载数据（适配新版数据结构）"""
        # 转换时间周期为文件名后缀
        tf_mapping = {
            '15m': '_15m',
            '30m': '_30m',
            '1h': '_1h',
            '4h': '_4h'
        }
        
        # 构建文件路径
        folder_name = f"{symbol.split('_')[0]}_USDT_USDT"  # 转换符号格式如 BTCUSDT -> BTC_USDT_USDT
        file_suffix = tf_mapping.get(timeframe, '')
        file_path = os.path.join(
            self.path,
            folder_name,
            f"{folder_name}{file_suffix}.xlsx"
        )
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local data file not found: {file_path}")
    
        # 读取Excel文件并处理格式
        df = pd.read_excel(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        return df.loc[start:end]
    
    def save_data_locally(self, symbol, timeframe, data):
        """保存数据到本地（适配新版格式）"""
        if not self.path:
            raise ValueError("Local path not specified")
        
        # 创建合约目录
        folder_name = f"{symbol.split('_')[0]}_USDT_USDT"
        folder_path = os.path.join(self.path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # 构建文件名
        tf_mapping = {
            '15m': '_15m',
            '30m': '_30m',
            '1h': '_1h',
            '4h': '_4h'
        }
        file_suffix = tf_mapping.get(timeframe, '')
        file_path = os.path.join(folder_path, f"{folder_name}{file_suffix}.xlsx")
        
        # 保存为Excel
        data.reset_index().to_excel(file_path, index=False)
        print(f"Data saved to {file_path}")
    
    def _load_from_binance(self, symbol, timeframe, start, end):
        """从币安API加载合约数据"""
        if not self.binance_client:
            raise RuntimeError("Binance client not initialized")
        
        data = pd.DataFrame()
        current_start = start
        
        # 添加日期验证
        if start >= end:
            raise ValueError("起始时间必须早于结束时间")
        
        max_retries = 5  # 最大重试次数
        retry_delay = 1  # 初始重试延迟秒数
        
        while current_start < end:
            current_end = min(current_start + timedelta(days=20), end)
            
            # 重试逻辑
            for attempt in range(max_retries):
                try:
                    klines = self.binance_client.futures_klines(
                        symbol=symbol,
                        interval=timeframe,
                        startTime=int(current_start.timestamp() * 1000),
                        endTime=int(current_end.timestamp() * 1000),
                        limit=480
                    )
                    break  # 成功则退出重试循环
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f"获取数据失败，已达最大重试次数: {str(e)}")
                    print(f"数据获取失败，第{attempt+1}次重试...")
                    time.sleep(retry_delay * (attempt + 1))
            
            if not klines:
                break
                
            # 列名保持与现货接口一致    
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # 后续处理保持不变...
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            data = pd.concat([data, df])
            current_start = current_end + timedelta(seconds=1)
            
            time.sleep(0.1)
        
        return data