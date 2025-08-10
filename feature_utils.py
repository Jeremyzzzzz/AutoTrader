import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import json
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def calculate_rsi(series, period=14):
        """手动计算RSI指标"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

def calculate_atr(df, period=14):
    """手动计算ATR指标"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def calculate_macd(series, fast=12, slow=26, signal=9):
    """手动计算MACD指标"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def calculate_cci(high, low, close, period=20):
    """手动计算CCI指标"""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    return cci

def calculate_adx(high, low, close, period=14):
    """手动计算ADX指标"""
    # 转换输入为pandas Series
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    # 计算方向运动
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    # 将numpy数组转换为pandas Series
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=low.index)

    
    # 计算真实波幅
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    # 平滑处理
    plus_dm_smooth = plus_dm.rolling(period).sum()
    minus_dm_smooth = minus_dm.rolling(period).sum()
    tr_smooth = tr.rolling(period).sum()
    
    # 计算方向指数
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # 计算方向指数差值和和
    di_diff = (plus_di - minus_di).abs()
    di_sum = plus_di + minus_di
    
    # 计算ADX
    dx = 100 * (di_diff / di_sum)
    adx = dx.rolling(period).mean()
    return adx

def get_feature_columns():
    """统一管理所有特征列（与训练代码完全一致）"""
    return [
        'open', 'high', 'low', 'close', 'volume',
        'ma6', 'ma24', 'ma72', 'rsi_ma6', 'atr_macd',
        'atr', 'sol_btc_ratio', 'sol_btc_ratio_ma6',  # 新增SOL/BTC比率特征
        'rsi', 'macd', 'macd_signal', 'volume_ratio','signal', 'stop_loss', 'take_profit','filter_data'
    ]

def is_filter_data(high, low, close, open):
    # """检测长上影线模式（实体较小且上影线较长）"""
    # # 计算实体大小
    # body_size = np.abs(close - open)
    # # 计算上影线长度
    # upper_shadow = high - np.maximum(close, open)
    # # 计算总波动范围
    # total_range = high - low
    
    # # 条件1：实体占比小于总波动的30%
    # condition1 = body_size / (total_range + 1e-6) < 0.3
    # # 条件2：上影线占比超过总波动的50%
    # condition2 = upper_shadow / (total_range + 1e-6) > 0.5
    # # 条件3：上影线绝对值超过ATR(14)的1.5倍（需要确保已计算ATR）
    # print(f"result is ==>{condition1 & condition2}")
    # return condition1 & condition2
    return True
def prepare_features(data):
    df = data.copy()
    # 新增基础校验
    if len(df) != 24:
        raise ValueError(f"特征计算只能24根K线，当前只有{len(df)}根")
    df.index = pd.to_datetime(df.index)
    
    # === 新增模式过滤器 ===
    # 标记所有出现条件的时点
    df['filter_data'] = is_filter_data(df['high'], df['low'], df['close'], df['open'])

    
    # 修正移动平均计算（增加滞后处理）
    df['ma6'] = df['close'].shift(1).rolling(6, min_periods=1).mean()
    df['ma24'] = df['close'].shift(1).rolling(24, min_periods=1).mean()
    df['ma72'] = df['close'].shift(1).rolling(72, min_periods=1).mean()

    # 波动率特征
    df['atr'] = calculate_atr(df, 14)
    # 修改波动率计算
    
    # 动量指标
    df['rsi'] = calculate_rsi(df['close'], 14)
    macd, macd_signal = calculate_macd(df['close'])  # 直接调用统一函数
    df['macd'] = macd
    df['macd_signal'] = macd_signal
        # 增加特征交叉
    df['rsi_ma6'] = df['rsi'] * df['ma6']
    df['atr_macd'] = df['atr'] * df['macd']
    # 改进成交量特征（增加对数变换）
    df['volume_ratio'] = np.log(df['volume'] / df['volume'].rolling(6).mean() + 1e-6)
    
        # === 新增标签生成 ===
    future_window = 4 # 可尝试4/8小时（需与训练代码同步修改）
    threshold = df['atr'] / df['close'] * 0.5 * (2)  # 增加时间因子

    
    # 价格目标
    df['future_price'] = df['close'].shift(-future_window)
    
    # 信号生成 (0:观望, 1:做多, 2:做空)
    df['signal'] = 0
    df.loc[df['future_price'] > df['close'] * (1 + threshold), 'signal'] = 1
    df.loc[df['future_price'] < df['close'] * (1 - threshold), 'signal'] = 2
    
    #  # === 新增数据校验 ===
    # print("\n=== 数据校验 ===")
    # print("最新5条SOL收盘价:")
    # print(df['close'].tail())
    # print("\n最新5条BTC收盘价:")
    # print(df['btc_close'].tail())
    # print(f"\n数据形状: {df.shape}, 缺失值数量: {df[['close', 'btc_close']].isnull().sum()}")

    # === 新增SOL/BTC价格比率特征 ===
    # 假设数据中包含BTC价格（需要确保数据适配器加载了BTC数据）
    df['sol_btc_ratio'] = df['close'] / df['btc_close']  # 添加SOL/BTC价格比率
    df['sol_btc_ratio_ma6'] = df['sol_btc_ratio'].rolling(6).mean()  # 比率6周期均线
    df['sol_btc_ratio_change'] = df['sol_btc_ratio'].pct_change(3)

    # 止损止盈目标 (基于波动率)
    df['stop_loss'] = df['atr'].rolling(future_window).mean() * 1 / df['close']  # 使用未来窗口计算波动率
    df['take_profit'] = df['atr'].rolling(future_window).mean() * 1.5 / df['close']

    # 创建有效训练窗口（出现信号前72小时 + 后24小时）
    window_before = 72  # 前导观察窗口
    window_after = 24   # 后续跟踪窗口
    df['valid_mask'] = False
    
    # 标记所有需要保留的样本
    for idx in df[df['filter_data']].index:
        start = idx - pd.Timedelta(hours=window_before)
        end = idx + pd.Timedelta(hours=window_after)
        df.loc[start:end, 'valid_mask'] = True
    
    # # 计算特征相关性
    # corr_matrix = df[get_feature_columns()].corr()
    
    # # 可视化保存
    # plt.figure(figsize=(20, 16))
    # sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    # plt.title('Feature Correlation Matrix')
    # plt.savefig('feature_correlation.png')
    # plt.close()/做多/做空）: [6.1025112e-06 2.7324775e-01 7.2674614e-01]
    # 最终返回时保留有效样本

    RECORD_TIME = '2025-03-10 15:00:00'
    if pd.to_datetime(RECORD_TIME) in df.index:
        debug_data = df.loc[[RECORD_TIME], get_feature_columns()].copy()
        debug_data['record_time'] = RECORD_TIME
        
        # === 修改移动平均计算明细记录方式 ===
        # 获取MA24计算窗口时间范围
        ma24_window = df['close'].shift(1).iloc[:24].dropna()
        debug_data['ma24_calculation'] = json.dumps({
            'window_size': len(ma24_window),
            'begin_time': str(ma24_window.index[0]),
            'end_time': str(ma24_window.index[-1])
        })
        
        # 获取MA72计算窗口时间范围
        ma72_window = df['close'].shift(1).iloc[:72].dropna()
        debug_data['ma72_calculation'] = json.dumps({
            'window_size': len(ma72_window),
            'begin_time': str(ma72_window.index[0]),
            'end_time': str(ma72_window.index[-1])
        })
        log_path = "fixed_feature_debug.csv"
        if os.path.exists(log_path):
            existing = pd.read_csv(log_path)
            if RECORD_TIME not in existing['record_time'].values:
                debug_data.to_csv(log_path, mode='a', header=False, index=False)
        else:
            debug_data.to_csv(log_path, index=False)

    result_df = df[df['valid_mask']][get_feature_columns()].dropna()
    
    # # === 新增特征日志 ===
    # log_dir = "logs/"
    # os.makedirs(log_dir, exist_ok=True)
    # log_file = f"{log_dir}/feature_log_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
    
    # # 添加索引到日志数据
    # log_df = result_df.reset_index()
    # log_df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
    
    return result_df