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
    up_move = high - high
    down_move = low - low
    
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
        'ma6', 'ma24', 'rsi_ma6',
        'atr', 'sol_btc_ratio', 'sol_btc_ratio_ma6',  # 新增SOL/BTC比率特征
        'sol_eth_ratio', 'sol_eth_ratio_ma6',  # 新增SOL/ETH比率特征    
        'rsi', 'volume_ratio','macd_signal','macd',
        'price_change', 'price_change_abs', 'price_trend',
        'rsi_momentum', 'macd_histogram'
    ]

def generate_labels(data, future_window=4):
    """独立标签生成函数"""
    df = data.copy()
    # 使用未来窗口极值
    df['future_high'] = df['high'].rolling(future_window).max().shift(-future_window)
    df['future_low'] = df['low'].rolling(future_window).min().shift(-future_window)
    
    # 波动率阈值计算
    threshold_multiplier = 0.8
    threshold = df['atr'] / df['close'] * threshold_multiplier
    
    # 生成信号
    long_cond = df['future_high'] > df['close'] * (1 + threshold)
    short_cond = df['future_low'] < df['close'] * (1 - threshold)
    
    valid_signals = (long_cond | short_cond) & (df['future_high'].notnull())
    
    df['signal'] = 0
    df.loc[long_cond & valid_signals, 'signal'] = 1
    df.loc[short_cond & valid_signals, 'signal'] = 2

    # 生成止损止盈
    df['stop_loss'] = df['atr'].rolling(future_window).mean() * 1 / df['close']
    df['take_profit'] = df['atr'].rolling(future_window).mean() * 1.5 / df['close']
    
    return df[['signal', 'stop_loss', 'take_profit']]

def generate_labels_from_csv(df, future_window=4):
    """基于完整CSV数据生成标签（存储绝对收益率）"""
    labels = pd.DataFrame(index=df.index)
    
    # 计算未来价格变化
    future_close = df['close'].shift(-future_window)
    price_change = (future_close - df['close']) / df['close']
    
    # 生成信号（1: 做多，0: 做空）
    labels['signal'] = np.where(price_change > 0, 1, 0)
    
    # 存储绝对收益率
    labels['return_pct'] = np.abs(price_change)  # 修改为绝对值
    
    # 过滤无效数据
    valid_mask = (price_change != 0) & future_close.notnull()
    labels = labels[valid_mask]
    
    # 计算止损止盈（保持原有逻辑）
    labels['stop_loss'] = (df['close'] - df['low'].shift(-future_window).rolling(future_window).min()) / df['close']
    labels['take_profit'] = (df['high'].shift(-future_window).rolling(future_window).max() - df['close']) / df['close']
    
    return labels[['signal', 'stop_loss', 'take_profit', 'return_pct']].fillna(0)

def prepare_features(data, window_mode=False):

    # min_length = 72  # 与策略序列长度一致
    # if len(data) < min_length:
    #     raise ValueError(f"特征生成需要至少{min_length}根K线，当前输入: {len(data)}")
    """纯特征生成函数"""
    df = data.copy()
    df.index = pd.to_datetime(df.index)
    
    # === 技术指标计算 ===
    # 移动平均
    df['ma6'] = df['close'].rolling(6).mean()
    df['ma24'] = df['close'].rolling(24).mean()
    
    # 波动率特征
    df['atr'] = calculate_atr(df, 14)
    
    # 动量指标
    df['rsi'] = calculate_rsi(df['close'], 14)

    if window_mode:
        window_size = 24
        df['macd'] = df['close'].rolling(window=window_size, min_periods=window_size).apply(
            lambda x: calculate_macd(pd.Series(x))[0].iloc[-1]
        )
        df['macd_signal'] = df['close'].rolling(window=window_size, min_periods=window_size).apply(
            lambda x: calculate_macd(pd.Series(x))[1].iloc[-1]
        )
    else:
        macd, macd_signal = calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
    
    # 特征交叉
    df['rsi_ma6'] = df['rsi'] * df['ma6']
    # df['atr_macd'] = df['atr'] * df['macd']
    
    # 成交量特征
    df['volume_ratio'] = np.log(df['volume'] / df['volume'].rolling(6).mean() + 1e-6)
    
    # SOL/BTC比率特征
    df['sol_btc_ratio'] = df['close'] / df['btc_close']
    df['sol_btc_ratio_ma6'] = df['sol_btc_ratio'].rolling(6).mean()

    # SOL/ETH比率特征
    df['sol_eth_ratio'] = df['close'] / df['eth_close']
    df['sol_eth_ratio_ma6'] = df['sol_eth_ratio'].rolling(6).mean()


    # # === 新增策略信号 ===
    # # 1. 均线交叉策略（金叉/死叉）
    # df['ma_fast'] = df['close'].rolling(5).mean()  # 5周期均线
    # df['ma_slow'] = df['close'].rolling(20).mean() # 20周期均线
    # df['ma_cross'] = np.where(df['ma_fast'] > df['ma_slow'], 1, -1)
    
    # # 2. 布林带策略
    # df['boll_mid'] = df['close'].rolling(20).mean()
    # df['boll_std'] = df['close'].rolling(20).std()
    # df['boll_upper'] = df['boll_mid'] + 2*df['boll_std']  # 上轨
    # df['boll_lower'] = df['boll_mid'] - 2*df['boll_std']  # 下轨
    
    # # 3. KDJ指标
    # low_min = df['low'].rolling(9).min()
    # high_max = df['high'].rolling(9).max()
    # rsv = (df['close'] - low_min) / (high_max - low_min + 1e-6) * 100
    # df['kdj_k'] = rsv.ewm(com=2).mean()          # K线
    # df['kdj_d'] = df['kdj_k'].ewm(com=2).mean() # D线
    
    # # 4. OBV能量潮
    # df['price_change'] = df['close'].diff()
    # df['obv'] = np.sign(df['price_change']) * df['volume']
    # df['obv'] = df['obv'].cumsum()
    
    # # 5. CCI信号
    # cci = calculate_cci(df['high'], df['low'], df['close'])
    # df['cci_signal'] = np.where(cci > 100, 1, np.where(cci < -100, -1, 0))
    
    # # === 新增特征交互 ===
    # df['rsi_boll'] = df['rsi'] / (df['boll_upper'] - df['boll_lower'] + 1e-6)






    #  # === 新增策略指标 ===
    # # 1. SuperTrend指标突破百分比
    # atr_multiplier = 3
    # hl2 = (df['high'] + df['low']) / 2
    # super_upper = hl2 + atr_multiplier * df['atr']
    # super_lower = hl2 - atr_multiplier * df['atr']
    # df['super_upper_pct'] = (df['close'] - super_upper) / df['close']  # 上轨突破幅度
    # df['super_lower_pct'] = (super_lower - df['close']) / df['close']  # 下轨突破幅度

    # # 2. Donchian通道突破
    # donchian_window = 20
    # df['donchian_upper'] = df['high'].rolling(donchian_window).max()
    # df['donchian_lower'] = df['low'].rolling(donchian_window).min()
    # df['donchian_upper_pct'] = (df['close'] - df['donchian_upper']) / df['close']
    # df['donchian_lower_pct'] = (df['donchian_lower'] - df['close']) / df['close']

    # # 3. EMA交叉策略
    # df['ema12'] = df['close'].ewm(span=12).mean()
    # df['ema26'] = df['close'].ewm(span=26).mean()
    # df['ema_cross'] = np.where(df['ema12'] > df['ema26'], 1, -1)

    # # 4. 抛物线转向SAR差异
    # sar = df['close'].copy()
    # af, ep = 0.02, df['close'].iloc[0]
    # for i in range(2, len(df)):
    #     if df['close'].iloc[i] > ep:
    #         ep = df['close'].iloc[i]
    #         af = min(af + 0.02, 0.2)
    #     else:
    #         ep = df['close'].iloc[i]
    #         af = max(af - 0.02, 0.02)
    #     sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
    # df['sar_diff'] = (df['close'] - sar) / df['close']

    # # 5. VWAP偏离度
    # df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close'])/3).cumsum() / df['volume'].cumsum()
    # df['vwap_diff'] = (df['close'] - df['vwap']) / df['close']

  
  
    # # === 新增支撑压力特征 ===
    # # 近期价格极值（24小时窗口）
    # df['recent_high_72'] = df['high'].rolling(72).max()
    # df['recent_low_72'] = df['low'].rolling(72).min()
    
    # # 压力支撑强度（距离百分比）
    # df['resistance_strength'] = (df['recent_high_72'] - df['close']) / df['close']
    # df['support_strength'] = (df['close'] - df['recent_low_72']) / df['close']
    
    # # 突破压力位信号（结合波动率过滤）
    # df['breakout_signal'] = np.where(
    #     (df['close'] > df['recent_high_72']) & (df['atr'] > df['atr'].rolling(72).mean()),
    #     1, 0
    # )
    
    # # 斐波那契回撤位（基于最近波动）
    # swing_high = df['high'].rolling(48).max()
    # swing_low = df['low'].rolling(48).min()
    # df['fib_38'] = swing_low + (swing_high - swing_low) * 0.382
    # df['fib_50'] = swing_low + (swing_high - swing_low) * 0.5
    # df['fib_62'] = swing_low + (swing_high - swing_low) * 0.618

        # 价格变化特征
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = df['price_change'].abs()
    df['price_trend'] = df['price_change'].rolling(6).sum()

    # # 支撑阻力特征
    # df['resistance_level'] = df['high'].rolling(24).max()
    # df['support_level'] = df['low'].rolling(24).min()
    # df['distance_to_resistance'] = (df['resistance_level'] - df['close']) / df['close']
    # df['distance_to_support'] = (df['close'] - df['support_level']) / df['close']

    # # 动量确认特征
    df['rsi_momentum'] = df['rsi'] - df['rsi'].shift(3)
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    return df[get_feature_columns()].dropna()  # 添加必要的原始列
    