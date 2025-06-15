import pandas as pd
import numpy as np

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
        'returns', 'log_ret',
        'ma6', 'ma24', 'ma72', 'ma_diff_short', 'ma_diff_long',
        'atr', 'volatility',
        'rsi', 'macd', 'macd_signal',
        'volume_ma', 'volume_ratio', 'obv','signal','stop_loss', 'take_profit','filter_data'
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
    df.index = pd.to_datetime(df.index)
    
    # === 新增模式过滤器 ===
    
    # 标记所有出现条件的时点
    df['filter_data'] = is_filter_data(df['high'], df['low'], df['close'], df['open'])

    # 价格特征
    df['returns'] = df['close'].pct_change()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
      # 修改移动平均计算方式（移除shift）
    df['ma6'] = df['close'].rolling(6, min_periods=1).mean()  # 添加min_periods
    df['ma24'] = df['close'].rolling(24, min_periods=1).mean()
    df['ma72'] = df['close'].rolling(72, min_periods=1).mean()
    df['ma_diff_short'] = df['ma6'] - df['ma24']
    df['ma_diff_long'] = df['ma24'] - df['ma72']
    
    # 波动率特征
    df['atr'] = calculate_atr(df, 14)
    # 修改波动率计算
    df['volatility'] = df['close'].rolling(24, min_periods=1).std()
    
    # 动量指标
    df['rsi'] = calculate_rsi(df['close'], 14)
    macd, macd_signal = calculate_macd(df['close'])  # 直接调用统一函数
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    
    # 成交量特征
    # 修改成交量特征计算
    df['volume_ma'] = df['volume'].rolling(6, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1e-6)  # 防止除零
    
    df['obv'] = (np.sign(df['close'].diff().fillna(0)) * df['volume']).cumsum()
    
        # === 新增标签生成 ===
    future_window = 6  # 预测未来6小时
    threshold = 0.01  # 1%阈值
    atr_multiplier = 1.5
    
    # 价格目标
    df['future_price'] = df['close'].shift(-future_window)
    
    # 信号生成 (0:观望, 1:做多, 2:做空)
    df['signal'] = 0
    df.loc[df['future_price'] > df['close'] * (1 + threshold), 'signal'] = 1
    df.loc[df['future_price'] < df['close'] * (1 - threshold), 'signal'] = 2
    
    # 止损止盈目标 (基于波动率)
    df['stop_loss'] = df['atr'] * atr_multiplier / df['close']  # 止损比例
    df['take_profit'] = df['atr'] * 3.0 / df['close']           # 止盈比例

    # 创建有效训练窗口（出现信号前72小时 + 后24小时）
    window_before = 48  # 前导观察窗口
    window_after = 24   # 后续跟踪窗口
    df['valid_mask'] = False
    
    # 标记所有需要保留的样本
    for idx in df[df['filter_data']].index:
        start = idx - pd.Timedelta(hours=window_before)
        end = idx + pd.Timedelta(hours=window_after)
        df.loc[start:end, 'valid_mask'] = True
    
    # 最终返回时保留有效样本
    return df[df['valid_mask']][get_feature_columns()].dropna()