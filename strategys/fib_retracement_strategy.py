import pandas as pd
import numpy as np
from strategy import BaseStrategy

class FibRetracementStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.position = 0
        
        # 参数优化：添加动态ATR计算和趋势确认
        self.fib_levels = self.params.get('fib_levels', [0.236, 0.382, 0.5, 0.618, 0.786])
        self.ma_window = self.params.get('ma_window', 20)
        self.atr_period = self.params.get('atr_period', 14)  # 新增ATR参数
        self.trend_ma_window = self.params.get('trend_ma_window', 50)  # 趋势确认均线
        
        # 多时间框架优化
        self.validation_frames = self.params.get('validation_frames', [])
        self.multi_tf_data = {}
        
        # 动态摆动点检测参数
        self.swing_period = self._calculate_swing_period()
        self.min_swing_distance = self.params.get('min_swing_distance', 0.03)  # 最小价格波动百分比

    def _calculate_swing_period(self):
        """根据K线周期自动计算摆动周期"""
        timeframe_map = {
            '5m': 288,     # 24小时
            '15m': 96,     # 24小时
            '30m': 48,     # 24小时
            '1h': 24,      # 24小时
            '4h': 84,       # 14天
            '1d': 30,       # 30天
        }
        return timeframe_map.get(self.timeframe, 24)

    def _calculate_atr(self):
        """计算平均真实波幅(ATR)"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        tr = pd.DataFrame(index=self.data.index)
        tr['h-l'] = high - low
        tr['h-pc'] = abs(high - close.shift(1))
        tr['l-pc'] = abs(low - close.shift(1))
        tr['tr'] = tr.max(axis=1)
        
        atr = tr['tr'].rolling(window=self.atr_period).mean()
        return atr.iloc[-1] if not atr.empty else 0

    def _find_valid_swings(self):
        """使用改进算法寻找有效的摆动高低点"""
        high = self.data['high']
        low = self.data['low']
        
        # 寻找摆动高点
        highs = []
        for i in range(2, len(high)):
            if high.iloc[i-1] > high.iloc[i-2] and high.iloc[i-1] > high.iloc[i]:
                highs.append((self.data.index[i-1], high.iloc[i-1]))
        
        # 寻找摆动低点
        lows = []
        for i in range(2, len(low)):
            if low.iloc[i-1] < low.iloc[i-2] and low.iloc[i-1] < low.iloc[i]:
                lows.append((self.data.index[i-1], low.iloc[i-1]))
        
        # 选择最近的有效摆动点
        swing_high = max(highs[-self.swing_period:], key=lambda x: x[1], default=(None, 0))[1]
        swing_low = min(lows[-self.swing_period:], key=lambda x: x[1], default=(None, float('inf')))[1]
        
        # 验证摆动点有效性
        if swing_high > 0 and swing_low < float('inf'):
            price_diff = (swing_high - swing_low) / swing_low
            if price_diff < self.min_swing_distance:
                return 0, 0  # 无效摆动点
        
        return swing_high, swing_low

    def _validate_multi_timeframe(self):
        """改进的多时间框架验证"""
        if not self.validation_frames:
            return True
        
        valid_count = 0
        for tf in self.validation_frames:
            if tf in self.multi_tf_data:
                tf_data = self.multi_tf_data[tf]
                if len(tf_data) > 20:
                    # 使用EMA判断趋势方向
                    ema_fast = tf_data['close'].ewm(span=12).mean()
                    ema_slow = tf_data['close'].ewm(span=26).mean()
                    
                    # 当前趋势方向
                    if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                        valid_count += 1
                    elif ema_fast.iloc[-1] < ema_slow.iloc[-1]:
                        valid_count -= 1
        
        # 要求多数时间框架趋势一致
        return valid_count > 0 if self.position >= 0 else valid_count < 0

    def calculate_signals(self):
        # 数据检查
        if len(self.data) < max(self.swing_period, self.ma_window, self.trend_ma_window):
            return
            
        close = self.data['close']
        current_price = close.iloc[-1]
        
        # 计算ATR用于动态阈值
        atr_value = self._calculate_atr()
        if atr_value == 0:
            return
            
        # 寻找有效的摆动点
        swing_high, swing_low = self._find_valid_swings()
        if swing_high <= swing_low:
            return
            
        # 计算斐波那契水平
        diff = swing_high - swing_low
        fib_prices = [swing_high - level * diff for level in self.fib_levels]
        
        # 计算趋势确认均线
        trend_ma = close.rolling(window=self.trend_ma_window).mean().iloc[-1]
        entry_ma = close.rolling(window=self.ma_window).mean().iloc[-1]
        
        # 多时间框架验证
        if not self._validate_multi_timeframe():
            return
            
        # 信号生成逻辑
        signal, signal_type, take_profit, stop_loss = self._generate_signal(
            current_price, entry_ma, trend_ma, fib_prices, diff, atr_value
        )

        # 记录信号
        if signal != 'HOLD':
            self._record_signal(signal, signal_type, current_price, take_profit, stop_loss)

    def _generate_signal(self, price, entry_ma, trend_ma, fib_prices, diff, atr_value):
        """生成交易信号 - 使用ATR动态阈值"""
        signal = 'HOLD'
        signal_type = ''
        take_profit = stop_loss = 0
        
        # 动态阈值（基于ATR）
        threshold = atr_value * 0.5
        
        # 趋势过滤
        if price > trend_ma:  # 多头趋势
            for level in fib_prices:
                if abs(price - level) <= threshold:
                    signal = 'LONG'
                    signal_type = '做多'
                    # 动态止盈止损（基于ATR）
                    take_profit = price + 2.5 * atr_value
                    stop_loss = price - 1.5 * atr_value
                    break
                    
        elif price < trend_ma:  # 空头趋势
            for level in reversed(fib_prices):
                if abs(price - level) <= threshold:
                    signal = 'SHORT'
                    signal_type = '做空' 
                    take_profit = price - 2.5 * atr_value
                    stop_loss = price + 1.5 * atr_value
                    break
                    
        return signal, signal_type, take_profit, stop_loss

    def _record_signal(self, signal, signal_type, price, tp, sl):
        """记录交易信号"""
        print(f"{signal_type}信号: 价格={price:.4f}, 止盈={tp:.4f}, 止损={sl:.4f}")
        self.position = 1 if signal == 'LONG' else -1
        new_signal = pd.DataFrame({
            'timestamp': [self.data.index[-1]],
            'signal': [signal_type],
            'price': [price],
            'take_profit': [tp],
            'stop_loss': [sl]
        })
        self.signals = pd.concat([self.signals, new_signal], ignore_index=True)

    def update_multi_timeframe(self, data_dict):
        """更新多时间框架数据"""
        self.multi_tf_data.update(data_dict)