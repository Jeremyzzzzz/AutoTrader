import pandas as pd
import numpy as np
from strategy import BaseStrategy

class FibRetracementStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.position = 0
        
        # 从配置获取参数（带默认值）
        self.fib_levels = self.params.get('fib_levels', [0.236, 0.382, 0.5, 0.618, 0.786])
        self.ma_window = self.params.get('ma_window', 20)
        self.swing_period = self._calculate_swing_period()
        
        # 多时间框架支持
        self.validation_frames = self.params.get('validation_frames', [])
        self.multi_tf_data = {}  # 存储其他时间框架数据

    def _calculate_swing_period(self):
        """根据K线周期自动计算摆动周期"""
        timeframe_map = {
            '15m': 96,    # 24小时
            '30m': 48,    # 24小时
            '1h': 24,     # 24小时
            '4h': 42,     # 7天
        }
        return timeframe_map.get(self.timeframe, 24)

    def calculate_signals(self):
        # 主时间框架数据检查
        if len(self.data) < self.swing_period:
            return
            
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        
        # 计算摆动高低点（使用动态周期）
        swing_high = high[-self.swing_period:].max()
        swing_low = low[-self.swing_period:].min()
        diff = swing_high - swing_low
        
        # 多时间框架验证
        if self.validation_frames:
            for tf in self.validation_frames:
                if tf in self.multi_tf_data:
                    # 示例：检查周线趋势
                    weekly_data = self.multi_tf_data[tf]
                    if len(weekly_data) > 2:
                        # 确保主趋势与周线趋势一致
                        if weekly_data['close'][-1] < weekly_data['close'][-2]:
                            return  # 周线下跌时不建议做多

        # 斐波那契计算
        fib_prices = [swing_high - level * diff for level in self.fib_levels]
        
        # 均线计算
        ma = close.rolling(window=self.ma_window).mean()
        current_price = close.iloc[-1]

        # 信号生成逻辑
        signal, signal_type, take_profit, stop_loss = self._generate_signal(
            current_price, ma.iloc[-1], fib_prices, diff
        )

        # 记录信号
        if signal != 'HOLD':
            self._record_signal(signal, signal_type, current_price, take_profit, stop_loss)

    def _generate_signal(self, price, ma_value, fib_prices, diff):
        """生成交易信号"""
        signal = 'HOLD'
        signal_type = ''
        take_profit = stop_loss = 0

        if price > ma_value:  # 多头区域
            for level in fib_prices:
                if abs(price - level) <= level * 0.005:
                    signal = 'LONG'
                    signal_type = '做多'
                    take_profit = level + diff * 0.618  # 61.8%扩展
                    stop_loss = level - diff * 0.382    # 38.2%回撤
                    break
        else:  # 空头区域
            for level in reversed(fib_prices):
                if abs(price - level) <= level * 0.005:
                    signal = 'SHORT'
                    signal_type = '做空' 
                    take_profit = level - diff * 0.618
                    stop_loss = level + diff * 0.382
                    break
        return signal, signal_type, take_profit, stop_loss

    def _record_signal(self, signal, signal_type, price, tp, sl):
        """记录交易信号"""
        print(f"{signal_type}信号: 价格={price}, 止盈={tp}, 止损={sl}")
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