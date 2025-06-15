import pandas as pd
import numpy as np
from strategy import BaseStrategy

class ScalpingStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.position = 0
        
        # 策略参数（带默认值）
        self.fast_ema = self.params.get('fast_ema', 5)         # 快速EMA周期
        self.slow_ema = self.params.get('slow_ema', 15)        # 慢速EMA周期
        self.rsi_period = self.params.get('rsi_period', 14)    # RSI周期
        self.atr_period = self.params.get('atr_period', 10)    # ATR周期
        self.rsi_upper = self.params.get('rsi_upper', 70)      # RSI超买阈值
        self.rsi_lower = self.params.get('rsi_lower', 30)      # RSI超卖阈值
        self.atr_multiplier = self.params.get('atr_multiplier', 1.5)  # ATR倍数

    def calculate_signals(self):
        # 确保有足够数据计算指标
        min_length = max(self.slow_ema * 3, self.rsi_period * 2, self.atr_period * 2)
        if len(self.data) < min_length:
            return
            
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        current_price = close.iloc[-1]
        
        # 计算技术指标
        fast_ema = self._calc_ema(close, self.fast_ema)
        slow_ema = self._calc_ema(close, self.slow_ema)
        rsi = self._calc_rsi(close, self.rsi_period)
        atr = self._calc_atr(high, low, close, self.atr_period)
        
        # 获取当前值
        current_fast_ema = fast_ema.iloc[-1]
        current_slow_ema = slow_ema.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]
        
        # 信号生成逻辑
        signal, signal_type, take_profit, stop_loss = 'HOLD', '', 0, 0
        
        # 多头条件：EMA金叉 + RSI未超买
        if (fast_ema.iloc[-2] < slow_ema.iloc[-2] and 
            fast_ema.iloc[-1] > slow_ema.iloc[-1] and 
            current_rsi < self.rsi_upper):
            
            signal = 'LONG'
            signal_type = '做多'
            stop_loss = current_price - current_atr * self.atr_multiplier
            take_profit = current_price + current_atr * self.atr_multiplier * 2
        
        # 空头条件：EMA死叉 + RSI未超卖
        elif (fast_ema.iloc[-2] > slow_ema.iloc[-2] and 
              fast_ema.iloc[-1] < slow_ema.iloc[-1] and 
              current_rsi > self.rsi_lower):
            
            signal = 'SHORT'
            signal_type = '做空'
            stop_loss = current_price + current_atr * self.atr_multiplier
            take_profit = current_price - current_atr * self.atr_multiplier * 2
        
        # 波动率突破策略（独立信号）
        volatility_break = self._volatility_breakout(high, low, close, atr)
        if volatility_break == 'LONG' and current_rsi < 60:
            signal = 'LONG'
            signal_type = '做多'
            stop_loss = low.iloc[-1] - current_atr * 0.5
            take_profit = current_price + current_atr * 3
        
        elif volatility_break == 'SHORT' and current_rsi > 40:
            signal = 'SHORT'
            signal_type = '做空'
            stop_loss = high.iloc[-1] + current_atr * 0.5
            take_profit = current_price - current_atr * 3
        
        # 记录信号
        if signal != 'HOLD':
            self._record_signal(signal, signal_type, current_price, take_profit, stop_loss)

    def _calc_ema(self, series, period):
        """计算指数移动平均线"""
        return series.ewm(span=period, adjust=False).mean()

    def _calc_rsi(self, series, period):
        """计算相对强弱指数"""
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calc_atr(self, high, low, close, period):
        """计算平均真实波幅"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def _volatility_breakout(self, high, low, close, atr):
        """波动率突破策略"""
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        prev_close = close.iloc[-2]
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        
        # 计算突破阈值
        volatility_factor = atr.iloc[-2] * 0.6
        long_break = prev_high + volatility_factor
        short_break = prev_low - volatility_factor
        
        # 突破信号
        if current_high > long_break:
            return 'LONG'
        elif current_low < short_break:
            return 'SHORT'
        return 'HOLD'

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
        """更新多时间框架数据（保留接口）"""
        pass