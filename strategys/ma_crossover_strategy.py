import pandas as pd
from strategy import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.position = 0  # 新增初始化持仓状态
    """
    双均线交叉策略
    参数:
        - fast_ma: 快速均线周期
        - slow_ma: 慢速均线周期
    """
    def calculate_signals(self):
        if len(self.data) < max(self.params.get('slow_ma', 30), self.params.get('fast_ma', 10)):
            return
            
        # 计算均线
        close = self.data['close']
        fast_ma = close.rolling(window=self.params.get('fast_ma', 10)).mean()
        slow_ma = close.rolling(window=self.params.get('slow_ma', 30)).mean()
        
        # 生成信号
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2] if len(fast_ma) > 1 else current_fast
        prev_slow = slow_ma.iloc[-2] if len(slow_ma) > 1 else current_slow
        
        price = close.iloc[-1]  # <-- 这里添加price的定义
       # 修正后的信号生成逻辑
        signal = 'HOLD'
        
        # 金叉条件
        if prev_fast < prev_slow and current_fast > current_slow:
            signal = 'LONG' if self.position <= 0 else 'CLOSE_LONG'
        
        # 死叉条件 
        elif prev_fast > prev_slow and current_fast < current_slow:
            signal = 'SHORT' if self.position >= 0 else 'CLOSE_SHORT'
        
        # 更新持仓状态（在记录信号之前）
        if signal == 'LONG':
            self.position = 1
            take_profit = price * 1.03
            stop_loss = price * 0.985
            signal_type = '做多'  # 初始信号
        elif signal == 'SHORT':
            self.position = -1
            entry_price = close.iloc[-1]
            take_profit = price * 0.97
            stop_loss = price * 1.015
            signal_type = '做空'
        elif signal in ('CLOSE_LONG', 'CLOSE_SHORT'):
            self.position = 0

        # 记录信号
        if signal != 'HOLD':
            new_signal = pd.DataFrame({
                'timestamp': [self.data.index[-1]],
                'signal': [signal_type],
                'price': [price],
                'take_profit': [take_profit],
                'stop_loss': [stop_loss]
            })
            self.signals = pd.concat([self.signals, new_signal], ignore_index=True)