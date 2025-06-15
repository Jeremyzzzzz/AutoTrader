import pandas as pd
from strategy import BaseStrategy
import json
class UpperShadowStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.position = 0
        
        # 从配置文件加载参数
        with open('strategys/shadowConfig.json', 'r') as f:
            default_params = json.load(f)
        # 合并用户自定义参数
        self.params = {**default_params, **self.params}
        
        # 初始化策略参数
        self.upper_shadow_ratio = self.params['upper_shadow_ratio']
        self.lower_shadow_ratio = self.params['lower_shadow_ratio']
        self.entry_multiplier = self.params['entry_multiplier']
        self.tp_multiplier = self.params['tp_multiplier'] 
        self.sl_multiplier = self.params['sl_multiplier']

    def calculate_signals(self):
        if len(self.data) < 2:  # 至少需要2根K线
            return
            
        # 获取最近两根K线数据
        prev_bar = self.data.iloc[-2]
        current_bar = self.data.iloc[-1]
        
        # 计算前一根K线特征
        body = abs(prev_bar['close'] - prev_bar['open'])
        upper_shadow = prev_bar['high'] - max(prev_bar['close'], prev_bar['open'])
        lower_shadow = min(prev_bar['close'], prev_bar['open']) - prev_bar['low']
        
        # 检查长上影线条件
        if body > 0 and \
           upper_shadow >= body * self.upper_shadow_ratio and \
           lower_shadow <= body * self.lower_shadow_ratio:
            
            # 使用当前K线开盘价作为基准
            price_range = prev_bar['high'] - prev_bar['open']
            entry_price = prev_bar['open'] + self.entry_multiplier * price_range
            # 计算交易参数
            take_profit = prev_bar['open'] - price_range * self.tp_multiplier
            stop_loss = prev_bar['open'] + price_range * self.sl_multiplier
            
            # 生成做空信号
            self._record_signal('SHORT', '做空', entry_price, take_profit, stop_loss)
        else:
            self._record_signal('HOLD', '', 0, 0, 0)
        
    def _record_signal(self, signal, signal_type, price, tp, sl):
        """记录交易信号"""
        self.position = -1  # 做空仓位
        new_signal = pd.DataFrame({
            'timestamp': [self.data.index[-1]],
            'signal': [signal_type],
            'price': [price],
            'take_profit': [tp],
            'stop_loss': [sl]
        })
        self.signals = pd.concat([self.signals, new_signal], ignore_index=True)