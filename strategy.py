from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, config):
        """
        策略基类初始化
        :param config: 策略配置字典
            - symbol: 交易对 (e.g. 'BTCUSDT')
            - timeframe: K线周期 (e.g. '1h')
            - params: 策略特定参数
        """
        self.symbol = config['symbol']
        self.timeframe = config['timeframe']
        self.params = config.get('params', {})
        self.data = pd.DataFrame()
        self.signals = pd.DataFrame(columns=[
            'timestamp', 
            'signal', 
            'price',
            'take_profit',  # 新增字段
            'stop_loss'     # 新增字段
        ])
        
    @abstractmethod
    def calculate_signals(self):
        """
        计算交易信号 - 子类必须实现
        计算结果应存储在self.signals DataFrame中:
            - timestamp: 信号时间
            - signal: 信号类型 ('BUY', 'SELL', 'HOLD')
            - price: 信号触发价格
        """
        pass
    
    def update_data(self, new_data):
        """
        更新策略数据
        :param new_data: 新的K线数据 (DataFrame)
        """
        # 合并新数据（修复数据累积问题）
        if not self.data.empty:
            # 仅保留新数据中比现有数据新的部分
            new_data = new_data[new_data.index > self.data.index[-1]]
        
        # 新增数据截断逻辑（保留最近72根K线）
        self.data = pd.concat([self.data, new_data]).iloc[-72:].sort_index()
        
        # 计算信号
        self.calculate_signals()
    
    def get_latest_signal(self):
        """
        获取最新交易信号
        :return: 最新信号 (dict) 或 None
        """
        if not self.signals.empty:
            return self.signals.iloc[-1].to_dict()
        return None
    
    def reset(self):
        """重置策略状态"""
        self.data = pd.DataFrame()
        self.signals = pd.DataFrame(columns=['timestamp', 'signal', 'price'])