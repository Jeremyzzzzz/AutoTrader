import torch
import numpy as np
from strategy import BaseStrategy
from data_adapter import DataAdapter
from sklearn.preprocessing import StandardScaler
from sol_nn_trainer import EnhancedSOLModel
from feature_utils import prepare_features, calculate_rsi, calculate_atr, get_feature_columns, is_filter_data
import pandas as pd
import torch.nn.functional as F
import datetime
import os
class NNStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.signals = pd.DataFrame(columns=[
            'timestamp', 
            'signal', 
            'price',
            'take_profit',
            'stop_loss'
        ])
        self.model = None
        self.scaler = None
        self.seq_length = 30
        print(f"[NNStrategy] 初始化策略，使用数据路径: {config.get('data_path')}")
        self._load_model()
        
        # 策略参数
        self.trade_threshold = 0.015  # 1.5% 阈值
        self.stop_loss_pct = 0.015    # 1.5% 止损
        self.take_profit_pct = 0.02   # 2% 止盈
        
        # 初始化数据适配器
        self.data_adapter = DataAdapter(
            source='local',
            path=config.get('data_path', '回测数据训练集'),
            mode=config.get('mode', 'backtest')
        )
    def _prepare_features(self, data):
        processed_df = prepare_features(data)
        feature_columns = get_feature_columns()  # 使用统一获取特征列的方法
        
        # 添加维度校验
        if processed_df.shape[1] != len(feature_columns):
            raise ValueError(f"特征维度不匹配！预期 {len(feature_columns)} 维，实际 {processed_df.shape[1]} 维")
        
        # 标准化时仅使用特征列
        scaled_data = self.scaler.transform(processed_df[feature_columns].values)
        return scaled_data[-self.seq_length:]
    def _load_model(self):
        """加载训练好的增强版神经网络模型"""
        print("[NNStrategy] 开始加载预训练模型...")
        print(f"self.symbol is ==>{self.symbol}")
        
        # 提取基础币种名称
        base_symbol = self.symbol.split('_')[0]
        model_path = os.path.join('model', f"{base_symbol}_USDT_USDT_nn_model.pth")
        
        # 添加安全全局变量（保持原有逻辑）
        import torch.serialization
        from sklearn.preprocessing._data import RobustScaler
        torch.serialization.add_safe_globals([RobustScaler])
        
        checkpoint = torch.load(
            model_path,  # 修改为动态路径
            map_location='cpu',
            weights_only=False
        )
        
        # 重建标准化器（保持原有逻辑）
        self.scaler = StandardScaler()
        self.scaler.mean_ = checkpoint['scaler_mean']
        self.scaler.scale_ = checkpoint['scaler_scale']
        
        # 修正模型加载逻辑（关键修改点）
        # 从checkpoint直接获取输入维度（通过特征列数量）
        feature_columns = get_feature_columns()
        input_size = len(feature_columns)
        # 从checkpoint直接获取序列长度
        self.seq_length = 30
        
        # 加载增强版模型结构（确保与训练器模型类一致）
        self.model = EnhancedSOLModel(input_size=input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"[NNStrategy] 增强模型加载完成，输入维度: {input_size}，序列长度: {self.seq_length}")

    def calculate_signals(self):
        """实现基类要求的信号计算方法"""
        latest_data = self.data.iloc[-1] if not self.data.empty else None
        if latest_data is None:
            print("[NNStrategy] 警告：没有可用数据")
            return
        
        result = self.on_bar(self.data)
        if not result:
            return
        signal, quantity, params = result
        
        take_profit, stop_loss = params if params else (None, None)
        new_signal = pd.DataFrame({
            'timestamp': [self.data.index[-1]],
            'signal': [signal],
            'price': [latest_data['close']],
            'take_profit': [take_profit],
            'stop_loss': [stop_loss]
        })

        # 过滤空值列
        new_signal = new_signal.dropna(axis=1, how='all')
        
        # 处理空DataFrame的情况
        if self.signals.empty:
            self.signals = new_signal
        else:
            self.signals = pd.concat(
                [self.signals, new_signal], 
                ignore_index=True
            ).infer_objects()
        
        print(f"[NNStrategy] 记录新信号: {signal} @ {latest_data['close']}")

    def _calculate_atr(self, df, period):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def on_bar(self, data):
        """接收最新行情数据并生成交易信号"""
        print(f"[NNStrategy] 收到新K线数据，时间: {data.index[-1]}")

        try:
            # 动态计算时间范围（获取过去30天数据）
            end_date = data.index[-1].to_pydatetime()
            start_date = end_date - datetime.timedelta(days=30)
            
            # 添加实时数据缓存（修复初始化问题）
            if not hasattr(self, 'cached_data') or self.cached_data.empty:
                hist_data = self.data_adapter.load_data(
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    start=start_date,
                    end=end_date
                )
                if hist_data.empty:
                    print("[NNStrategy] 警告：初始化数据加载失败")
                    return 'HOLD', 0, (None, None)
                self.cached_data = hist_data
            else:
                new_data = data.iloc[[-1]].copy()
                new_time = new_data.index[0]
                
                if new_time > self.cached_data.index[-1]:
                    self.cached_data = pd.concat([
                        self.cached_data,
                        new_data
                    ]).sort_index()

            # 使用缓存数据继续后续处理
            hist_data = self.cached_data[-self.seq_length*3:]
            
            if len(hist_data) < self.seq_length:
                print(f"[NNStrategy] 数据不足，需要{self.seq_length}条，当前{len(hist_data)}条")
                return 'HOLD', 0, (None, None)

            # 准备特征数据
            features = self._prepare_features(hist_data)
            
            # === 修改为直接使用 is_filter_data 函数 ===
            # 获取最新K线的原始数据
            last_open = hist_data['open'].iloc[-1]
            last_high = hist_data['high'].iloc[-1]
            last_low = hist_data['low'].iloc[-1]
            last_close = hist_data['close'].iloc[-1]
            
            # 直接调用过滤函数进行判断
            if not is_filter_data(last_high, last_low, last_close, last_open):
                print("[NNStrategy] 未通过过滤器条件，保持观望")
                return 'HOLD', 0, (0, 0)
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            with torch.no_grad():
            # 新模型输出包含信号概率和价格参数
                outputs = self.model(input_tensor).numpy()[0]

             # 修改信号概率计算方式
            signal_logits = outputs[:3]
            signal_probs = np.exp(signal_logits)  # 将log概率转换为实际概率
            signal_probs = signal_probs / signal_probs.sum()  # 确保概率和为1
            
            print(f"[DEBUG] 信号概率（观望/做多/做空）: {signal_probs}")
            stop_loss = outputs[3]     # 止损比例
            take_profit = outputs[4]    # 止盈比例
            
            # 获取预测信号 (取概率最高的类别)
            signal_idx = 0
            for idx in signal_probs:
                if idx > 0.95:
                    print(f"signal_probs is ===>{signal_probs}")
                    signal_idx = np.argmax(signal_probs)
            print(f"[DEBUG] 预测信号索引: {signal_idx}")
            entry_price = data['close'].iloc[-1]
            
            # 生成交易信号
            if signal_idx == 1:  # 做多
                return '做多', 0.5, (
                    entry_price * (1 + take_profit),
                    entry_price * (1 - stop_loss)
                )
            elif signal_idx == 2:  # 做空
                return '做空', 0.5, (
                    entry_price * (1 - take_profit),
                    entry_price * (1 + stop_loss)
                )
            else:  # 观望
                return 'HOLD', 0, (0, 0)

        except Exception as e:
            print(f"[NNStrategy] 处理K线时发生异常: {str(e)}")
            return 'HOLD', 0, (None, None)