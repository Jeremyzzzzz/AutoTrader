import torch
import numpy as np
from strategy import BaseStrategy
from data_adapter import DataAdapter
from sklearn.preprocessing import StandardScaler
from sol_nn_trainer import EnhancedSOLModel
from feature_utils import prepare_features, calculate_rsi, calculate_atr, get_feature_columns
import pandas as pd
import torch.nn.functional as F
import datetime
import os
class NNStrategy(BaseStrategy):
    def __init__(self, config, data_adapter):
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
        self.seq_length = 24
        print(f"[NNStrategy] 初始化策略，使用数据路径: {config.get('data_path')}")
        self._load_model()
        # self._load_risk_model() 
        # 策略参数
        self.trade_threshold = 0.015  # 1.5% 阈值
        self.stop_loss_pct = 0.015    # 1.5% 止损
        self.take_profit_pct = 0.02   # 2% 止盈
        self.last_blocked_signal = None
        # 初始化数据适配器
        self.data_adapter = data_adapter
        # 初始化日志路径
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"nn_signals_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx")
        self.lastSignal = None
        # self.alert_model = self._load_alert_model()  # 新增预警模型加载
        self.btc_symbol = config.get('btc_symbol', 'BTC_USDT_USDT')  # 新增BTC交易对
        self.processed_df = None
        self.future_window = 4  # 与训练器保持一致

    def _load_alert_model(self):
        """加载训练好的风险预警模型"""
        import joblib
        base_symbol = self.symbol.split('_')[0]
        model_path = os.path.join('model', f"SOL_USDT_USDT_alert_model.pkl")
        return joblib.load(model_path)

    def _log_signal(self, prob_long, prob_short, kline_data):
        """记录信号日志"""
        timestamp = kline_data.name.strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = {
        'timestamp': timestamp,
        'open': kline_data['open'],
        'high': kline_data['high'],
        'low': kline_data['low'],
        'close': kline_data['close'],
        'volume': kline_data['volume'],
        'prob_long': prob_long,  # 做多概率
        'prob_short': prob_short  # 做空概率
        }
    
        df = pd.DataFrame([log_entry])
        
        if not os.path.exists(self.log_file):
            df.to_excel(self.log_file, index=False, engine='openpyxl')
        else:
            with pd.ExcelWriter(
                self.log_file,
                engine='openpyxl',
                mode='a',
                if_sheet_exists='overlay'
            ) as writer:
                df.to_excel(
                    writer, 
                    index=False,
                    header=False,
                    startrow=writer.sheets['Sheet1'].max_row
                )

    def _prepare_features(self, data):
        # 记录原始数据信息
        print(f"[特征准备] 输入数据形状: {data.shape}, 时间范围: {data.index.min()} - {data.index.max()}")
        valid_data = data.iloc[:-self.future_window] if len(data) > self.future_window else data
        processed_df = prepare_features(valid_data)
        feature_columns = get_feature_columns()
        
        # 新增数据校验日志
        print(f"[特征校验] 处理后数据形状: {processed_df.shape}")
        print(f"[特征校验] 特征列匹配检查: {set(feature_columns).issubset(processed_df.columns)}")
        
        # 标准化前检查
        if processed_df[feature_columns].shape[0] == 0:
            print("[致命错误] 空特征矩阵！可能原因：")
            print("1. prepare_features返回空DataFrame")
            print("2. 特征列不匹配")
            print("3. 数据过滤过严")
            print("4. 原始数据不足（当前长度 {}）".format(len(data)))
        
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
        self.scaler.mean_ = checkpoint['scaler_mean']  # 新增均值加载
        self.scaler.scale_ = checkpoint['scaler_scale']  # 新增标准差加载
        # 修正模型加载逻辑（关键修改点）
        # 从checkpoint直接获取输入维度（通过特征列数量）
        feature_columns = get_feature_columns()
        print(f"\n[策略特征校验] 加载模型使用的特征数量: {len(feature_columns)}")
        print("特征列表:", feature_columns)
        input_size = len(feature_columns)
        # 从checkpoint直接获取序列长度
        self.seq_length = 24  # 替换原来的硬编码30
        
        # 加载增强版模型结构（确保与训练器模型类一致）
        self.model = EnhancedSOLModel(input_size=input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])  # 添加缺失的权重加载
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
        
        # 修改参数解包方式，添加持仓时间
        take_profit, stop_loss, holding_time = (params + (100,))[:3]  # 默认持仓2小时
        new_signal = pd.DataFrame({
            'timestamp': [self.data.index[-1]],
            'signal': [signal],
            'price': [latest_data['close']],
            'take_profit': [take_profit],
            'stop_loss': [stop_loss],
            'holding_time': [holding_time]  # 新增持仓时间字段
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

    def _load_risk_model(self):
        """加载风险预警模型"""
        base_symbol = self.symbol.split('_')[0]
        model_path = os.path.join('model', f"{base_symbol}_USDT_USDT_alarm_model.pth")
        
        # 添加安全全局变量（关键修复）
        import torch.serialization
        from numpy._core.multiarray import _reconstruct
        torch.serialization.add_safe_globals([_reconstruct])
        
        checkpoint = torch.load(
            model_path,
            map_location='cpu',
            weights_only=False  # 保持与主模型加载方式一致
        )
        
        self.risk_scaler = StandardScaler()
        self.risk_scaler.mean_ = checkpoint['scaler_mean']
        self.risk_scaler.scale_ = checkpoint['scaler_scale']
        
        feature_columns = get_feature_columns()
        # self.risk_model = RiskAlertModel(len(feature_columns))
        # self.risk_model.load_state_dict(checkpoint['model_state_dict'])
        # self.risk_model.eval()

    # def _risk_check(self, signal_type, entry_price, cached_data):
    #     """使用神经网络进行风险预测"""
    #     # 准备输入数据
    #     features = prepare_features(cached_data)
    #     scaled_features = self.risk_scaler.transform(features[get_feature_columns()].values[-self.seq_length:])
    #     input_tensor = torch.FloatTensor(scaled_features).unsqueeze(0)
        
    #     with torch.no_grad():
    #         # 修改输出处理逻辑
    #         output = self.risk_model(input_tensor)
    #         # 根据信号类型获取对应风险概率
    #         if signal_type == '做多':
    #             risk_prob = output[0][1].item()  # 取做多风险概率
    #         elif signal_type == '做空':
    #             risk_prob = output[0][2].item()  # 取做空风险概率
    #         else:
    #             risk_prob = 0.0
        
    #     # 风险阈值可根据验证集表现调整
    #     print(f"[NNStrategy] 预测的{signal_type}风险概率: {risk_prob:.2%}")
    #     return risk_prob > 0.95  # 示例阈值
    # 生成交易信号前添加风险检查

    # def _risk_check(self, signal_type, entry_price, cached_data):
    #     """风险控制过滤器"""
    #     # 1. 波动率检查（过去24小时ATR）
    #     atr_4h = self._calculate_atr(cached_data[-24:], period=8).iloc[-1]
    #     current_range = cached_data['high'].iloc[-1] - cached_data['low'].iloc[-1]
        
    #     # 2. 趋势强度检查（72周期EMA斜率）
    #     ma72 = cached_data['close'].rolling(72).mean()
    #     ma72_slope = (ma72.iloc[-1] - ma72.iloc[-6]) / 5  # 最近5根K线斜率
        
    #     # 3. 短期超买超卖检查（6小时RSI）
    #     rsi_6h = calculate_rsi(cached_data['close'], period=6).iloc[-1]
        
    #     # 风险规则（可根据需要调整参数）
    #     risk_conditions = [
    #         current_range > atr_4h * 1.5,          # 当前波动超过平均波动1.5倍
    #         abs(ma72_slope) < 0.0005,               # 中期趋势不明朗
    #         (signal_type == '做多' and rsi_6h > 70) or 
    #         (signal_type == '做空' and rsi_6h < 30) # 短期超买超卖
    #     ]
        
    #     return any(risk_conditions)

    def _generate_signal_probs(self, data):
        """统一信号生成逻辑（与训练阶段保持一致）"""
        # 数据预处理
        features = self._prepare_features(data)
        
        # 创建序列（与训练阶段相同的逻辑）
        seq_length = self.seq_length
        if len(features) < seq_length:
            return None, None
        
        # 取最近seq_length个特征
        sequence = features[-seq_length:]
        
        # 转换为tensor
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # 添加batch维度
        
        # 模型推理（禁用梯度计算）
        with torch.no_grad():
            output = self.model(input_tensor)
            signal_prob = torch.sigmoid(output[0, 0]).item()  # 做多概率
        
        return signal_prob, 1 - signal_prob  # (做多概率, 做空概率)

    # 修改on_bar方法
    def on_bar(self, data):
        """接收最新行情数据并生成交易信号（统一版）"""
        # 更新数据缓存
        if not hasattr(self, 'cached_data') or self.cached_data.empty:
            self.cached_data = data
        else:
            last_cached_time = self.cached_data.index[-1]
            if data.index[-1] > last_cached_time:
                self.cached_data = pd.concat([self.cached_data, data.iloc[[-1]]])
        
        # 保留最近5倍序列长度的数据
        self.cached_data = self.cached_data[-self.seq_length*5:]
        
        # 检查数据长度
        if len(self.cached_data) < self.seq_length:
            return 'HOLD', 0, (None, None, None)
        
        # 统一信号生成
        long_prob, short_prob = self._generate_signal_probs(self.cached_data)
        print(f"[NNStrategy] 生成信号 - 做多概率: {long_prob:.2%}, 做空概率: {short_prob:.2%}")
        # 生成交易信号（阈值与训练阶段一致）
        entry_price = self.cached_data['close'].iloc[-1]
        if long_prob > 0.5:  # 使用与训练相同的0.5阈值
            return '做多', 1.0, (
                entry_price * (1 + self.take_profit_pct),
                entry_price * (1 - self.stop_loss_pct),
                4  # 持仓时间（小时）
            )
        elif short_prob > 0.5:
            return '做空', 1.0, (
                entry_price * (1 - self.take_profit_pct),
                entry_price * (1 + self.stop_loss_pct),
                4
            )
        return 'HOLD', 0, (None, None, None)