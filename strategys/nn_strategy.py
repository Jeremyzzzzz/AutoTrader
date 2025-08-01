import torch
import numpy as np
from strategy import BaseStrategy
from data_adapter import DataAdapter
from sklearn.preprocessing import StandardScaler
from sol_nn_trainer import EnhancedSOLModel
from sol_nn_alarm_trainer import RiskAlertModel
from feature_utils import prepare_features, calculate_rsi, calculate_atr, get_feature_columns, is_filter_data
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
        self.seq_length = 72
        print(f"[NNStrategy] 初始化策略，使用数据路径: {config.get('data_path')}")
        self._load_model()
        # self._load_risk_model() 
        # 策略参数
        self.trade_threshold = 0.015  # 1.5% 阈值
        self.stop_loss_pct = 0.015    # 1.5% 止损
        self.take_profit_pct = 0.02   # 2% 止盈
        
        # 初始化数据适配器
        self.data_adapter = data_adapter
        # 初始化日志路径
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"nn_signals_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx")
        self.lastSignal = None

    def _log_signal(self, signal_probs, kline_data):
        """记录信号日志"""
        # 获取K线时间戳（使用结束时间）
        timestamp = kline_data.name.strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = {
        'timestamp': timestamp,
        'open': kline_data['open'],
        'high': kline_data['high'],
        'low': kline_data['low'],
        'close': kline_data['close'],
        'volume': kline_data['volume'],
        'prob_hold': signal_probs[0],
        'prob_long': signal_probs[1], 
        'prob_short': signal_probs[2]
        }
    
        # 新增写入逻辑
        df = pd.DataFrame([log_entry])
        
        # 如果文件不存在，创建新文件并写入header
        if not os.path.exists(self.log_file):
            df.to_excel(self.log_file, index=False, engine='openpyxl')
        else:
            # 追加模式写入（保留历史记录）
            with pd.ExcelWriter(
                self.log_file,
                engine='openpyxl',
                mode='a',
                if_sheet_exists='overlay'
            ) as writer:
                # 找到最后一行并追加数据
                df.to_excel(
                    writer, 
                    index=False,
                    header=False,
                    startrow=writer.sheets['Sheet1'].max_row
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
        self.seq_length = checkpoint['config']['seq_length']  # 替换原来的硬编码30
        
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
        self.risk_model = RiskAlertModel(len(feature_columns))
        self.risk_model.load_state_dict(checkpoint['model_state_dict'])
        self.risk_model.eval()

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

    def _risk_check(self, signal_type, entry_price, cached_data):
        """风险控制过滤器"""
        # 1. 波动率检查（过去24小时ATR）
        atr_4h = self._calculate_atr(cached_data[-24:], period=8).iloc[-1]
        current_range = cached_data['high'].iloc[-1] - cached_data['low'].iloc[-1]
        
        # 2. 趋势强度检查（72周期EMA斜率）
        ma72 = cached_data['close'].rolling(72).mean()
        ma72_slope = (ma72.iloc[-1] - ma72.iloc[-6]) / 5  # 最近5根K线斜率
        
        # 3. 短期超买超卖检查（6小时RSI）
        rsi_6h = calculate_rsi(cached_data['close'], period=6).iloc[-1]
        
        # 风险规则（可根据需要调整参数）
        risk_conditions = [
            current_range > atr_4h * 1.5,          # 当前波动超过平均波动1.5倍
            abs(ma72_slope) < 0.0005,               # 中期趋势不明朗
            (signal_type == '做多' and rsi_6h > 70) or 
            (signal_type == '做空' and rsi_6h < 30) # 短期超买超卖
        ]
        
        return any(risk_conditions)

    def on_bar(self, data):
        """接收最新行情数据并生成交易信号"""
        print(f"[NNStrategy] 收到新K线数据，时间: {data.index[-1]}")

        try:
            # 确保数据格式统一（新增数据校验）
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError("数据格式异常，缺少必要字段")

            # 统一时间索引处理（修复时间格式问题）
            if not isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index().rename(columns={'datetime': 'timestamp'})
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')

            # 简化数据缓存逻辑（移除外部数据加载）
            if not hasattr(self, 'cached_data') or self.cached_data.empty:
                self.cached_data = data
            else:
                # 仅当收到新数据时追加（修复时间比较逻辑）
                last_cached_time = self.cached_data.index[-1]
                if data.index[-1] > last_cached_time:
                    self.cached_data = pd.concat([self.cached_data, data.iloc[[-1]]])
            
            # 保留最近5倍序列长度的数据（原逻辑保留）
            self.cached_data = self.cached_data[-self.seq_length*100:]
            
            # 数据长度检查（原逻辑保留）
            if len(self.cached_data) < self.seq_length:
                print(f"[NNStrategy] 数据不足，需要{self.seq_length}条，当前{len(self.cached_data)}条")
                return 'HOLD', 0, (None, None)

            # 统一截取指定长度的输入数据
            input_data = self.cached_data[-self.seq_length:]
            print(f"[INPUT] 统一输入长度: {len(input_data)}根K线")  # 新增调试日志

            # 新增时间戳打印
            print(f"[INPUT] 时间范围: {input_data.index[0]} ~ {input_data.index[-1]}")
            print(f"[INPUT] 最新K线时间: {input_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
            # 准备特征数据
            features = self._prepare_features(input_data)
            # print(f"[校验] 输入维度:{len(input_data)} | 缓存总量:{len(self.cached_data)}")
            # print(f"[校验] 最早缓存时间:{self.cached_data.index[0]}") 
            # print(f"[校验] 最新MA72值:{input_data['ma72'].iloc[-1]:.4f}")
            # === 修改为直接使用 is_filter_data 函数 ===
            # 获取最新K线的原始数据
            last_open = self.cached_data['open'].iloc[-1]
            last_high = self.cached_data['high'].iloc[-1]
            last_low = self.cached_data['low'].iloc[-1]
            last_close = self.cached_data['close'].iloc[-1]
            print(f"[INPUT] 最新K线: {last_open:.2f} {last_high:.2f} {last_low:.2f} {last_close:.2f}")
            # 修改数据获取后的处理逻辑（约218行附近）
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
            # 在计算signal_probs之后添加日志记录
            # latest_kline = self.cached_data.iloc[-1]  # 获取最新完整K线
            # self._log_signal(signal_probs, latest_kline)
            # 获取预测信号 (取概率最高的类别)
            signal_idx = 0
            for idx in signal_probs:
                if idx > 0.65:
                    print(f"signal_probs is ===>{signal_probs}")
                    signal_idx = np.argmax(signal_probs)
            print(f"[DEBUG] 预测信号索引: {signal_idx}")
            entry_price = data['close'].iloc[-1]
            
            # #  # 在返回信号前添加风险检查
            # if signal_idx in [1, 2]:
            #     signal_type = '做多' if signal_idx == 1 else '做空'
            #     if self._risk_check(signal_type, entry_price, self.cached_data):
            #         print(f"[RISK] 风险控制触发 {signal_type} 减仓 | 当前价格:{entry_price:.4f}")
            #         if signal_type == '做多':
            #             if self.lastSignal != None and self.lastSignal == '做多':
            #                 return '减仓', 0.5, (  # 仓位系数改为0.25
            #                     entry_price * (1 + take_profit * 10),
            #                     entry_price * (1 - stop_loss * 10),
            #                     100
            #                 )
            #         if signal_type == '做空':
            #             if self.lastSignal != None and self.lastSignal == '做空':
            #                 return '减仓', 0.5, (  # 仓位系数改为0.25
            #                     entry_price * (1 + take_profit * 10),
            #                     entry_price * (1 - stop_loss * 10),
            #                     100
            #                 )
                    # return 'HOLD', 0, (0, 0, 100)
            
            # 生成交易信号（保持原有逻辑）
            if signal_idx == 1:  # 做多
                self.lastSignal = '做多'
                return '做多', 0.5, (
                    entry_price * (1 + take_profit * 10),
                    entry_price * (1 - stop_loss * 10),
                    100  # 添加持仓时间（小时）
                )
            elif signal_idx == 2:  # 做空
                self.lastSignal = '做空'
                return '做空', 0.5, (
                    entry_price * (1 - take_profit * 10),
                    entry_price * (1 + stop_loss * 10),
                    100  # 添加持仓时间（小时）
                )
            else:  # 观望
                return 'HOLD', 0, (0, 0, 100)

        except Exception as e:
            print(f"[NNStrategy] 处理K线时发生异常: {str(e)}")
            return 'HOLD', 0, (None, None, None)