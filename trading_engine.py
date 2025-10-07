import pandas as pd
import numpy as np
import time
from binance.ws.streams import BinanceSocketManager
from binance import ThreadedWebsocketManager
from binance.client import Client
from threading import Thread
import logging
import os
from datetime import datetime  # 添加这行
import asyncio
from threading import Event  # 新增事件控制
from datetime import timedelta
from report_generate import generate_summary_report
def calculate_shadow_indicators(df):
    """计算影线相关指标"""
    df['body_size'] = (df['close'] - df['open']).abs()
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    return df
def process_kline_data(klines):
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    # 修复：将volume加入数值类型转换
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    return calculate_shadow_indicators(df)
    
class TradingEngine:
    MODE_BACKTEST = 'backtest'
    MODE_LIVE = 'live'
    
    def __init__(self, strategy, data_adapter, mode, capital=10000, trade_fee=0.0002):
        """
        交易引擎初始化
        :param strategy: 交易策略实例
        :param data_adapter: 数据适配器
        :param mode: 运行模式 ('backtest' or 'live')
        :param capital: 初始资金
        :param trade_fee: 交易手续费率
        """
        self.strategy = strategy
        self.data_adapter = data_adapter
        self.mode = mode
        self.capital = capital
        self.trade_fee = trade_fee
        self.position = 0  # 当前持仓数量
        self.firstPosition = 0
        self.positions = []  # 持仓记录
        self.trades = []  # 交易记录
        self.performance = {}  # 性能指标
        self.running = False
        self.entry_price = 0  # 记录入场价格
        self.position_type = None  # 记录仓位类型(做多/做空)
        # 实盘交易相关
        self.binance_client = None
        self.ws_manager = None
        self.backtest_start = None
        
        # 日志配置
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('TradingEngine')
        self.initial_capital = capital  # 新增初始化属性
        self.total_profit = 0.0  # 新增总利润跟踪
        self.entry_time = None  # 新增入场时间跟踪
        self.holding_hours = 100  # 新增持仓时长配置
    
        # 在TradingEngine类中添加新方法
    def run_quick_backtest(self, start_date, end_date):
        """实时回测（从交易所获取历史数据）"""
        # 强制使用交易所数据源
        self.data_adapter.source = 'binance'
        
        # 获取历史数据
        symbol = self.strategy.symbol + "USDT"
        data = self.data_adapter.load_data(
            symbol,
            self.strategy.timeframe,
            start_date,
            end_date,
            btc_symbol='BTC_USDT_USDT'  # 新增BTC数据参数
        )
        # 验证数据日期范围
        data_start = data.index.min()
        data_end = data.index.max()
        print(f"已获取数据，期望截止时间：{end_date}，实际截止时间：{data_end}")
        time.sleep(5)
        # 运行回测逻辑（复用原有回测流程）
        self._run_backtest_core(data, start_date, end_date)
        return self._generate_report()

    def _run_backtest_core(self, data, start_date, end_date):
        """复用回测核心逻辑"""
        print(f"Starting realtime backtest from {start_date} to {end_date}")
        # 清空历史记录
        self.positions = []
        self.trades = []
        init_window = max(self.strategy.seq_length, 72)  # 取序列长度和特征窗口的较大值
        # init_window = 72
        for i in range(init_window, len(data)):

            # 修改为仅使用已闭合K线（排除最新未闭合K线）
            current_data = data.iloc[:i]  # 原为 data.iloc[:i]
                
            # 提前更新时间戳
            current_time = data.index[i]
            # 转换为北京时间 (UTC+8)
            current_time = current_time.tz_localize('UTC').tz_convert('Asia/Shanghai').tz_localize(None)
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            self.strategy.update_data(current_data)
            signal = self.strategy.get_latest_signal()
            trade_time = current_time  # 最后一根K线后1小时  
            # 新增：获取当前处理时段的完整K线数据（第i根K线）
            current_kline = data.iloc[i]  # 新增代码     
            # 新增数据同步机制
            self.data_adapter.raw_data = current_data  # 供交易引擎访问原始数据
            if signal:
                self._execute_trade(signal, trade_time, current_kline)
                # 记录持仓（与常规回测保持一致）
                self.positions.append({
                    'timestamp': trade_time,  # 存储已转换的时间
                    'price': signal['price'],
                    'position': self.position,
                    'capital': self.capital
                })
        
        # 统一生成报告
        self.backtest_start = start_date
        self.backtest_end = end_date
        return self.real_generate_report()

    def run_backtest(self, start, end):
        """运行回测"""
        if self.mode != self.MODE_BACKTEST:
            raise RuntimeError("Engine not in backtest mode")
        self.backtest_start = start
        # 加载历史数据
        data = self.data_adapter.load_data(
            self.strategy.symbol,
            self.strategy.timeframe,
            start,
            end
        )
        
        self.logger.info(f"Starting backtest from {start} to {end} with {len(data)} data points")
        
        # 向量化回测
        for i in range(len(data)):
            # 更新策略数据
            current_data = data.iloc[:i+1]
            current_time = data.index[i]
            self.strategy.update_data(current_data)
            # 获取最新信号
            signal = self.strategy.get_latest_signal()
            if not signal:
                continue
            df = df[:-1] if self.mode == 'live' else df  # 第218行附近
            # 执行交易
            self._execute_trade(signal, current_time)
            # 记录持仓
            self.positions.append({
                'timestamp': signal['timestamp'],
                'price': signal['price'],
                'position': self.position,
                'capital': self.capital
            })
        
        # 生成报告
        # 在运行回测时保存结束时间
        self.backtest_end = end  # 新增代码
        self._generate_report()
        return self.performance
    
    def run_live(self, api_key, api_secret):
        """运行实盘交易（改用REST API轮询）"""
        if self.mode != self.MODE_LIVE:
            raise RuntimeError("Engine not in live mode")
        
        # 初始化期货客户端
        self.binance_client = Client(
            api_key,
            api_secret,
            testnet=False,
            tld='com'
        )
        
        # 创建停止事件
        self.stop_event = Event()
        
        # 启动轮询线程
        self.polling_thread = Thread(target=self._polling_worker)
        self.polling_thread.start()
        
        self.running = True
        self.logger.info("Live trading started (REST API mode)")

    def _polling_worker(self):
        """轮询工作线程（使用winMoney的分页获取逻辑）"""
        while not self.stop_event.is_set():
            try:
                symbol = self.strategy.symbol + "USDT"
                interval = self.strategy.timeframe
                print(f"symbol is ===>{symbol}")
                # 计算时间范围（获取最近3天数据）
                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=3)
                end_ts = int(end_dt.timestamp() * 1000)
                start_ts = int(start_dt.timestamp() * 1000)
                # 检查持仓时间
                if self.position != 0:
                    time_elapsed = datetime.now() - self.entry_time
                    if time_elapsed.total_seconds() >= self.holding_hours * 3600:
                        self._close_position('时间平仓', self.strategy.data['close'].iloc[-1], datetime.now())
                # 分页获取K线数据（与winMoney.py保持一致）
                klines = []
                btc_klines = []
                while True:
                    chunk = self.binance_client.futures_klines(
                        symbol=symbol,
                        interval=interval,
                        startTime=start_ts,
                        endTime=end_ts,
                        limit=500
                    )

                    if not chunk:
                        break
                    klines.extend(chunk)
                
                    # 修改数据获取后的处理逻辑（约218行附近）
                    df = process_kline_data(klines)
                    df = df[:-1] if self.mode == 'live' else df  # 实盘模式排除最新未闭合K线
                    df = df[-200:]  # 保留最近200根K线

          
                    btc_chunk = self.binance_client.futures_klines(
                        symbol='BTCUSDT',
                        interval=interval,
                        startTime=start_ts,
                        endTime=end_ts,
                        limit=500
                    )
                    if not btc_chunk:
                        break
                    btc_klines.extend(btc_chunk)
                    start_ts = chunk[-1][0] + 1  # 更新起始时间为最后一条K线时间+1ms
                    if start_ts >= end_ts:
                        break
                    # 主数据长度: 71 | BTC数据长度: 71
# 合并后缺失值数量: 0
# [NNStrategy] 收到新K线数据，时间: 70
# [INPUT] 统一输入长度: 24根K线
# [INPUT] 时间范围: 2025-07-22 17:00:00 ~ 2025-07-23 16:00:00
# [INPUT] 最新K线时间: 2025-07-23 16:00:00
# [INPUT] 最新K线: 192.73 192.74 187.91 188.28
# [DEBUG] 信号概率（观望/做多/做空）: [1.13328555e-04 1.82985321e-01 8.16901326e-01]
# signal_probs is ===>[1.13328555e-04 1.82985321e-01 8.16901326e-01]
                    # 修改数据获取后的处理逻辑（约218行附近）
                    btc_df = process_kline_data(btc_klines)
                    btc_df = btc_df[:-1] if self.mode == 'live' else btc_df  # 实盘模式排除最新未闭合K线
                    btc_df = btc_df[-200:]  # 保留最近200根K线
                   # 合并数据并验证（关键修改）
                    df = df.join(btc_df[['close']].rename(columns={'close':'btc_close'}), how='left')
                    print("\n[数据合并验证]")
                    print(f"主数据长度: {len(df)} | BTC数据长度: {len(btc_df)}")
                    print(f"合并后缺失值数量: {df['btc_close'].isnull().sum()}")
                    # 新增时间戳验证逻辑
                    valid_timestamp = (
                        not df.empty 
                        and not btc_df.empty 
                        and df.index[0] == btc_df.index[0] 
                        and df.index[-1] == btc_df.index[-1]
                    )
                    print(f"主数据首尾时间: {df.index[0]} - {df.index[-1]}")
                    print(f"BTC数据首尾时间: {btc_df.index[0]} - {btc_df.index[-1]}")
                    if not valid_timestamp:
                        self.logger.error("BTC数据与主数据时间戳不匹配，重新获取数据")
                        klines = []  # 清空已获取数据
                        btc_klines = []
                        continue  # 跳过后续处理，重新获取数据
                    # 更新策略数据（保持原有逻辑）
                    self.strategy.update_data(df)
                
                # 获取信号（后续逻辑保持不变）

                signal = self.strategy.get_latest_signal()
                if signal:
                    self._execute_real_trade(signal)
                print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                # 间隔与策略周期一致（例如1小时策略间隔3600秒）
                self.stop_event.wait(5)

            except Exception as e:
                self.logger.error(f"数据获取异常: {str(e)}")
                time.sleep(10)
    
    def _process_klines(self, klines):
        # 原始处理逻辑
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # 新增精确时间处理
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Shanghai')  # 统一时区
        
        # 新增精度处理（币安返回字符串需要精确转换）
        numeric_cols = ['open', 'high', 'low', 'close']
        df[numeric_cols] = df[numeric_cols].apply(lambda x: x.astype('decimal.Decimal'))
        df[numeric_cols] = df[numeric_cols].astype('float64').round(4)  # 保留4位小数
        
        return df

    def stop(self):
        """停止交易引擎"""
        if self.mode == self.MODE_LIVE and self.running:
            self.stop_event.set()
            self.polling_thread.join()
            self.running = False
            self.logger.info("Live trading stopped")

    def _execute_trade(self, signal_info, current_time, current_kline):
        """在回测中执行多空交易"""
        current_timestamp = self.strategy.data.index[-1]
        price = signal_info['price']
        take_profit = signal_info.get('take_profit', 0)
        stop_loss = signal_info.get('stop_loss', 0)
        signal_type = signal_info.get('signal', 'HOLD')
        # print(f"current_timestamp is ==>{current_timestamp},  signal_info['signal'] is ==>{signal_info['signal']}, price is =>{price}, take_profit is =>{take_profit}, stop_loss is =>{stop_loss}, signal_type is =>{signal_type}")
        # 正确方式：使用最后一条数据
        # 获取策略数据中的最新K线（原代码使用iloc[-1]）
        current_high = current_kline['high']
        current_low = current_kline['low']
        current_price = current_kline['open']

        # 新增价格验证逻辑
        def is_price_valid(signal_price, signal_type):
            """验证信号价格是否在当前K线范围内"""
            print(f"|验证价格是否在当前K线范围内|, entry_price is ==>{signal_price}, current_price is ===>{current_price}, current_low is ===>{current_low}, current_high is ===>{current_high}")
            if signal_type == '做多':
                return current_low <= signal_price <= current_high
            elif signal_type == '做空':
                return current_low <= signal_price <= current_high
            return False
        # 初始化加仓计数器
        if not hasattr(self, 'max_added'):
            self.max_added = 0

        # 计算价格接近止损的程度
        # self.near_stop_loss = False
        # 检查持仓的止盈止损
        # if self.position != 0:
        #     if self.position > 0:  # 多仓
        #         if current_low <= self.stop_loss:
        #             self._close_position('做多止损', self.stop_loss, current_time)
        #             if self.max_added < 2:
        #                 self.near_stop_loss = True
        #         elif current_high >= self.take_profit:
        #             self._close_position('做多止盈', self.take_profit, current_time)
        #     elif self.position < 0:  # 空仓
        #         if current_high >= self.stop_loss:
        #             self._close_position('做空止损', self.stop_loss, current_time)
        #             if self.max_added < 2:
        #                 self.near_stop_loss = True
        #         elif current_low <= self.take_profit:
        #             print(f"current_low is ==>{current_low}, self.take_profit is ==>{self.take_profit}")
        #             self._close_position('做空止盈', self.take_profit, current_time)

        # 在现有止盈止损检查前添加方向冲突平仓逻辑
        # # 处理新信号
        #  # 新增持仓方向检查
        # if self.position != 0:
        #     current_side = 'LONG' if self.position > 0 else 'SHORT'
        #     signal_side = 'LONG' if signal_info['signal'] == '做多' else 'SHORT'
            
        #     if current_side == signal_side:
        #         self.logger.info(f"已有{current_side}仓位，跳过{signal_side}开仓")
        #         return
        # 新增持仓时间检查
        # if self.position != 0 and self.entry_time is not None:
        #     hours_held = (current_time - self.entry_time).total_seconds() / 3600
        #     if hours_held >= self.holding_hours:
        #         if self.position > 0:
        #             self._close_position('做多超时平仓', current_price, current_time)
        #         else:
        #             self._close_position('做空超时平仓', current_price, current_time)
        if signal_info['signal'] == '减仓':
            if self.position > 0:
                close_qty = abs(self.firstPosition) * 1
                self._close_position('做多信号改变减仓', current_price, current_time, close_quantity=close_qty)  # 添加关键字参数
                return 
            elif self.position < 0:
                close_qty = abs(self.firstPosition) * 1
                self._close_position('做空信号改变减仓', current_price, current_time, close_quantity=close_qty)  # 添加关键字参数
                return

        # # 修改开仓逻辑（添加加仓类型）
        # if '加仓' in signal_info['signal']:
        #     self._close_position('风险加仓', current_price, current_time, close_qty)

        if self.position != 0:
            if self.position > 0 and signal_info['signal'] != '做多':
                self._close_position('做多信号改变平仓', current_price, current_time)
            if self.position < 0 and signal_info['signal'] != '做空':
                self._close_position('做空信号改变平仓', current_price, current_time)
    
        self.near_stop_loss  = False
        if signal_info['signal'] == '做多' and (self.position <= 0 or self.near_stop_loss):
            entry_price = signal_info['price'] * (1 - 0)
            print(f"signal_info['price'] is ====>{signal_info['price']}")
            if is_price_valid(entry_price, '做多'):
                self.holding_hours = signal_info.get('holding_time', 100)  # 从信号获取持仓时间
                self.entry_time = current_time  # 记录入场时间
                # 记录止损止盈价格（示例：3%止盈，2%止损）

                self.take_profit = take_profit
                self.stop_loss = stop_loss
                
                # 开多仓逻辑
                available_capital = 10000  # 使用动态资金
                fee = available_capital * 0.0002
                quantity = (available_capital - fee) / entry_price
                if (self.near_stop_loss):
                    self._open_position(
                        trade_type='做多加仓',
                        price=entry_price,
                        quantity=quantity * 2,
                        take_profit=self.take_profit,
                        stop_loss=self.stop_loss,
                        timestamp=current_time
                    )
                else:
                    self._open_position(
                        trade_type='做多',
                        price=entry_price,
                        quantity=quantity,
                        take_profit=self.take_profit,
                        stop_loss=self.stop_loss,
                        timestamp=current_time
                    )
        elif signal_info['signal'] == '做空' and (self.position >= 0 or self.near_stop_loss):
            entry_price = signal_info['price'] * (1 + 0)
            print(f"signal_info['price'] is ====>{signal_info['price']}")
            if is_price_valid(entry_price, '做空'):
                self.take_profit = take_profit
                self.stop_loss = stop_loss
                self.holding_hours = signal_info.get('holding_time', 100)  # 从信号获取持仓时间
                self.entry_time = current_time  # 记录入场时间
                # 开空仓逻辑
                available_capital = self.initial_capital * 1.0
                fee = available_capital * 0.0002
                quantity = (available_capital - fee) / entry_price
                if (self.near_stop_loss):
                    self._open_position(
                        trade_type='做空加仓',
                        price=entry_price,
                        quantity=-quantity * 2,
                        take_profit=self.take_profit,
                        stop_loss=self.stop_loss,
                        timestamp=current_time
                    )
                else:
                    self._open_position(
                    trade_type='做空',
                    price=entry_price,
                    quantity=-quantity,
                    take_profit=self.take_profit,
                    stop_loss=self.stop_loss,
                    timestamp=current_time
                )

    def _open_position(self, trade_type, price, quantity, take_profit, stop_loss, timestamp):
        """统一处理开仓逻辑"""
        print(f"\n=== {trade_type}开仓 ===")
        print(f"时间: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"合约: {self.strategy.symbol}")
        print(f"方向: {trade_type} | 价格: {price:.4f}")
        print(f"数量: {abs(quantity):.4f} | 止盈: {take_profit:.4f} | 止损: {stop_loss:.4f}")
        
        self.trades.append({
            'timestamp': timestamp,
            '类型': trade_type,
            '价格': price,
            '数量': abs(quantity),
            '止盈价': take_profit,
            '止损价': stop_loss,
            '收益': None,
            '手续费': price * abs(quantity) * 0.0002,
            '持仓时间(小时)': self.holding_hours  # 新增持仓时间记录
        })
        self.position = self.position + quantity
        self.firstPosition = self.position
        self.entry_price = price
        self.max_added = self.max_added + 1
    def _close_position(self, close_type, price, timestamp, close_quantity=None):
        """统一处理平仓逻辑"""
        print(f"\n=== 平仓操作 ===")
        print(f"时间: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"类型: {close_type} | 价格: {price:.4f}")
        print(f"入场价: {self.entry_price:.4f} | 盈亏: {price - self.entry_price:.4f}")
         # 计算实际持仓时间
        holding_hours = (timestamp - self.entry_time).total_seconds() / 3600 if self.entry_time else 0
        # 当前使用策略最新价格，应区分平仓类型
        close_price = price
        if '信号改变' in close_type or '减仓' in close_type:
            close_price = self.strategy.data['close'].iloc[-1]  # 使用下根K线开盘价
        close_quantity = abs(self.position) if close_quantity is None else min(abs(self.position), abs(close_quantity))
        fee = close_quantity * close_price * 0.0002
        proceeds = close_quantity * close_price - fee
        
        self.trades.append({
            'timestamp': timestamp,
            '类型': close_type,
            '价格': close_price,
            '数量': close_quantity,
            '收益': (close_price - self.entry_price) * close_quantity - fee if '做多' in close_type 
                else (self.entry_price - close_price) * close_quantity - fee,
            '手续费': fee,
            '持仓时间(小时)': round(holding_hours, 2)  # 新增实际持仓时间
        })
        self.near_stop_loss = False
        self.position = self.position - (close_quantity if self.position > 0 else -close_quantity)
        self.capital += proceeds
        self.total_profit += self.trades[-1]['收益']
        self.max_added = self.max_added - 1

    def _execute_real_trade(self, signal_info):
        """实盘交易执行（完整版）"""
        try:
            from trade_controller import place_order,close_existing_position, has_pending_limit_orders
            ticker = self.binance_client.futures_symbol_ticker(symbol=self.strategy.symbol + "USDT")
            current_price = float(ticker['price'])
            price = current_price
            # 根据方向调整挂单价格
            if signal_info['signal'] == '做多':
                price = signal_info['price'] * 0.999  # 当前价下方0.1%挂单
            elif signal_info['signal'] == '做空':
                price = signal_info['price'] * 1.001  # 当前价上方0.1%挂单
            else:
                price = signal_info['price']

            take_profit = signal_info['take_profit']
            stop_loss = signal_info['stop_loss']
            signal_type = signal_info.get('signal_type', 'manual')  # 新增信号类型字段
            take_profit = signal_info['take_profit']
            # 添加持仓时间记录
            self.entry_time = datetime.now()  # 记录入场时间
            self.holding_hours = signal_info.get('holding_time', 100)  # 默认2小时
            symbol = self.strategy.symbol + "USDT"
            # 从策略中获取止损止盈参数

            # 添加交易开始日志
            self.logger.info(f"开始处理交易信号 | 类型: {signal_info['signal']} | 价格: {price:.4f} | "
                           f"止盈: {take_profit:.4f} | 止损: {stop_loss:.4f}")
            # 获取账户余额
            try:
                positions = self.binance_client.futures_position_information()
                symbol = self.strategy.symbol + "USDT"
                position_amt = next(
                    (float(p['positionAmt']) for p in positions if p['symbol'] == symbol), 
                    0.0
                )
                # 更新本地持仓状态
                self.position = position_amt
                self.logger.info(f"持仓更新 | {symbol} 当前仓位: {position_amt}")
            except Exception as e:
                self.logger.error(f"持仓查询失败: {str(e)}")
                return
            # 检查持仓方向与信号是否冲突
            if self.position != 0:
                current_side = 'NONE'
                if self.position > 0:
                    current_side = 'LONG'
                if self.position < 0:
                    current_side = 'SHORT'
                # 方向冲突时先平仓
                if (current_side == 'LONG' and signal_info['signal'] != '做多') or \
                   (current_side == 'SHORT' and signal_info['signal'] != '做空'):
                    
                    # 平仓逻辑
                    close_success = close_existing_position(
                        symbol=symbol
                    )
                    if close_success:
                        self.position = 0
                        self.logger.info(f"冲突仓位已平仓 | {symbol} {self.position}")
                    else:
                        self.logger.warning("平仓失败，跳过本次交易")
                        return
                else:
                    return
            balance = self.binance_client.futures_account_balance()
            usdt_balance = float([b for b in balance if b['asset'] == 'USDT'][0]['balance'])
            
            # 计算下单数量（使用账户余额的30%）
            quantity = (1400) / price
            
            # 获取交易精度
            exchange_info = self.binance_client.futures_exchange_info()
            symbol_info = next(s for s in exchange_info['symbols'] if s['symbol'] == symbol)
            
            # 修改后的精度获取逻辑
            lot_size_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')
            price_filter = next(f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER')
            step_size = float(lot_size_filter['stepSize'])
            tick_size = float(price_filter['tickSize'])
            
            # 调整数量精度
            quantity = quantity - (quantity % step_size)
            quantity = round(quantity, 8)
            
            # 调整价格精度
            price = round(price - (price % tick_size), 8)
            stop_loss = round(stop_loss - (stop_loss % tick_size), 8)
            take_profit = round(take_profit - (take_profit % tick_size), 8)

            # 确定交易方向
            if signal_info['signal'] == '做多':
                side = 'BUY'
            elif signal_info['signal'] == '做空':
                side = 'SELL'
            else:
                print("HOLD 信号")
                return
            # 修改后的限价单检查逻辑
            if has_pending_limit_orders(symbol):
                now = datetime.now()
                current_hour = now.hour
                current_minute = now.minute
                
                # 每小时只触发一次，且在2分或4分时执行
                if current_minute in [2, 4] and current_hour != getattr(self, 'last_cleanup_hour', -1):
                    self.logger.warning(f"存在未成交限价单，执行强制平仓 | {symbol} (当前时间: {now.strftime('%H:%M')})")
                    close_existing_position(symbol)
                    self.last_cleanup_hour = current_hour  # 记录最后一次清理的小时
                    return
                else:
                    self.logger.warning(f"存在未成交限价单（已在本小时处理过） | {symbol}")
                    return
            success = place_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            if success:
                self.logger.info(f"订单提交成功 | {symbol} {side} {quantity}@{price}")
                # 更新本地持仓状态
                self.position = quantity if side == 'BUY' else -quantity
                self.entry_price = price
                self.trades.append({
                    'timestamp': datetime.now(),
                    '类型': '实盘-' + signal_info['signal'],
                    '价格': price,
                    '数量': quantity,
                    '止盈价': take_profit,
                    '止损价': stop_loss,
                    '信号来源': signal_type,
                    '持仓时间配置': self.holding_hours
                })
            else:
                self.logger.warning("订单提交失败")

        except Exception as e:
            self.logger.error(f"交易执行失败：{str(e)}", exc_info=True)
            # 失败后取消所有未完成订单
    
    def real_generate_report(self):
        """生成统一格式的回测报告"""
        if not self.positions:
            self.performance = {'error': 'No positions recorded'}
            return self.performance
        
        # 生成报告路径（使用回测时间范围）
        report_name = f"{self.backtest_start.strftime('%Y%m%d')}-{self.backtest_end.strftime('%Y%m%d')}_report.xlsx"
        report_path = os.path.join("回测报告", report_name)
        
        # 保存交易记录和持仓记录
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            pd.DataFrame(self.trades).to_excel(writer, sheet_name='交易明细', index=False)
            pd.DataFrame(self.positions).to_excel(writer, sheet_name='持仓记录', index=False)
        
        # 计算绩效指标（与常规回测保持一致）
        self.performance = {
            'start_date': self.backtest_start,
            'end_date': self.backtest_end,
            'final_equity': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'report_path': os.path.abspath(report_path)
        }
        return self.performance
    def _generate_report(self):
        """生成回测报告"""
        if not self.positions:
            self.performance = {'error': 'No positions recorded'}
            return

        # 生成报告路径（使用回测结束时间）
        report_path = "回测报告"
        os.makedirs(report_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol = self.strategy.symbol.replace('/', '')
        trades_file = f"{report_path}/{symbol}_trades_{timestamp}.xlsx"
        print(f"数据已经保存到{symbol}_trades_{timestamp}.xlsx")
        # 仅保存交易明细
        with pd.ExcelWriter(trades_file, engine='openpyxl') as writer:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_excel(writer, sheet_name='交易明细', index=False)

        self.performance['report_file'] = trades_file
        generate_summary_report(trades_file)
        return self.performance