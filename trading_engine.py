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
    return calculate_shadow_indicators(df)
    
class TradingEngine:
    MODE_BACKTEST = 'backtest'
    MODE_LIVE = 'live'
    
    def __init__(self, strategy, data_adapter, mode, capital=10000, trade_fee=0.001):
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
                start_dt = end_dt - timedelta(days=31)
                end_ts = int(end_dt.timestamp() * 1000)
                start_ts = int(start_dt.timestamp() * 1000)
                
                # 分页获取K线数据（与winMoney.py保持一致）
                klines = []
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
                    start_ts = chunk[-1][0] + 1  # 更新起始时间为最后一条K线时间+1ms
                    if start_ts >= end_ts:
                        break
                
                # 处理数据（使用winMoney的处理方式）
                df = process_kline_data(klines)
                df = df[-200:]  # 保留最近200根K线
                
                # 更新策略数据
                self.strategy.update_data(df)
                
                # 获取信号（后续逻辑保持不变）

                signal = self.strategy.get_latest_signal()
                if signal:
                    self._execute_real_trade(signal)
                print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                # 间隔与策略周期一致（例如1小时策略间隔3600秒）
                self.stop_event.wait(60)

            except Exception as e:
                self.logger.error(f"数据获取异常: {str(e)}")
                time.sleep(60)
    
    def _process_klines(self, klines):
        """处理币安K线数据为DataFrame（修复列名和格式）"""
        df = pd.DataFrame(klines, columns=[
            'datetime', 'open', 'high', 'low', 'close', 'volume',  # 修改列名匹配回测数据
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # 统一时间格式为datetime类型
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # 确保数值类型一致
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # 添加与回测数据相同的timestamp列（毫秒时间戳）
        df['timestamp'] = (df.index.astype('int64') // 10**6).astype(int)
        
        return df

    def stop(self):
        """停止交易引擎"""
        if self.mode == self.MODE_LIVE and self.running:
            self.stop_event.set()
            self.polling_thread.join()
            self.running = False
            self.logger.info("Live trading stopped")

    def _execute_trade(self, signal_info, current_time):
        """在回测中执行多空交易"""
        current_timestamp = self.strategy.data.index[-1]
        price = signal_info['price']
        take_profit = signal_info.get('take_profit', 0)
        stop_loss = signal_info.get('stop_loss', 0)
        signal_type = signal_info.get('signal', 'HOLD')
        # print(f"current_timestamp is ==>{current_timestamp},  signal_info['signal'] is ==>{signal_info['signal']}, price is =>{price}, take_profit is =>{take_profit}, stop_loss is =>{stop_loss}, signal_type is =>{signal_type}")
        current_price = self.strategy.data['close'].iloc[-1]

        # 检查持仓的止盈止损
        if self.position != 0:
            if self.position > 0:  # 多仓
                if current_price >= self.take_profit:
                    self._close_position('做多止盈', self.take_profit, current_time)
                elif (signal_info['signal'] == '做空' and self.entry_price < current_price):
                    self._close_position('做多止盈', current_price, current_time)
                elif current_price <= self.stop_loss:
                    self._close_position('做多止损', self.stop_loss, current_time)
                elif (signal_info['signal'] == '做空' and self.entry_price >= current_price):
                    self._close_position('做多止损', current_price, current_time)
            elif self.position < 0:  # 空仓
                if current_price <= self.take_profit:
                    self._close_position('做空止盈', self.take_profit, current_time)
                elif (signal_info['signal'] == '做多' and self.entry_price > current_price):
                    self._close_position('做空止盈', current_price, current_time)
                elif current_price >= self.stop_loss:
                    self._close_position('做空止损', self.stop_loss, current_time)
                elif (signal_info['signal'] == '做多' and self.entry_price <= current_price):
                    self._close_position('做空止损', current_price, current_time)

        # # 处理新信号
        #  # 新增持仓方向检查
        # if self.position != 0:
        #     current_side = 'LONG' if self.position > 0 else 'SHORT'
        #     signal_side = 'LONG' if signal_info['signal'] == '做多' else 'SHORT'
            
        #     if current_side == signal_side:
        #         self.logger.info(f"已有{current_side}仓位，跳过{signal_side}开仓")
        #         return
        if signal_info['signal'] == '做多' and self.position <= 0 and current_price <= price:
            # 记录止损止盈价格（示例：3%止盈，2%止损）
            entry_price = current_price
            self.take_profit = take_profit
            self.stop_loss = stop_loss
            
            # 开多仓逻辑
            available_capital = 10000  # 使用动态资金
            fee = available_capital * 0.0002
            quantity = (available_capital - fee) / entry_price
            
            self._open_position(
                trade_type='做多',
                price=entry_price,
                quantity=quantity,
                take_profit=self.take_profit,
                stop_loss=self.stop_loss,
                timestamp=current_time
            )
        elif signal_info['signal'] == '做空' and self.position >= 0 and current_price >= price:
            # 记录止损止盈价格
            entry_price = price
            self.take_profit = take_profit
            self.stop_loss = stop_loss
            
            # 开空仓逻辑
            available_capital = self.initial_capital * 1.0
            fee = available_capital * 0.0002
            quantity = (available_capital - fee) / entry_price
            
            self._open_position(
                trade_type='做空',
                price=current_price,
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
            '手续费': price * abs(quantity) * 0.0002
        })
        self.position = quantity
        self.entry_price = price

    def _close_position(self, close_type, price, timestamp):
        """统一处理平仓逻辑"""
        print(f"\n=== 平仓操作 ===")
        print(f"时间: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"类型: {close_type} | 价格: {price:.4f}")
        print(f"入场价: {self.entry_price:.4f} | 盈亏: {price - self.entry_price:.4f}")
        
        close_quantity = abs(self.position)
        fee = close_quantity * price * 0.0002
        proceeds = close_quantity * price - fee
        
        self.trades.append({
            'timestamp': timestamp,
            '类型': close_type,
            '价格': price,
            '数量': close_quantity,
            '收益': (price - self.entry_price) * close_quantity - fee if '做多' in close_type 
                else (self.entry_price - price) * close_quantity - fee,
            '手续费': fee
        })
        
        self.position = 0
        self.capital += proceeds
        self.total_profit += self.trades[-1]['收益']

    def _execute_real_trade(self, signal_info):
        """实盘交易执行（完整版）"""
        try:
            from trade_controller import place_order,close_existing_position
            price = signal_info['price']
            take_profit = signal_info['take_profit']
            stop_loss = signal_info['stop_loss']
            signal_type = signal_info.get('signal_type', 'manual')  # 新增信号类型字段
            take_profit = signal_info['take_profit']
            symbol = self.strategy.symbol + "USDT"
            # 从策略中获取止损止盈参数

            # 获取账户余额
            # 检查持仓方向与信号是否冲突
            if self.position != 0:
                current_side = 'LONG' if self.position > 0 else 'SHORT'
                signal_side = 'SELL' if signal_info['signal'] == '做空' else 'BUY'
                
                # 方向冲突时先平仓
                if (current_side == 'LONG' and signal_side == 'SELL') or \
                   (current_side == 'SHORT' and signal_side == 'BUY'):
                    
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
            quantity = (750) / price
            
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
            # 调用trade_controller的place_order
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
            else:
                self.logger.warning("订单提交失败")

        except Exception as e:
            self.logger.error(f"交易执行失败：{str(e)}", exc_info=True)
            # 失败后取消所有未完成订单
            from trade_controller import cancel_all_orders
            cancel_all_orders(symbol)
    
    def _generate_report(self):
        """生成回测报告"""
        if not self.positions:
            self.performance = {'error': 'No positions recorded'}
            return

        # 生成报告路径（使用回测结束时间）
        report_path = "C:/Users/mazhao/Desktop/MAutoTrader/回测报告"
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