import pandas as pd
import numpy as np
import time
from binance.ws.streams import BinanceSocketManager
from binance.client import Client
from threading import Thread
import logging
import os
from datetime import datetime  # 添加这行
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
        """运行实盘交易"""
        if self.mode != self.MODE_LIVE:
            raise RuntimeError("Engine not in live mode")
        
        self.binance_client = Client(api_key, api_secret)
        self.ws_manager = BinanceSocketManager(self.binance_client)
        
        # 加载初始历史数据
        self._load_initial_data()
        
        # 启动WebSocket
        self.running = True
        self.conn_key = self.ws_manager.start_kline_socket(
            symbol=self.strategy.symbol,
            interval=self.strategy.timeframe,
            callback=self._handle_socket_message
        )
        
        # 在单独的线程中启动WebSocket
        self.ws_thread = Thread(target=self.ws_manager.start)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        self.logger.info("Live trading started")
    
    def stop(self):
        """停止交易引擎"""
        if self.mode == self.MODE_LIVE and self.ws_manager:
            self.ws_manager.stop_socket(self.conn_key)
            self.running = False
            self.logger.info("Live trading stopped")
    
    def _load_initial_data(self):
        """加载初始历史数据预热策略"""
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=30)
        
        data = self.data_adapter.load_data(
            self.strategy.symbol,
            self.strategy.timeframe,
            start,
            end
        )
        
        # 用历史数据预热策略
        self.strategy.update_data(data)
        self.logger.info(f"Loaded {len(data)} historical data points for initialization")
    
    def _handle_socket_message(self, msg):
        """处理WebSocket消息"""
        if msg['e'] == 'kline' and msg['k']['x']:  # 确认K线闭合
            kline = msg['k']
            candle = {
                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v'])
            }
            
            # 转换为DataFrame
            new_data = pd.DataFrame([candle]).set_index('timestamp')
            
            # 更新策略
            self.strategy.update_data(new_data)
            
            # 获取信号
            signal = self.strategy.get_latest_signal()
            if signal:
                self.logger.info(f"New signal: {signal['signal']} at {signal['price']}")
                # 执行交易
                self._execute_real_trade(signal['signal'], signal['price'])
    
    def _execute_trade(self, signal_info, current_time):
        """在回测中执行多空交易"""
        current_timestamp = self.strategy.data.index[-1]
        price = signal_info['price']
        take_profit = signal_info['take_profit']
        stop_loss = signal_info['stop_loss']
        signal_type = signal_info.get('signal_type', 'manual')  # 新增信号类型字段
        print(f"if signal_info['signal'] is ==>{signal_info['signal']}")
        current_price = self.strategy.data['close'].iloc[-1]
        # 检查持仓的止盈止损
        if self.position != 0:
            if self.position > 0:  # 多仓
                if current_price >= self.take_profit:
                    self._close_position('做多止盈', current_price, current_time)
                elif current_price <= self.stop_loss:
                    self._close_position('做多止损', current_price, current_time)
            elif self.position < 0:  # 空仓
                if current_price <= self.take_profit:
                    self._close_position('做空止盈', current_price, current_time)
                elif current_price >= self.stop_loss:
                    self._close_position('做空止损', current_price, current_time)

        # 处理新信号
        if signal_info['signal'] == '做多' and self.position == 0 and current_price <= price:
            # 记录止损止盈价格（示例：3%止盈，2%止损）
            entry_price = price
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
        elif signal_info['signal'] == '做空' and self.position == 0 and current_price >= price:
            # 记录止损止盈价格
            entry_price = price
            self.take_profit = entry_price * 0.97
            self.stop_loss = entry_price * 1.02
            
            # 开空仓逻辑
            available_capital = self.initial_capital * 1.0
            fee = available_capital * 0.0002
            quantity = (available_capital - fee) / entry_price
            
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

    def _execute_real_trade(self, signal, price):
        """在实盘中执行交易"""
        try:
            symbol = self.strategy.symbol
            
            if signal == '做多':
                # 获取账户余额
                balance = float(self.binance_client.get_asset_balance(asset='USDT')['free'])
                
                if balance <= 10:  # 最小交易金额
                    self.logger.warning("Insufficient balance for 做多 order")
                    return
                
                # 计算可买入数量（考虑手续费）
                fee = self.trade_fee
                quantity = (balance * (1 - fee)) / price
                
                # 获取交易对精度
                info = self.binance_client.get_symbol_info(symbol)
                step_size = float([f['stepSize'] for f in info['filters'] if f['filterType'] == 'LOT_SIZE'][0])
                quantity = round(quantity - (quantity % step_size), 8)
                
                # 下订单
                order = self.binance_client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_做多,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                # 更新本地状态
                self.position += quantity
                self.capital = balance - (quantity * price)
                
                self.logger.info(f"Executed 做多 order: {quantity} at {price}")
                
            elif signal == '做空' and self.position > 0:
                # 获取交易对精度
                info = self.binance_client.get_symbol_info(symbol)
                step_size = float([f['stepSize'] for f in info['filters'] if f['filterType'] == 'LOT_SIZE'][0])
                quantity = round(self.position - (self.position % step_size), 8)
                
                if quantity <= 0:
                    self.logger.warning("Insufficient position for 做空 order")
                    return
                
                # 下订单
                order = self.binance_client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_做空,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                # 更新账户余额
                balance = float(self.binance_client.get_asset_balance(asset='USDT')['free'])
                
                # 更新本地状态
                self.capital = balance
                self.position = 0
                
                self.logger.info(f"Executed 做空 order: {quantity} at {price}")
                
            # 记录交易
            if order:
                self.trades.append({
                    'timestamp': pd.Timestamp.now(),
                    'order_id': order['orderId'],
                    'symbol': symbol,
                    'side': signal,
                    'price': float(order['fills'][0]['price']),
                    'quantity': float(order['executedQty']),
                    'fee': float(order['fills'][0]['commission'])
                })
                
        except Exception as e:
            self.logger.error(f"Trade execution failed: {str(e)}")
    
    def _generate_report(self):
        """生成回测报告"""
        if not self.positions:
            self.performance = {'error': 'No positions recorded'}
            return

        # 生成报告路径（使用回测结束时间）
        report_path = "C:/Users/mazhao/Desktop/MAutoTrader/回测报告"
        os.makedirs(report_path, exist_ok=True)
        timestamp = self.backtest_end.strftime("%Y%m%d_%H%M%S")
        symbol = self.strategy.symbol.replace('/', '')
        trades_file = f"{report_path}/{symbol}_trades_{timestamp}.xlsx"

        # 仅保存交易明细
        with pd.ExcelWriter(trades_file, engine='openpyxl') as writer:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_excel(writer, sheet_name='交易明细', index=False)

        self.performance['report_file'] = trades_file
        return self.performance