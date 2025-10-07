import pandas as pd
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timedelta
from strategy import BaseStrategy
# from strategys.ma_crossover_strategy import MACrossoverStrategy
# from strategys.nn_strategy import NNStrategy
from strategys.nn_strategy import NNStrategy
from data_adapter import DataAdapter
from trading_engine import TradingEngine
import time
import json
from strategys.upper_shadow_strategy import UpperShadowStrategy

def main():
    import asyncio
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading System')
    config_path = r'config.json'
    args, _ = parser.parse_known_args()
    
    # 加载配置文件
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    # 创建主解析器（移除--config参数）
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading System')
    parser.add_argument('--mode', choices=['backtest', 'live'], help='Run mode')
    parser.add_argument('--symbol', help='Trading symbol')
    parser.add_argument('--timeframe', help='Kline timeframe')
    parser.add_argument('--data_source', choices=['local', 'binance'], help='Data source for backtest')
    parser.add_argument('--data_path', help='Path to local data storage')
    parser.add_argument('--api_key', help='Binance API key (required for live trading)')
    parser.add_argument('--api_secret', help='Binance API secret (required for live trading)')
    parser.add_argument('--capital', type=float, help='Initial capital')
    
    # 使用配置文件中的值作为默认值
    args = parser.parse_args(namespace=argparse.Namespace(**config))
    args = parser.parse_args(namespace=argparse.Namespace(**config))
    
    # 参数验证
    if not args.mode:
        raise ValueError("必须指定运行模式（--mode）")

    # 策略配置
    strategy_config = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'params': {
            'fast_ma': 12,
            'slow_ma': 26
        }
    }
    

    if args.mode == 'live':
        args.data_source = 'binance'  # 实时数据来源
    # 创建数据适配器
    data_adapter = DataAdapter(
        source=args.data_source,
        path=args.data_path,
        api_key=args.api_key,
        api_secret=args.api_secret,
        mode=args.mode
    )
        # 创建策略实例
    strategy = NNStrategy({
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'data_source': args.data_source,  # 新增data_source参数
        'data_path': args.data_path,
        'mode': args.mode  # 明确传递运行模式
    }, data_adapter)
    # strategy = UpperShadowStrategy(
    #     config={
    #         'symbol': args.symbol,
    #         'timeframe': args.timeframe,
    #         'params': {  # 添加上影线策略参数
    #             'upper_shadow_ratio': 3.0,
    #             'lower_shadow_ratio': 0.3,
    #             'entry_multiplier': 0.5,
    #             'tp_multiplier': 0.5,
    #             'sl_multiplier': 1.5,
    #             'holding_hours': 100
    #         }
    #     }, 
    #     data_adapter=data_adapter
    # )
    # 创建交易引擎
    engine = TradingEngine(
        strategy=strategy,
        data_adapter=data_adapter,
        mode=args.mode,
        capital=args.capital
    )
    
    if args.mode == 'backtest':
        # 回测配置
        end_date = '2024-01-01'
        start_date = '2025-08-01'
        
        print(f"Starting backtest for {args.symbol} from {start_date} to {end_date}")
        
        # 运行回测
        report = engine.run_backtest(start_date, end_date)
        
    elif args.mode == 'live':
        if not args.api_key or not args.api_secret:
            raise ValueError("API key and secret are required for live trading")
        
        print("Starting live trading...")
        engine.run_live(args.api_key, args.api_secret)
        
        try:
            # 保持程序运行
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping trading...")
            engine.stop()      
    elif args.mode == 'quick_backtest':  # 新增模式7
        start_date = datetime(2025, 9, 20)  # 可改为参数化
        end_date = datetime(2025, 10, 6)
        print(f"Starting realtime backtest for {args.symbol} from {start_date} to {end_date}")
        report = engine.run_quick_backtest(start_date, end_date)

if __name__ == "__main__":
    main()

#[DEBUG] 做多信号概率: [0.47886804]， 做空信号概率: [0.521132]