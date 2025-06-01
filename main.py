import pandas as pd
import argparse
from datetime import datetime, timedelta
from strategy import BaseStrategy
from ma_crossover_strategy import MACrossoverStrategy
from data_adapter import DataAdapter
from trading_engine import TradingEngine
import time
import json
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading System')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args, _ = parser.parse_known_args()
    
    # 加载JSON配置文件（添加编码参数）
    with open(args.config, encoding='utf-8') as f:  # <-- 这里添加encoding参数
        config = json.load(f)
    
    # 合并命令行参数和配置文件参数（命令行参数优先）
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading System')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
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
    
    # 创建策略实例
    strategy = MACrossoverStrategy(strategy_config)
    
    # 创建数据适配器
    data_adapter = DataAdapter(
        source=args.data_source,
        path=args.data_path,
        api_key=args.api_key,
        api_secret=args.api_secret
    )
    
    # 创建交易引擎
    engine = TradingEngine(
        strategy=strategy,
        data_adapter=data_adapter,
        mode=args.mode,
        capital=args.capital
    )
    
    if args.mode == 'backtest':
        # 回测配置
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
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

if __name__ == "__main__":
    main()