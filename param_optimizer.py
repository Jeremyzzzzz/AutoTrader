import itertools
import json
import time
import pandas as pd
import sys
from trading_engine import TradingEngine
from data_adapter import DataAdapter
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategys.upper_shadow_strategy import UpperShadowStrategy

class ParameterOptimizer:
    def __init__(self, config_path, data_source, data_path, symbol, timeframe, capital=10000, weight=(0.4, 0.4, 0.2)):
        """
        :param weight: (收益率权重, 夏普率权重, 回撤权重)
        """
        with open(config_path, 'r') as f:
            self.base_config = json.load(f)
        
        # 参数调优范围配置
        self.param_ranges = {
            'upper_shadow_ratio': {'type': 'float', 'range': [1.5, 3.0], 'step': 0.1},
            'lower_shadow_ratio': {'type': 'float', 'range': [0.1, 0.3], 'step': 0.1},
            'entry_multiplier': {'type': 'int', 'range': [1, 3], 'step': 1},
            'tp_multiplier': {'type': 'int', 'range': [2, 5], 'step': 1},
            'sl_multiplier': {'type': 'int', 'range': [1, 3], 'step': 1}
        }
        
        # 回测配置
        self.data_source = data_source
        self.data_path = data_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.capital = capital
        self.weights = weight
        self.best_score = -float('inf')
        self.best_params = None

        # 添加多线程锁和线程池
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)  # 根据CPU核心数调整
        
        # 初始化日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('strategys/optimization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('Optimizer')
    def generate_param_combinations(self):
        param_grid = {}
        for param, spec in self.param_ranges.items():
            start, end = spec['range']
            step = spec['step']
            
            if spec['type'] == 'float':
                values = [round(start + i*step, 2) for i in range(int((end-start)/step)+1)]
            else:
                values = list(range(start, end+1, step))
                
            param_grid[param] = values
        
        keys = param_grid.keys()
        values = param_grid.values()
        return [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    def evaluate_params(self, returns, sharpe, max_drawdown):
        # 标准化处理（示例评分算法，可根据需求调整）
        return_score = returns * self.weights[0]
        sharpe_score = sharpe * self.weights[1]
        drawdown_score = (1 - max_drawdown) * self.weights[2]
        return return_score + sharpe_score + drawdown_score

    def _calculate_performance(self, trades_df):
        """根据交易记录计算绩效指标（修复版）"""
        if trades_df.empty:
            return 0.0, 0.0, 1.0
        
        # 修复1：计算累计收益（基于初始资本）
        initial_capital = self.capital
        cumulative_profit = trades_df['收益'].cumsum()
        equity_curve = initial_capital + cumulative_profit
        
        # 总收益率（百分比）
        total_return = (equity_curve.iloc[-1] / initial_capital - 1) * 100 if len(equity_curve) > 0 else 0
        
        # 修复2：计算夏普比率（基于日收益率）
        daily_returns = equity_curve.pct_change().dropna()
        if len(daily_returns) == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  # 年化夏普
        
        # 修复3：正确计算最大回撤
        max_drawdown = 0
        peak = equity_curve[0]
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_drawdown:
                max_drawdown = dd
        
        return total_return, sharpe_ratio, max_drawdown

    def optimize(self, start_date, end_date):
        combinations = self.generate_param_combinations()
        
        # 使用线程池提交任务
        futures = []
        for params in combinations:
            future = self.executor.submit(
                self._run_backtest,
                params,
                start_date,
                end_date
            )
            futures.append(future)
        
        # 等待所有任务完成
        for future in futures:
            future.result()
        
        return self.best_params
    def _run_backtest(self, params, start_date, end_date):
        # 原有回测逻辑封装成独立方法
        try:
            strategy_config = {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'params': {**self.base_config, **params}
            }
            strategy = UpperShadowStrategy(strategy_config)
            
            data_adapter = DataAdapter(
                source=self.data_source,
                path=self.data_path,
                mode='backtest'
            )
            
            engine = TradingEngine(
                strategy=strategy,
                data_adapter=data_adapter,
                mode='backtest',
                capital=self.capital
            )
            
            engine.run_backtest(start_date, end_date)
            trades_df = pd.DataFrame(engine.trades)
            
            total_return, sharpe_ratio, max_drawdown = self._calculate_performance(trades_df)
            score = self.evaluate_params(total_return, sharpe_ratio, max_drawdown)
            
            # 使用锁保护共享资源
            with self.lock:
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    
                    log_msg = (
                        f"New best params | Score: {score:.2f}\n"
                        f"Params: {params}\n"
                        f"Return: {total_return:.2f}% | "
                        f"Sharpe: {sharpe_ratio:.2f} | "
                        f"Drawdown: {max_drawdown:.2%}\n"
                        "--------------------------------"
                    )
                    self.logger.info(log_msg)
                    
                    optimized_config = {**self.base_config, **self.best_params}
                    with open('strategys/optimized_config.json', 'w') as f:
                        json.dump(optimized_config, f, indent=2)
                        
        except Exception as e:
            self.logger.error(f"参数组合 {params} 回测失败: {str(e)}")
# 使用示例
if __name__ == "__main__":
    optimizer = ParameterOptimizer(
        config_path='strategys/shadowConfig.json',
        data_source='local',  # 或 'binance'
        data_path='C:/Users/mazhao/Desktop/MAutoTrader/回测数据训练集',
        symbol='SOL_USDT_USDT',
        timeframe='1h',
        capital=10000
    )
    
    # 设置回测时间范围
    optimizer.optimize(
        start_date='2023-01-01',
        end_date='2025-01-01'
    )