import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data_adapter import DataAdapter
from feature_utils import prepare_features, get_feature_columns
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import datetime
import pandas as pd
import json
import os
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from pandas.api.types import is_numeric_dtype
# 自定义数据集类
class SOLDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 增强版神经网络模型
class EnhancedSOLModel(nn.Module):
    def __init__(self, input_size):
        super(EnhancedSOLModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.2)
        
        # 多任务输出层
        self.signal_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
            nn.LogSoftmax(dim=1)  # 新增激活层
        )
        self.price_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # 止盈和止损价格
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out.permute(1,0,2), lstm_out.permute(1,0,2), lstm_out.permute(1,0,2))
        context = attn_out.permute(1,0,2)[:, -1, :]
        
        signal = self.signal_head(context)
        prices = self.price_head(context)
        return torch.cat([signal, prices], dim=1)

class SOLTrainer:
    def __init__(self, config):
        self.data_adapter = DataAdapter(
            source=config['source'],
            path=config['data_path'],
            mode='backtest'
        )
        self.timeframe = config['timeframe']
        self.batch_size = config.get('batch_size', 64)
        self.seq_length = config.get('seq_length', 24)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _calculate_metrics(self, returns, processed_df):
        """计算风险管理指标（修复版）"""
        # 处理无效值
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 修复累计收益计算
        clipped_returns = np.clip(returns, -0.99, np.inf)  # 限制收益率范围
        cumulative_returns = np.exp(np.log1p(clipped_returns).cumsum()) - 1
        
        # 夏普比率计算增加容错
        if len(returns) < 2 or np.std(returns) == 0:
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # 最大回撤计算优化
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = np.zeros_like(peak)
        valid_mask = (peak != 0)
        drawdown[valid_mask] = (peak[valid_mask] - cumulative_returns[valid_mask]) / peak[valid_mask]
        max_drawdown = np.nanmax(drawdown) if np.any(valid_mask) else 0.0

        # 修复特征有效性计算
        feature_effectiveness = {}
        
        # 收益率相关性计算
        valid_columns = [col for col in processed_df.columns if is_numeric_dtype(processed_df[col])]
        corr_values = processed_df[valid_columns].corrwith(pd.Series(returns, index=processed_df.index[:len(returns)]))
        feature_effectiveness['return_correlation'] = corr_values.to_dict()
        
        # 分组计算改为显式数值处理
        signal_groups = processed_df.groupby('signal')
        for col in valid_columns:
            if col not in ['signal', 'filter_data']:
                group_stats = signal_groups[col].agg(['mean', 'std'])
                # 转换为标量值
                feature_effectiveness[f'{col}_signal_entropy'] = {
                    k: v.item() if hasattr(v, 'item') else v 
                    for k, v in group_stats.to_dict().items()
                }
        
        # 保存为结构化数据
        pd.DataFrame.from_dict(feature_effectiveness, orient='index').to_csv('feature_effectiveness.csv')

        return sharpe_ratio, max_drawdown

    def create_sequences(self, data, labels):
        xs, ys = [], []
        # 扩展时间窗口到72小时（24+48）
        for i in range(len(data)-self.seq_length-72):  
            # 仅当原始数据点包含有效信号时才创建样本
            if labels[i+self.seq_length][0] in [1, 2]:
                # 确保包含完整的时间窗口
                if (i + self.seq_length + 72) < len(data):
                    xs.append(data[i:i+self.seq_length])
                    ys.append([
                        labels[i+self.seq_length][0], 
                        labels[i+self.seq_length][1],
                        labels[i+self.seq_length][2]
                    ])
        return np.array(xs), np.array(ys)

    def train(self, symbol='SOL_USDT_USDT', epochs=50):
        # 加载数据
        raw_data = self.data_adapter.load_data(
            symbol=symbol,
            timeframe=self.timeframe,
            start=datetime.datetime(2020, 1, 1),
            end=datetime.datetime(2025, 3, 1)
        )
        # 使用统一特征工程（先生成原始特征）
        processed_df = prepare_features(raw_data)
        
        # 定义特征列（必须与feature_utils完全一致）
        feature_columns = get_feature_columns()  # 使用统一接口
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(processed_df[feature_columns])
        
        # 标签列
        label_columns = ['signal', 'stop_loss', 'take_profit']
        
        # 标准化处理
        labels = processed_df[label_columns].values
        
        # 创建时间序列样本
        X, y = self.create_sequences(features, labels)
        
        # 数据集分割
        split = int(0.8 * len(X))
        train_data = SOLDataset(X[:split], y[:split])
        val_data = SOLDataset(X[split:], y[split:])
        
        # 创建数据加载器
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)

        # 初始化模型
        model = EnhancedSOLModel(len(feature_columns)).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        
        # 多任务损失函数
        signal_criterion = nn.CrossEntropyLoss()
        price_criterion = nn.HuberLoss()
    
        # === 新增特征重要性分析 ===
        # # 使用随机森林进行初步特征筛选
        # rf = RandomForestClassifier(n_estimators=100)
        # rf.fit(features, processed_df['signal'])
        
        # # 可视化特征重要性
        # importances = rf.feature_importances_
        # indices = np.argsort(importances)[::-1]
        
        # plt.figure(figsize=(12, 8))
        # plt.title("Feature Importances")
        # plt.bar(range(len(indices)), importances[indices], align='center')
        # plt.xticks(range(len(indices)), np.array(feature_columns)[indices], rotation=90)
        # plt.tight_layout()
        # plt.savefig('feature_importance.png')
        # plt.close()
    
        # 训练循环
        current_sharpe = 0.0
        current_dd = 1.0
        max_sharpe = -1000
        min_max_dd = 2000

        for epoch in range(epochs):
            model.train()
            total_loss, total_signal_loss, total_price_loss = 0, 0, 0
            for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                # 分解损失
                signal_loss = signal_criterion(outputs[:, :3], targets[:, 0].long())
                price_loss = price_criterion(outputs[:, 3:], targets[:, 1:])
                loss = signal_loss + price_loss
                
                loss.backward()
                
                # 添加梯度监控
                grad_norms = [p.grad.data.norm(2).item() for p in model.parameters() if p.grad is not None]
                avg_grad_norm = np.mean(grad_norms) if grad_norms else 0
                
                optimizer.step()
                total_loss += loss.item()
                total_signal_loss += signal_loss.item()
                total_price_loss += price_loss.item()

            # 训练集统计
            avg_loss = total_loss / len(train_loader)
            avg_signal_loss = total_signal_loss / len(train_loader)
            avg_price_loss = total_price_loss / len(train_loader)
            
            print(f"\n[训练统计] Epoch {epoch+1}")
            print(f"总损失: {avg_loss:.4f} | 信号损失: {avg_signal_loss:.4f} | 价格损失: {avg_price_loss:.4f}")
            print(f"平均梯度范数: {avg_grad_norm:.4f}")

            # 验证集评估
            model.eval()
            with torch.no_grad():
                val_returns = []
                signal_dist = {0:0, 1:0, 2:0}  # 信号分布统计
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    
                    # 收集预测信号分布
                    signals = torch.argmax(outputs[:, :3], dim=1)
                    for s in signals.cpu().numpy():
                        signal_dist[s] += 1
                    
                    # 收集预测收益率（调整为实际模拟收益）
                    entry_prices = inputs[:, -1, 3]  # 使用收盘价作为入场价
                    # 应改为基于实际信号方向计算
                    signals = torch.argmax(outputs[:, :3], dim=1)
                    long_mask = (signals == 1)
                    short_mask = (signals == 2)

                    # 做多收益计算
                    long_returns = (outputs[long_mask, 3] - entry_prices[long_mask]) / entry_prices[long_mask]
                    # 做空收益计算
                    short_returns = (entry_prices[short_mask] - outputs[short_mask, 4]) / entry_prices[short_mask]

                    val_returns = torch.cat([long_returns, short_returns]).cpu().numpy()
                
                # 计算验证集统计
                val_returns = np.array(val_returns)
                mean_return = np.mean(val_returns)
                std_return = np.std(val_returns)
                
                print("\n[验证统计]")
                print(f"信号分布 - 空仓: {signal_dist[0]} | 做多: {signal_dist[1]} | 做空: {signal_dist[2]}")
                print(f"平均收益率: {mean_return:.4%} | 收益波动率: {std_return:.4%}")

            # 验证集评估
            model.eval()
            with torch.no_grad():
                val_returns = []
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    
                    # 收集预测收益率
                    val_returns.extend(outputs.view(-1).cpu().numpy())
                
            # 计算风险管理指标
            val_returns = np.array(val_returns)
            current_sharpe, current_dd = self._calculate_metrics(val_returns, processed_df)
            print(f"current_sharpe: {current_sharpe}, current_dd: {current_dd}")
            if current_sharpe > max_sharpe:
                max_sharpe = current_sharpe
            if current_dd < min_max_dd:
                min_max_dd = current_dd
            
            print(f'Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | max_sharpe: {max_sharpe:.4f} | min_max_dd: {min_max_dd:.4f}'  )

        # 保存完整模型
        model_dir = r'C:\Users\mazhao\Desktop\MAutoTrader\model'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{symbol}_nn_model.pth")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'training_metrics': {
                'best_sharpe': max_sharpe,
                'best_drawdown': min_max_dd,
                'final_returns': val_returns.tolist()
            },
            'config': {
                'input_size': X.shape[2],
                'seq_length': self.seq_length
            }
        }, model_path)

if __name__ == "__main__":
    # 加载配置文件
    config_path = os.path.join(os.path.dirname(__file__), 'nnconfig.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)['model_train']
    
    # 初始化训练器
    trainer = SOLTrainer({
        'source': 'local',
        'data_path': config['data_path'],
        'timeframe': '1h',
        'batch_size': 128,
        'seq_length': 24
    })
    
    # 遍历所有币种进行训练
    for symbol in config['symbols']:
        print(f"\n{'='*40}")
        print(f"开始训练 {symbol} 模型")
        print(f"{'='*40}")
        trainer.train(
            symbol=symbol,
            epochs=config.get('epochs', 50)
        )