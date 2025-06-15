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
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        
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

    def _calculate_metrics(self, returns):
        """计算风险管理指标"""
        cumulative_returns = np.cumprod(1 + returns)
        
        # 夏普率（假设无风险利率为0）
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # 最大回撤
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        max_drawdown = np.max(drawdown)
        
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
            start=datetime.datetime(2023, 1, 1),
            end=datetime.datetime.now()
        )
        # 使用统一特征工程（先生成原始特征）
        processed_df = prepare_features(raw_data)
        
        # 使用统一特征工程
        processed_df = processed_df[processed_df['filter_data']]  # 仅保留有效样本
        
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

        # 训练循环
        current_sharpe = 0.0
        current_dd = 1.0
        max_sharpe = -1000
        min_max_dd = 2000
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                # 分解损失
                signal_loss = signal_criterion(outputs[:, :3], targets[:, 0].long())
                price_loss = price_criterion(outputs[:, 3:], targets[:, 1:])
                loss = signal_loss + price_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

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
            current_sharpe, current_dd = self._calculate_metrics(val_returns)
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