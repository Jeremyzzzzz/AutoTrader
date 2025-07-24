import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_adapter import DataAdapter
from feature_utils import prepare_features, get_feature_columns, calculate_atr, calculate_rsi
from sklearn.preprocessing import StandardScaler
import pandas as pd
import datetime

class AlarmDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class RiskAlertModel(nn.Module):
    def __init__(self, input_size):
        super(RiskAlertModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3),  # 修改输出层为3个类别
            nn.Softmax(dim=1)  # 添加Softmax激活
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(
            lstm_out.permute(1,0,2), 
            lstm_out.permute(1,0,2), 
            lstm_out.permute(1,0,2)
        )
        context = attn_out.mean(dim=0)
        return self.classifier(context)

class SOLAlarmTrainer:
    def __init__(self, config):
        self.data_adapter = DataAdapter(
            source=config['source'],
            path=config['data_path'],
            mode='backtest'
        )
        self.timeframe = config['timeframe']
        self.seq_length = config.get('seq_length', 24)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_labels(self, data):
        """生成三维标签 [无风险, 做多风险, 做空风险]"""
        labels = []
        for i in range(len(data)):
            cached_data = data.iloc[max(0,i-24):i+1]
            if len(cached_data) < 24:
                labels.append([1, 0, 0])  # 默认无风险
                continue

            current_signal = data['signal'].iloc[i]
            atr_4h = calculate_atr(cached_data[-24:], 8).iloc[-1]
            current_range = cached_data['high'].iloc[-1] - cached_data['low'].iloc[-1]
            
            ma72 = cached_data['close'].rolling(72).mean()
            ma72_slope = (ma72.iloc[-1] - ma72.iloc[-6]) / 5
            
            rsi_6h = calculate_rsi(cached_data['close'], 6).iloc[-1]
            
            # 分别检测做多和做空风险
            long_risk = [
                current_range > atr_4h * 1.5,
                abs(ma72_slope) < 0.0005,
                current_signal == 1 and rsi_6h > 70
            ]
            
            short_risk = [
                current_range > atr_4h * 1.5,
                abs(ma72_slope) < 0.0005, 
                current_signal == 2 and rsi_6h < 30
            ]
            
            # 生成三维标签
            if any(long_risk) and current_signal == 1:
                labels.append([0, 1, 0])  # 做多风险
            elif any(short_risk) and current_signal == 2:
                labels.append([0, 0, 1])  # 做空风险
            else:
                labels.append([1, 0, 0])  # 无风险
                
        return np.array(labels)

    def train(self, symbol='SOL_USDT_USDT', epochs=30):
        # 加载并预处理数据
        raw_data = self.data_adapter.load_data(
            symbol=symbol,
            timeframe=self.timeframe,
            start=datetime.datetime(2020, 1, 1),
            end=datetime.datetime(2025, 3, 1)
        )
        
        # 生成特征和标签
        processed_df = prepare_features(raw_data)
        feature_columns = get_feature_columns()
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(processed_df[feature_columns])
        labels = self._create_labels(processed_df)

        # 创建序列数据
        X, y = [], []
        for i in range(len(features)-self.seq_length):
            X.append(features[i:i+self.seq_length])
            y.append(labels[i+self.seq_length-1])
        X, y = np.array(X), np.array(y)

        # 数据集分割
        split = int(0.8 * len(X))
        train_data = AlarmDataset(X[:split], y[:split])
        val_data = AlarmDataset(X[split:], y[split:])

        # 初始化模型
        model = RiskAlertModel(len(feature_columns)).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()  # 修改损失函数

        # 训练循环
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=64)
        
        best_acc = 0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, torch.argmax(targets, dim=1))
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 验证评估
            model.eval()
            val_loss, correct = 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, torch.argmax(targets, dim=1)).item()
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == torch.argmax(targets, dim=1)).sum().item()
            
            val_acc = correct / len(val_data)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'scaler_mean': self.scaler.mean_,
                    'scaler_scale': self.scaler.scale_,
                    'config': {'input_size': len(feature_columns)}
                }, f'model/{symbol}_alarm_model.pth')
            
            print(f'Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}')

if __name__ == "__main__":
    trainer = SOLAlarmTrainer({
        'source': 'local',
        'data_path': 'SOL回测结果训练集',
        'timeframe': '1h',
        'seq_length': 24
    })
    trainer.train(symbol='SOL_USDT_USDT', epochs=100)