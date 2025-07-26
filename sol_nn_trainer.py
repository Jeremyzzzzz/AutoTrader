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
        self.seq_length = config['seq_length']  # 替换原来的24
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
        self.best_return = -np.inf  # 新增最佳收益率跟踪
        self.best_loss = np.inf     # 新增最佳损失跟踪
        self.model_version = 1  # 新增模型版本跟踪
        self.existing_model = config.get('existing_model')  # 新增已有模型路径参数

    def _calculate_metrics(self, returns, processed_df):
        """计算风险管理指标（修复版）"""
        # 处理无效值
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 修复累计收益计算
        clipped_returns = np.clip(returns, -0.99, np.inf)  # 限制收益率范围
        cumulative_returns = np.exp(np.log1p(clipped_returns).cumsum()) - 1
        
        # 增加时间衰减因子
        time_decay = np.linspace(1, 0.5, len(returns))  # 近期数据权重更高
        weighted_returns = returns * time_decay
        
        # 修改夏普比率计算
        if len(returns) > 24:  # 至少24小时数据
            sharpe_ratio = np.mean(weighted_returns) / np.std(weighted_returns) * np.sqrt(252*24)
        
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
        # 修改1：调整时间窗口为4小时（原72改为4+24）
        predict_time = 4
        for i in range(len(data)-self.seq_length- 24 - predict_time):  # 4+24=28
            # 修改2：预测4小时后的信号
            if labels[i+predict_time][0] in [1, 2]:  # 原i+self.seq_length改为i+4
                # 修改3：调整验证窗口
                if (i + predict_time + 24) < len(data):  # 保留24小时验证窗口
                    xs.append(data[i:i+self.seq_length])
                    ys.append([
                        labels[i+predict_time][0],  # 预测4小时后
                        labels[i+predict_time][1],
                        labels[i+predict_time][2]
                    ])
        return np.array(xs), np.array(ys)

    def train(self, symbol='SOL_USDT_USDT', epochs=50):
 
        best_return = -np.inf
        best_loss = np.inf
        raw_data = self.data_adapter.load_data(
            symbol=symbol,
            timeframe=self.timeframe,
            start=datetime.datetime(2020, 1, 1),
            end=datetime.datetime(2025, 3, 1),
            btc_symbol='BTC_USDT_USDT'
        )
        processed_df = prepare_features(raw_data)
        
        # 特征工程
        feature_columns = get_feature_columns()
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(processed_df[feature_columns])
        labels = processed_df[['signal', 'stop_loss', 'take_profit']].values
        
        # 创建序列数据
        X, y = self.create_sequences(features, labels)
        split = int(0.8 * len(X))
        train_data = SOLDataset(X[:split], y[:split])
        val_data = SOLDataset(X[split:], y[split:])
        # 加载数据
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)
         # 初始化模型和优化器
        if self.existing_model:
            print(f"正在加载已有模型进行增量训练: {self.existing_model}")
            # 扩展安全加载白名单
            import numpy
            from torch.serialization import add_safe_globals
            add_safe_globals([
                numpy._core.multiarray.scalar,
                numpy.dtype,  # 新增dtype支持
                numpy.ndarray
            ])
            
            checkpoint = torch.load(
                self.existing_model, 
                weights_only=True,
                mmap=True
            )
            model = EnhancedSOLModel(**checkpoint['config']).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler.mean_ = np.array(checkpoint['scaler_mean'])
            self.scaler.scale_ = np.array(checkpoint['scaler_scale'])
            self.best_return = checkpoint['training_metrics']['best_return']
            self.best_loss = checkpoint['training_metrics']['best_loss']
            self.model_version = checkpoint.get('model_version', 1) + 1
        else:
            model = EnhancedSOLModel(len(feature_columns)).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        signal_criterion = nn.CrossEntropyLoss()
        price_criterion = nn.HuberLoss()

        # 训练循环
        for epoch in range(epochs):
            model.train()
            total_loss, total_signal_loss, total_price_loss = 0, 0, 0
            
            # 训练阶段
            for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                
                outputs = model(inputs)
                signal_loss = signal_criterion(outputs[:, :3], targets[:, 0].long())
                price_loss = price_criterion(outputs[:, 3:], targets[:, 1:])
                loss = signal_loss + price_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_signal_loss += signal_loss.item()
                total_price_loss += price_loss.item()

            # 计算平均损失
            avg_loss = total_loss / len(train_loader)
            avg_signal_loss = total_signal_loss / len(train_loader)
            avg_price_loss = total_price_loss / len(train_loader)
            
            # 验证阶段
            model.eval()
            val_returns = []
            signal_dist = {0:0, 1:0, 2:0}
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                    
                    # 计算收益率
                    signals = torch.argmax(outputs[:, :3], dim=1)
                    entry_prices = inputs[:, -1, 3]
                    
                    long_mask = (signals == 1)
                    short_mask = (signals == 2)
                    
                    long_returns = (outputs[long_mask, 3] - entry_prices[long_mask]) / entry_prices[long_mask]
                    short_returns = (entry_prices[short_mask] - outputs[short_mask, 4]) / entry_prices[short_mask]
                    
                    batch_returns = torch.cat([long_returns, short_returns]).cpu().numpy()
                    val_returns.extend(batch_returns)
                    
                    # 统计信号分布
                    for s in signals.cpu().numpy():
                        signal_dist[s] += 1

            # 计算验证指标
            mean_return = np.mean(val_returns) if len(val_returns) > 0 else 0.0
            current_return = mean_return
            current_loss = avg_loss
            
           # 修改模型保存逻辑（在原有保存代码处修改）
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
            best_return = current_return
            best_loss = current_loss
            
            model_dir = r'model'
            os.makedirs(model_dir, exist_ok=True)
            self.best_model_path = os.path.join(model_dir, f"{symbol}_best_nn_model.pth")
            
            # 在torch.save的字典中添加版本信息
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_mean': self.scaler.mean_.tolist(),
                'scaler_scale': self.scaler.scale_.tolist(),
                'training_metrics': {
                    'best_return': best_return,
                    'best_loss': best_loss,
                    'final_returns': val_returns
                },
                'config': {
                    'input_size': X.shape[2],
                    'seq_length': self.seq_length
                },
                'model_version': self.model_version  # 新增版本号
            }, self.best_model_path)

            # 打印训练日志
            print(f"\n[Epoch {epoch+1}/{epochs}]")
            print(f"训练损失: {avg_loss:.4f} | 信号损失: {avg_signal_loss:.4f} | 价格损失: {avg_price_loss:.4f}")
            print(f"验证平均收益率: {mean_return:.4%}")
            print(f"信号分布 - 空仓: {signal_dist[0]} | 做多: {signal_dist[1]} | 做空: {signal_dist[2]}")

        # 最终保存
        print(f"\n训练完成，最佳模型已保存至: {self.best_model_path}")
        print(f"最佳平均收益率: {best_return:.4%}")
        print(f"对应最低训练损失: {best_loss:.4f}")

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
        'seq_length': 24,
        'existing_model': config.get('existing_model')
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