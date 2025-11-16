import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data_adapter import DataAdapter
from feature_utils import prepare_features, get_feature_columns, generate_labels, generate_labels_from_csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import datetime
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import copy
from torch.serialization import default_restore_location
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from pandas.api.types import is_numeric_dtype

class SOLDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class EnhancedSOLModel(nn.Module):
    def __init__(self, input_size):
        super(EnhancedSOLModel, self).__init__()
        # 修改LSTM输出维度
        self.lstm = nn.LSTM(input_size, 64,  # 进一步减少隐藏层维度
                          num_layers=2,  # 减少层数
                          batch_first=True,
                          bidirectional=True,
                          dropout=0.3)
        
        # 调整注意力维度
        self.seq_dropout = nn.Dropout2d(0.2)
        self.attention = nn.MultiheadAttention(embed_dim=128,  # 匹配LSTM输出
                                             num_heads=4,
                                             dropout=0.2)
        
       # 在信号头增加更多Dropout
        self.signal_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),  # 从0.4增加到0.5
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),  # 新增第二层Dropout
            nn.Linear(64, 1)
        )

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)  # 改用Xavier初始化
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        # 调整初始化维度
        num_directions = 2 if self.lstm.bidirectional else 1
        h0 = torch.zeros(self.lstm.num_layers * num_directions, x.size(0), 64).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * num_directions, x.size(0), 64).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))  # 保持原有代码不变
        
        # 修正注意力计算
        attn_out, _ = self.attention(
            lstm_out.permute(1,0,2),
            lstm_out.permute(1,0,2),
            lstm_out.permute(1,0,2)
        )
        context = attn_out.mean(dim=0)
        signal = self.signal_head(context)
        return signal

class SOLTrainer:
    def __init__(self, config):
        # 新增随机种子设置
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.data_adapter = DataAdapter(
            source=config['source'],
            path=config['data_path'],
            mode='backtest'
        )
        torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
        self.timeframe = config['timeframe']
        self.batch_size = config.get('batch_size', 256)
        self.seq_length = config.get('seq_length', 24)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_return = -np.inf
        self.best_loss = np.inf
        self.model_version = 1
        self.existing_model = config.get('existing_model')
        # 替换原有时间设置
        self.cv_config = config.get('cv_config', {})
        self.current_fold = 0
        self.total_folds = self.cv_config.get('num_folds', 5)
        self._init_time_windows(config)
        self.future_window = config.get('future_window', 1)  # 从配置读取
        self.scaler = StandardScaler()
        # 新增验证结果跟踪
        self.best_val_returns = []
        self.best_epochs = []
        self.loss_weights = {
            'signal': config.get('signal_weight', 1), #1.0
            'return': config.get('return_weight', 25),  # 默认提高收益项权重， #50
            'smooth': config.get('smooth_weight', 0),    # 提高平滑项权重 #2
            'drawdown': config.get('drawdown_weight', 0)  # 新增回撤惩罚项
        }

         # 新增早停配置
        self.early_stop_patience = config.get('early_stop_patience', 300)  # 默认7个epoch无改进停止
        self.no_improve_epochs = 0
        self.best_val_acc = 0
        self.best_val_return = -np.inf
        self.feature_importance_history = []  # 新增特征重要性记录
        self.feature_blacklist = set()  # 新增无效特征黑名单
        self.conservative_rate = config.get('conservative_rate', 1)
        # 新增绘图配置
        self.plot_validation = config.get('plot_validation', False)  # 默认关闭绘图
        self.initial_balance = 10000  # 初始本金
        self.save_best_checkpoint = config.get('save_best_checkpoint', True)  # 新增配置项
        self.best_checkpoint_path = ""  # 新增最佳模型路径跟踪
        self.train_loader = None
        self.val_loader = None
        # 新增自动训练参数
        self.auto_train_config = {
            'target_acc': 0.55,        # 目标验证集准确率
            'max_drawdown': 0.1,       # 最大允许回撤
            'lr_search_space': [0.001, 0.0005, 0.0001],  # 学习率搜索空间
            'weight_search_space': {    # 权重搜索空间
                'signal': [1, 2, 3],
                'return': [50, 100, 200],
                'smooth': [10, 30, 50]
            },
            'max_retries': 10           # 最大尝试次数
        }
        self.adaptive_config = {
            'max_dropout': 0.7,  # 最大dropout概率
            'weight_decay_range': [0.01, 0.1],  # 权重衰减范围
            'noise_scale': 0.01  # 输入噪声强度
        }


    # 新增自动训练方法
    def auto_train(self, symbol, initial_epochs=1):

        best_metrics = {
            'acc': 0, 
            'max_drawdown': 1.0,
            'weights': self.loss_weights.copy(),
            'model': None,
            'attempt': -1
        }
        attempt = 0
        saved_models = []
        model_dir = os.path.join("model", "auto_models", symbol)
        os.makedirs(model_dir, exist_ok=True)

        while attempt < self.auto_train_config['max_retries']:
            print(f"\n=== 自动训练尝试 #{attempt+1} ===")
            
            # 动态调整参数
            current_lr = self.auto_train_config['lr_search_space'][attempt % len(self.auto_train_config['lr_search_space'])]
            self.loss_weights = {
                'signal': self.auto_train_config['weight_search_space']['signal'][attempt % 3],
                'return': self.auto_train_config['weight_search_space']['return'][attempt % 3],
                'smooth': self.auto_train_config['weight_search_space']['smooth'][attempt % 3],
                'drawdown': 0.5 if best_metrics['max_drawdown'] > 0.1 else 0
            }
            
            # 执行训练
            model = self.train_model(symbol, epochs=initial_epochs, conservative_rate=0.8)
            val_result = self._evaluate_model(model, self.val_loader, is_val=True)
            
            # 更新全局最佳模型
            if (val_result['acc'] > best_metrics['acc'] or 
               (val_result['acc'] == best_metrics['acc'] and 
                val_result['max_drawdown'] < best_metrics['max_drawdown'])):
                
                best_metrics.update({
                    'acc': val_result['acc'],
                    'max_drawdown': val_result['max_drawdown'],
                    'weights': self.loss_weights.copy(),
                    'model': copy.deepcopy(model.state_dict()),  # 保存模型参数
                    'attempt': attempt + 1
                })
                
                # 保存到文件（新增时间戳）
                timestamp = datetime.datetime.now().strftime("%m%d%H%M")
                model_name = (f"{symbol}_best_acc{val_result['acc']:.2f}_"
                             f"dd{val_result['max_drawdown']:.2f}_{timestamp}.pth")
                checkpoint_path = os.path.join(model_dir, model_name)
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'metrics': val_result,
                    'train_params': {
                        'lr': current_lr,
                        'weights': self.loss_weights.copy(),
                        'attempt': attempt+1,
                        'epochs': initial_epochs
                    },
                    'scaler_mean': trainer.scaler.mean_,
                    'scaler_scale': trainer.scaler.scale_
                }, checkpoint_path)
                print(f"\n🏆 更新全局最佳模型: {os.path.basename(checkpoint_path)}")
                saved_models.append(checkpoint_path)

                # 更新最佳指标
                best_metrics.update({
                    'acc': val_result['acc'],
                    'max_drawdown': val_result['max_drawdown']
                })

            # 检查终止条件
            if val_result['acc'] >= self.auto_train_config['target_acc'] and \
               val_result['max_drawdown'] <= self.auto_train_config['max_drawdown']:
                print("✅ 达到训练目标，停止自动训练")
                return model
                
            attempt += 1
            
        # 最终处理
        print(f"\n⚠️ 未达到目标，使用最佳参数组合重新训练...")
        print("\n=== 候选模型列表 ===")
        for path in saved_models:
            print(f"- {os.path.basename(path)}")
        
        # 用最佳参数强化训练
        self.loss_weights = best_metrics['weights']
        final_model = self.train_model(symbol, epochs=int(initial_epochs*1.5), conservative_rate=0.5)
        
        # 保存最终模型
        final_path = os.path.join(model_dir, f"{symbol}_final.pth")
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'scaler_mean': trainer.scaler.mean_,
            'scaler_scale': trainer.scaler.scale_
        }, final_path)

        
        print(f"\n🔥 最终模型已保存至: {final_path}")
        return final_model

    def _init_time_windows(self, config):
        """初始化时间窗口"""
        # 从配置获取初始时间
        self.train_start = pd.to_datetime(
            config.get('train_start') or self.cv_config['initial_train_start']
        )
        self.train_end = pd.to_datetime(
            config.get('train_end') or self.cv_config['initial_train_end']
        )
        self.val_start = pd.to_datetime(
            config.get('val_start') or self.cv_config['initial_val_start']
        )
        self.val_end = pd.to_datetime(
            config.get('val_end') or self.cv_config['initial_val_end']
        )
        self.test_start = pd.to_datetime(
            config.get('test_start') or self.cv_config['initial_test_start']
        )
        self.test_end = pd.to_datetime(
            config.get('test_end') or self.cv_config['initial_test_end']
        )

    def move_time_window(self):
        """移动时间窗口到下一个折叠"""
        if self.current_fold >= self.total_folds - 1:
            return False
            
        # 计算时间增量
        step = pd.DateOffset(months=self.cv_config.get('step_months', 2))
        train_window = pd.DateOffset(months=self.cv_config['train_window_months'])
        val_window = pd.DateOffset(months=self.cv_config['val_window_months'])
        
        # 更新时间窗口
        self.train_start += step
        self.train_end = self.train_start + train_window
        self.val_start = self.train_end
        self.val_end = self.val_start + val_window
        
        self.current_fold += 1
        return True

    def create_sequences(self, data, labels):
        xs, ys = [], []
        total_length = len(data)
        
        # 新增调试信息
        debug_samples = 3  # 打印前3个样本的时间信息
        print("\n=== 序列时间对齐调试 ===")
        
        # 直接使用所有有效样本（已过滤非交易信号）
        for i in range(total_length - self.seq_length - self.future_window + 1):
            label_idx = i + self.seq_length + self.future_window - 2
            if label_idx >= len(labels):
                continue
                
            # 获取时间范围（假设data是DataFrame）
            input_start = i
            input_end = i + self.seq_length
            label_position = label_idx
            

            timestamps = data.index if hasattr(data, 'index') else pd.date_range(start=self.train_start, periods=len(data), freq='H')
            # 新增时间戳计算
            input_start_time = timestamps[i].strftime('%m-%d %H:%M')
            input_end_time = timestamps[i+self.seq_length-1].strftime('%m-%d %H:%M')
            label_time = timestamps[label_idx].strftime('%m-%d %H:%M')

            if i < 3:
                print(f"样本{i} 输入时段: {input_start_time} 至 {input_end_time}")
                print(f"标签时段: {label_time} (future_window={self.future_window}h)")
                print(f"标签值: signal={labels[label_idx][0]}, return={labels[label_idx][3]:.8f}\n")
            
            xs.append(data[i:i+self.seq_length])
            ys.append([
                labels[label_idx][0],
                labels[label_idx][1],
                labels[label_idx][2],
                labels[label_idx][3]
            ])
        
        print(f"\n生成有效序列数量: {len(xs)} (总数据长度: {total_length})")
        return np.array(xs), np.array(ys)

    def prepare_and_save_features(self, symbol):
        def process_rolling_data(raw_data, window_size=24):
            processed = []
            raw_data = raw_data.sort_index()
            for i in range(window_size, len(raw_data)):
                window_start = raw_data.index[i - window_size]
                window_end = raw_data.index[i-1]
                window_data = raw_data.loc[window_start:window_end]
                if len(window_data) == window_size:
                    # 仅生成特征（移除标签生成）
                    features = prepare_features(window_data)
                    processed.append(features)
            return pd.concat(processed)

        print("\n=== 加载原始数据 ===")
        train_raw = self.data_adapter.load_data(
            symbol=symbol,
            timeframe=self.timeframe,
            start=self.train_start,
            end=self.train_end,
            btc_symbol='BTC_USDT_USDT',
            eth_symbol='ETH_USDT_USDT'

        )
        val_raw = self.data_adapter.load_data(
            symbol=symbol,
            timeframe=self.timeframe,
            start=self.val_start,
            end=self.val_end,
            btc_symbol='BTC_USDT_USDT',
            eth_symbol='ETH_USDT_USDT'
        )
        test_raw = self.data_adapter.load_data(
            symbol=symbol,
            timeframe=self.timeframe,
            start=self.test_start,
            end=self.test_end,
            btc_symbol='BTC_USDT_USDT',
            eth_symbol='ETH_USDT_USDT'
        )
        print("\n=== 处理滚动窗口 ===")
        train_df = process_rolling_data(train_raw)
        val_df = process_rolling_data(val_raw)
        test_df = process_rolling_data(test_raw)
        feature_dir = "model/features"
        os.makedirs(feature_dir, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        
        save_columns = ['datetime'] + get_feature_columns()
    
        train_df.reset_index().rename(columns={'index':'datetime'})\
            .to_csv(f"{feature_dir}/{symbol}_train_{timestamp_str}.csv", 
                    index=False,
                    columns=save_columns,  # 使用包含标签的列
                    encoding='utf-8-sig')
        
        val_df.reset_index().rename(columns={'index':'datetime'})\
            .to_csv(f"{feature_dir}/{symbol}_val_{timestamp_str}.csv",
                    index=False,
                    columns=save_columns,  # 使用包含标签的列
                    encoding='utf-8-sig')
        # 保存测试集特征
        test_df.reset_index().rename(columns={'index':'datetime'})\
            .to_csv(f"{feature_dir}/{symbol}_test_{timestamp_str}.csv",
                    index=False,
                    columns=save_columns,
                    encoding='utf-8-sig')
        print(f"\n特征保存完成: {feature_dir}/{symbol}_[train/val/test]_{timestamp_str}.csv")
        # 新增标签生成步骤
        self._generate_labels_for_csv(feature_dir, symbol, timestamp_str)
        return train_df, val_df
    def _generate_labels_for_csv(self, feature_dir, symbol, timestamp_str):
        """为已保存的CSV文件生成标签"""
        for mode in ['train', 'val', 'test']:
            file_path = f"{feature_dir}/{symbol}_{mode}_{timestamp_str}.csv"
            df = pd.read_csv(file_path, parse_dates=['datetime'])
            
            # 使用完整数据生成标签
            labels = generate_labels_from_csv(df, self.future_window)
            
            # 合并标签到原始数据
            df = pd.concat([df, labels], axis=1)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')

    def load_features(self, symbol, mode='train'):
        feature_dir = "model/features"
        file_pattern = f"{symbol}_{mode}_"
        latest_file = max(
            [f for f in os.listdir(feature_dir) if f.startswith(file_pattern)],
            key=lambda x: os.path.getctime(os.path.join(feature_dir, x))
        )
        
        df = pd.read_csv(os.path.join(feature_dir, latest_file), 
                        parse_dates=['datetime'],
                        index_col='datetime')
    
        df = df.dropna(subset=['signal', 'stop_loss', 'take_profit', 'return_pct'])

        #   # 新增调试信息
        # print(f"\n=== 特征列验证 ===")
        # print("CSV文件中的列:", df.columns.tolist())
        # print("期望的特征列:", get_feature_columns())
        # print("缺失的列:", list(set(get_feature_columns()) - set(df.columns)))
        if mode == 'train':
            self.scaler.fit(df[get_feature_columns()])
        
        features = self.scaler.transform(df[get_feature_columns()])  # 使用过滤后的特征列
        labels = df[['signal', 'stop_loss', 'take_profit', 'return_pct']].values
        
        return features, labels, df
    def train_model(self, symbol, epochs=50, existing_model=None, conservative_rate=1.0):
        # 合并加载训练集和验证集
        train_features, train_labels, _ = self.load_features(symbol, 'train')
        val_features, val_labels, _ = self.load_features(symbol, 'val')
        
        # 提前生成所有序列
        X_train, y_train = self.create_sequences(train_features, train_labels)
        X_val, y_val = self.create_sequences(val_features, val_labels)
        
        # 创建持久化的数据加载器
        train_data = SOLDataset(X_train, y_train)
            # 新增硬件加速配置
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        torch.backends.cudnn.benchmark = True  # 启用cudnn自动优化器
        self.num_workers = min(8, os.cpu_count()//1.5)  # 根据CPU核心数设置工作进程数
                # 修改数据加载部分（约第291行）
        self.train_loader = DataLoader(train_data, 
                                     batch_size=self.batch_size, 
                                     shuffle=True,
                                     num_workers=self.num_workers,
                                     pin_memory=True,
                                     persistent_workers=True,
                                     prefetch_factor=2)
        val_data = SOLDataset(X_val, y_val)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, pin_memory=True)
        
        fold_results = {
            'best_epoch': 0,
            'best_val_return': -np.inf,
            'val_returns_per_epoch': []
        }
        # 标签分布分析
        signal_dist = pd.Series(train_labels[:,0]).value_counts()
        print(f"\n=== 标签分布分析 ===")
        print(f"做多样本: {signal_dist.get(1.0, 0)} ({signal_dist.get(1.0, 0)/len(train_labels):.2%})")
        print(f"做空样本: {signal_dist.get(0.0, 0)} ({signal_dist.get(0.0, 0)/len(train_labels):.2%})")

        

        if existing_model:
            model = self.load_model(symbol, existing_model)
            # 新增保守训练逻辑
            print(f"\n启用保守训练模式 (rate={conservative_rate})")
            base_lr = 0.001 * conservative_rate
            head_lr = 0.005 * conservative_rate
            # 增强L2正则化
            weight_decay = 0.01 / conservative_rate  
        else:
            model = EnhancedSOLModel(len(get_feature_columns())).to(self.device)
            base_lr = 0.001 * conservative_rate
            head_lr = 0.005 * conservative_rate
            # 增强L2正则化
            weight_decay = 0.01 / conservative_rate  

        optimizer = optim.AdamW([
            {'params': model.lstm.parameters(), 'lr': base_lr, 'weight_decay': weight_decay},
            {'params': model.attention.parameters(), 'lr': base_lr, 'weight_decay': weight_decay},
            {'params': model.signal_head.parameters(), 'lr': head_lr, 'weight_decay': weight_decay*2}
        ])
        
        # 在参数更新处添加梯度限制
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 * conservative_rate)  # 缩小梯度裁剪阈值
        
        # 新增学习率调度
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.5
        )

        positive_count = signal_dist.get(1.0, 1e-6)
        negative_count = signal_dist.get(0.0, 1e-6)
        pos_weight = torch.tensor([negative_count / positive_count]).to(self.device)
        signal_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # 早停相关变量初始化
        self.no_improve_epochs = 0
        self.best_val_acc = 0
        self.best_val_return = -np.inf

        best_model_state = model.state_dict().copy()
        best_val_return = -np.inf

        best_val_metrics = {
            'acc': 0,
            'return': -np.inf,
            'drawdown': 1.0
        }
        best_max_drawdown = np.inf
        best_drawdown_model_state = None
        print("\n=== 开始训练 ===")
        for epoch in range(epochs):
            model.train()
            # current_dropout = min(
            #     self.adaptive_config['max_dropout'],
            #     0.3 + 0.4 * (epoch / epochs)  # 随训练进度增加dropout
            # )
            # model.lstm.dropout = current_dropout
            # model.attention.dropout = current_dropout
            
            # # 添加输入数据噪声
            # for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader)):
            #     noise = torch.randn_like(inputs) * self.adaptive_config['noise_scale']
            #     inputs = inputs + noise
            total_loss = 0
            batch_returns = []  # 新增：记录每个batch的收益
            for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch+1}')):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                
                # 新增收益率权重计算
                return_pct = targets[:, 3]  # 提取收益率数据
                sample_weights = torch.exp(return_pct * 5)  # 指数放大高收益样本权重
                sample_weights = sample_weights / sample_weights.mean()  # 新增归一化
                # 修改损失计算部分，约第256行
                outputs = model(inputs)
                base_loss = signal_criterion(outputs[:, 0], targets[:, 0].float())
                
                # 然后应用样本权重
                signal_loss = (base_loss * sample_weights).mean()

                # 计算收益损失（使用绝对收益率和信号匹配度）
                pred_signals = (torch.sigmoid(outputs[:, 0]) > 0.5).long()
                correct_direction = (pred_signals == targets[:, 0].long()).float()
                abs_returns = targets[:, 3]
                
                # 新增实际收益计算（修复缺失变量）
                actual_returns = correct_direction * return_pct
                return_loss = -torch.mean(actual_returns * sample_weights)  # 加权收益损失
                
                # 新增三项惩罚项
                smooth_loss = 0
                
                # 使用detach()避免影响主损失梯度
                actual_returns_detached = actual_returns.detach()
                
                # 1. 夏普比率惩罚（鼓励高夏普）
                sharpe_ratio = torch.mean(actual_returns_detached) / (torch.std(actual_returns_detached) + 1e-6)
                smooth_loss += 0.3 * (1 - sharpe_ratio)
                    # 添加实时计算最大回撤和收益波动性      
                # 2. 最大回撤惩罚（实时计算）
                # 2. 最大回撤惩罚（实时计算）
                cumulative = torch.cumprod(1 + actual_returns, dim=0)
                peak = torch.cummax(cumulative, dim=0)[0]
                drawdown = (peak - cumulative) / (peak + 1e-6)
                smooth_loss += 0.5 * torch.max(drawdown)  # 系数可调
                # 3. 收益波动性惩罚（惩罚方差过大）
                return_variance = torch.var(actual_returns)
                smooth_loss += 0.2 * return_variance  # 系数可调
                
                # 组合损失（调整总损失公式）
                # print(f"signal_loss is =>{self.loss_weights['signal'] * signal_loss}, return_loss is =>{self.loss_weights['return'] * return_loss}, smooth_loss is =>{self.loss_weights['smooth'] * smooth_loss}, max_drawdown is =>{self.loss_weights['drawdown'] * max_drawdown}")
                loss = (
                    self.loss_weights['signal'] * signal_loss +
                    self.loss_weights['return'] * return_loss +
                    self.loss_weights['smooth'] * smooth_loss
                )
    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()

                
            # # # 统一评估训练集和验证集
            # eval_result = self._evaluate_model(model, self.train_loader)
            # train_acc = eval_result['acc']
            # train_return = eval_result['avg_return']

            val_result = self._evaluate_model(model, self.val_loader, is_val=True)
            current_val_return = val_result['total_return']
            # 动态调整损失权重（当回撤过大时增强正则化）
            if val_result['max_drawdown'] > 0.2:
                self.loss_weights['smooth'] *= 1.2
                self.loss_weights['drawdown'] = min(self.loss_weights['drawdown'] + 0.1, 0.5)
            # 新增最大回撤判断逻辑
            current_drawdown = val_result['max_drawdown']
            if current_drawdown < best_max_drawdown:
                best_max_drawdown = current_drawdown
                best_drawdown_model_state = model.state_dict().copy()
                print(f"🎯 发现更小回撤模型 (回撤: {best_max_drawdown:.2%})")
            # 当验证收益下降时增强权重衰减
            if val_result['total_return'] < best_val_metrics['return']:
                for param_group in optimizer.param_groups:
                    param_group['weight_decay'] = min(
                        param_group['weight_decay'] * 1.1,
                        self.adaptive_config['weight_decay_range'][1]
                    )
                    
            # 更新最佳指标
            if val_result['total_return'] > best_val_metrics['return']:
                best_val_metrics.update({
                    'acc': val_result['acc'],
                    'return': val_result['total_return'],
                    'drawdown': val_result['max_drawdown']
                })
                    
            #     # 根据准确率调整样本权重
            #     if val_result['acc'] < 0.55:
            #         self.loss_weights['signal'] = min(self.loss_weights['signal'] * 1.1, 5)

            # ... 原有记录验证结果的代码 ...
            val_return = val_result['avg_return']
            val_acc = val_result['acc']
            # 修改模型保存条件判断部分
            if current_val_return > best_val_return:
                best_val_return = current_val_return
                best_model_state = model.state_dict().copy()
                self.no_improve_epochs = 0
                
                # 新增模型保存逻辑
                if self.save_best_checkpoint:
                    checkpoint_path = os.path.join(
                        'C://Users//mazhao//Desktop//MAutoTrader//model',
                        f"{symbol}_epoch{epoch+1}_return{best_val_return:.2f}.pth"
                    )
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'scaler_mean': self.scaler.mean_,
                        'scaler_scale': self.scaler.scale_,
                        'val_return': best_val_return,
                        'epoch': epoch+1
                    }, checkpoint_path)
                    
                    # 保留最佳模型路径
                    self.best_checkpoint_path = checkpoint_path
                    print(f"\n✅ 保存验证集最佳模型: {checkpoint_path}")
            # if current_val_return > best_val_return:
            #     best_val_return = current_val_return
            #     best_model_state = model.state_dict().copy()
            #     self.no_improve_epochs = 0
                
            #     # 当验证集表现提升时，增强正则化
            #     for param_group in optimizer.param_groups:
            #         param_group['weight_decay'] *= 1.2  # 增强L2正则
            # else:
            #     self.no_improve_epochs += 1
            #     # 降低学习率
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] *= 0.95

            # # === 早停机制 ===
            # if self.no_improve_epochs >= self.early_stop_patience:
            #     print(f"早停触发，恢复最佳模型")
            #     model.load_state_dict(best_model_state)
            #     break

            val_return = val_result['avg_return']
            val_acc = val_result['acc']
            self.val_loader = DataLoader(SOLDataset(X_val, y_val), batch_size=self.batch_size)
            
            # 在每epoch末尾记录验证结果
            fold_results['val_returns_per_epoch'].append(val_return)
            if val_return > fold_results['best_val_return']:
                fold_results['best_val_return'] = val_return
                fold_results['best_epoch'] = epoch
            # 同时修改报告输出部分（约第301行）

            # 修改报告输出格式
            print(f"\nEpoch {epoch+1} 综合报告:")
            print(f"[训练集] 损失: {total_loss/len(self.train_loader):.4f}")
            # print(f"[训练集] 最终收益: {eval_result['total_return']:.2%} | 最大回撤: {eval_result['max_drawdown']:.2%}")
            # print(f"[训练集] 损失: {total_loss/len(self.train_loader):.4f} | 准确率: {train_acc:.2%} | 平均收益: {train_return:.2%}")
            # print(f"[训练集] 预测信号 | 做多: {eval_result['pred_long']} | 做空: {eval_result['pred_short']}")

            print(f"\n[验证集] 准确率: {val_acc:.2%} | 平均收益: {val_return:.2%}")
            print(f"[验证集] 最终收益: {val_result['total_return']:.2%} | 最大回撤: {val_result['max_drawdown']:.2%}")
            print(f"[验证集] 预测信号 | 做多: {val_result['pred_long']} | 做空: {val_result['pred_short']}") 
            # 验证集评估后添加早停判断
            # 训练结束后分析特征重要性
            stability = val_result['stability_metrics']
            print(f"\n[验证集] 低效窗口: {stability['low_efficiency_ratio']:.2%} | 波动评分: {stability['volatility_score']:.2f}")
            print(f"[验证集] 最大连续盈利: {stability['max_consecutive_profit']}小时 | 最大连续亏损: {stability['max_consecutive_loss']}小时")
            self._plot_equity_curve(val_result['returns_series'], symbol)


            # # 双重条件判断（准确率或收益任一提升都视为有改进）
            # if (current_acc > self.best_val_acc + 1e-4) or (current_return > self.best_val_return + 1e-4):
            #     self.best_val_acc = max(current_acc, self.best_val_acc)
            #     self.best_val_return = max(current_return, self.best_val_return)
            #     self.no_improve_epochs = 0
            #     # 保存最佳模型
            #     best_model_state = model.state_dict().copy()
            # else:
            #     self.no_improve_epochs += 1
            #     print(f"早停计数器: {self.no_improve_epochs}/{self.early_stop_patience}")

            # # 早停检查
            # if self.no_improve_epochs >= self.early_stop_patience:
            #     print(f"\n早停触发！在epoch {epoch+1} 验证集准确率({current_acc:.2%})和收益({current_return:.2%})连续{self.early_stop_patience}次未提升")
            #     model.load_state_dict(best_model_state)  # 恢复最佳模型
            #     break
                

        # 记录交叉验证结果
        self.best_val_returns.append(fold_results['best_val_return'])
        self.best_epochs.append(fold_results['best_epoch'])
        
        # 输出当前折叠结果
        print(f"\n=== 交叉验证折叠 {self.current_fold+1}/{self.total_folds} ===")
        print(f"时间窗口: Train({self.train_start.date()}~{self.train_end.date()})"
            f" Val({self.val_start.date()}~{self.val_end.date()})")
        print(f"最佳epoch: {fold_results['best_epoch']} 验证收益: {fold_results['best_val_return']:.2%}")
        # 训练结束后恢复最佳回撤模型
        # if best_drawdown_model_state is not None:
        #     model.load_state_dict(best_drawdown_model_state)
        #     print(f"\n🔥 最终使用最小回撤模型 (回撤: {best_max_drawdown:.2%})")
    
        self._analyze_feature_importance()
        return model

    def _evaluate_model(self, model, data_loader, is_val = False):
        """统一评估函数，新增收益率计算和特征重要性分析"""
        model.eval()
        all_preds = []
        all_targets = []
        total_returns = []
        all_max_probs = []
        feature_importance = np.zeros(len(get_feature_columns()))
        total_samples = 0
        all_signals = []  # 新增信号方向存储
        fixed_loader = DataLoader(data_loader.dataset, 
                                batch_size=max(2, self.batch_size),
                                shuffle=False)

        # 修改梯度计算上下文
        with torch.set_grad_enabled(True):
            for inputs, targets in fixed_loader:
                inputs = inputs.to(self.device).requires_grad_(True)
                targets = targets.to(self.device)
                
                # 修改前向传播方式
                outputs = model(inputs)
                # loss = outputs[:, 0].mean()  # 使用更明确的损失计算
                
                # # 梯度计算优化
                # model.zero_grad()
                # loss.backward()
                
                # # 获取梯度并标准化
                # gradients = inputs.grad.abs().cpu().numpy()
                # gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
                
                # # 修改聚合方式（按时间步加权）
                # time_weights = np.linspace(0.5, 1.5, gradients.shape[1])  # 近期的时序更重要
                # weighted_grad = gradients * time_weights[:, np.newaxis]
                # feature_importance += weighted_grad.sum(axis=(0,1))
                # total_samples += len(inputs)

                # 预测和收益计算部分保持无梯度
                # === 修改信号处理逻辑 ===
                with torch.no_grad():
                    # 获取当前预测信号
                    current_preds = (torch.sigmoid(outputs[:, 0]) > 0.5).long()
                    true_signals = targets[:, 0].long()
                    return_pct = targets[:, 3]

                    # 有效性过滤保持原有逻辑
                    valid_mask = (targets[:, 1] != 0.5) & (targets[:, 2] != 0.5) & (true_signals != -1)
                    valid_preds = current_preds[valid_mask]  # 使用当前预测信号
                    valid_true = true_signals[valid_mask]
                    valid_returns = return_pct[valid_mask]
                    all_signals.extend(valid_preds)  # 新增
                    # 保持与model_paintor相同的收益率调整逻辑
                    adjusted_returns = torch.where(
                        valid_preds == valid_true,
                        valid_returns,
                        -valid_returns
                    )
                    
                    # 收集有效数据
                    total_returns.extend(adjusted_returns.cpu().numpy())
                    all_preds.extend(valid_preds.cpu().numpy())
                    all_targets.extend(valid_true.cpu().numpy())
                     # 修改信号概率计算方式
                    prob_long = torch.sigmoid(outputs[:, 0])
                    prob_short = 1 - prob_long
                    max_probs = torch.maximum(prob_long, prob_short)
                    
                    # 只统计有效样本的概率
                    valid_max_probs = max_probs[valid_mask].cpu().numpy()
                    all_max_probs.extend(valid_max_probs)
                # 新增概率分布分析
            if len(all_max_probs) > 0:
                bins = [0.5, 0.55, 0.6, 0.7, 1.0]
                hist, _ = np.histogram(all_max_probs, bins=bins)
                total = len(all_max_probs)
                distribution = {
                    '50%-55%': hist[0]/total,
                    '55%-60%': hist[1]/total,
                    '60%-70%': hist[2]/total,
                    '70%+': hist[3]/total
                }
                print("\n信号置信度分布:")
                for k, v in distribution.items():
                    print(f"{k}: {v:.2%}")
        # === 重新计算关键指标 ===
        returns_series = pd.Series(total_returns)
        if len(returns_series) == 0:
            print("警告：没有有效交易信号")
            return {
                'acc': 0, 'total_return': 0, 'avg_return': 0, 
                'sharpe': 0, 'max_drawdown': 0, 'win_rate': 0,
                'pred_long': 0, 'pred_short': 0, 'returns_series': returns_series
            }
        
        # 计算与策略报告一致的指标
        total_return = returns_series.sum()
        valid_signals = pd.DataFrame({
            'pred': all_preds,
            'true': all_targets
        })
        
        # 准确率计算
        acc = (valid_signals['pred'] == valid_signals['true']).mean()
        
        # 信号统计
        pred_long = valid_signals['pred'].eq(1).sum()
        pred_short = valid_signals['pred'].eq(0).sum()
        
        # 风险指标计算（与model_paintor相同逻辑）
        equity_curve = self.initial_balance + np.cumsum(returns_series * self.initial_balance)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = drawdown.max()
        
        sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(24*365) if returns_series.std() != 0 else 0
        win_rate = (returns_series > 0).mean()
        # 新增稳定性指标计算
        def calculate_stability_metrics(returns):
            # 1. 滚动窗口收益稳定性（168小时窗口）
            rolling_7d = returns.rolling(168, min_periods=1).sum()
            # 统计低效窗口（收益<5%）
            low_efficiency = (rolling_7d < 0.05).sum() / len(returns)
            
            # 2. 收益平滑度（滚动标准差）
            rolling_std = returns.rolling(24).std().dropna()
            volatility_score = 1 / (1 + rolling_std.mean())
            
            # 3. 连续收益/亏损天数统计
            positive_streaks = []
            negative_streaks = []
            current_streak = 0
            current_sign = 0
            
            for r in returns:
                if r > 0:
                    new_sign = 1
                elif r < 0:
                    new_sign = -1 
                else:
                    new_sign = 0
                    
                if new_sign == current_sign:
                    current_streak += 1
                else:
                    if current_sign == 1:
                        positive_streaks.append(current_streak)
                    elif current_sign == -1:
                        negative_streaks.append(current_streak)
                    current_streak = 1
                    current_sign = new_sign
            
            # 计算最大连续天数
            max_positive = max(positive_streaks) if positive_streaks else 0
            max_negative = max(negative_streaks) if negative_streaks else 0
            
            return {
                'low_efficiency_ratio': low_efficiency,
                'volatility_score': volatility_score,
                'max_consecutive_profit': max_positive,
                'max_consecutive_loss': max_negative
            }
        return {
            'acc': acc,
            'total_return': total_return,
            'avg_return': returns_series.mean(),
            'sharpe': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'pred_long': pred_long,
            'pred_short': pred_short,
            'returns_series': returns_series,
            'signals': np.array(all_signals),
            'stability_metrics': calculate_stability_metrics(returns_series)
        }
    
    def _calculate_sharpe(self, returns_series):
        """计算年化夏普比率"""
        daily_mean = returns_series.mean() * 24  # 假设小时数据
        daily_std = returns_series.std() * np.sqrt(24)
        return daily_mean / daily_std if daily_std != 0 else 0

    def _calculate_max_drawdown(self, returns_series):
        """计算最大回撤和回撤持续时间"""
        # 修改为与简单累加一致的计算方式
        equity_curve = self.initial_balance + np.cumsum(returns_series)  # 使用累加收益
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        
        max_dd = drawdown.max()
        dd_duration = (drawdown == max_dd).sum()
        
        return max_dd, dd_duration

    def _calculate_smoothness(self, returns_series):
        """修改为72小时窗口的滚动收益标准差"""
        if len(returns_series) < 168:
            return 0
        # 计算72小时累计收益的波动率
        rolling_returns = returns_series.rolling(168).sum()
        return rolling_returns.std()

    def _calculate_max_drawdown(self, returns_series):
        """修改为72小时窗口内的最大回撤"""
        if len(returns_series) < 168:
            return 0, 0
        
        max_drawdown = 0
        for i in range(len(returns_series)-168):
            window = returns_series[i:i+168]
            cumulative = np.cumprod(1 + window)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative)/peak
            max_drawdown = max(max_drawdown, drawdown.max())
        return max_drawdown, 0  # 简化持续时间计算

    def _calculate_profit_ratio(self, returns_series):
        """新增稳定性指标（连续盈利/亏损比例）"""
        consecutive_pos = 0
        consecutive_neg = 0
        current_streak = 0
        
        for r in returns_series:
            if r > 0:
                current_streak = current_streak + 1 if current_streak >=0 else 1
            else:
                current_streak = current_streak - 1 if current_streak <=0 else -1
                
            if current_streak > consecutive_pos:
                consecutive_pos = current_streak
            elif current_streak < -consecutive_neg:
                consecutive_neg = abs(current_streak)
        
        stability_ratio = consecutive_pos / (consecutive_neg + 1e-6)
        return stability_ratio

    def _calculate_smoothness(self, returns_series):
        """计算收益平滑度（滚动波动率）"""
        rolling_volatility = returns_series.rolling(24).std().dropna().mean()  # 24小时窗口
        return 1 / (1 + rolling_volatility)  # 波动率越低平滑度越高

    def _calculate_profit_ratio(self, returns_series):
        """计算盈亏比"""
        gains = returns_series[returns_series > 0]
        losses = returns_series[returns_series < 0]
        return gains.mean() / abs(losses.mean()) if len(losses) > 0 else np.inf
    
    def _analyze_feature_importance(self):
        """分析历史特征重要性"""
        if not self.feature_importance_history:
            return
        
        # 计算平均重要性
        avg_importance = np.mean(self.feature_importance_history, axis=0)
        feature_names = get_feature_columns()
        
        # 生成特征报告
        report = pd.DataFrame({
            'feature': feature_names,
            'avg_importance': avg_importance,
            'std_importance': np.std(self.feature_importance_history, axis=0)
        }).sort_values('avg_importance', ascending=False)
        
        # 自动识别无效特征（重要性低于均值1个标准差）
        threshold = report['avg_importance'].mean() - report['avg_importance'].std()
        low_importance_features = report[report['avg_importance'] < threshold]['feature'].tolist()
        
        # 更新黑名单
        self.feature_blacklist.update(low_importance_features)
        
        # 保存报告
        report_path = "model/feature_analysis.csv"
        report.to_csv(report_path, index=False)
        print(f"\n特征分析报告已保存至 {report_path}")
        print("建议移除以下低效特征:", low_importance_features)

    def _plot_equity_curve(self, returns, symbol):
        """绘制收益曲线"""
        plt.figure(figsize=(12, 6))
        
        # 修改为累加计算
        cumulative_returns = np.cumsum(np.array(returns))  # 直接累加收益率
        equity_curve = self.initial_balance + cumulative_returns  # 初始本金 + 累计收益
        
        # 绘制曲线
        plt.plot(equity_curve, label='资金曲线', color='#2ca02c')
        plt.fill_between(range(len(equity_curve)), 
                        self.initial_balance, 
                        equity_curve,
                        color='#2ca02c', alpha=0.1)

        
        plt.title(f'{symbol} 验证集收益曲线 (初始本金 {self.initial_balance})')
        plt.xlabel('交易次数')
        plt.ylabel('资金价值 (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        plot_dir = "model/plots"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{symbol}_validation_curve.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        print(f"\n收益曲线已保存至: {plot_path}")
        plt.close()

    def train(self, symbol='SOL_USDT_USDT', epochs=50, is_save=True, save_epoch_range=0, existing_model=None):
        if is_save:
            self.prepare_and_save_features(symbol)
        model = self.train_model(symbol, epochs, existing_model, self.conservative_rate)
        
        model_dir = "model"
        os.makedirs(model_dir, exist_ok=True)
        
        # 最终模型保存
        final_model_path = os.path.join(model_dir, f"{symbol}_nn_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'trained_epochs': epochs  # 记录总训练轮数
        }, final_model_path)

    def load_model(self, symbol, model):
        """加载已有模型参数"""
        model_path = os.path.join("model", f"{symbol}_nn_model.pth")
        if os.path.exists(model_path):
            # 添加安全全局变量声明
            torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
            
            checkpoint = torch.load(
                model_path, 
                map_location=self.device,
                weights_only=False  # 显式关闭安全加载
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\n成功加载已有模型: {model_path}")
        return model

        # 生成标签逻辑

# 修改主函数部分
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), 'nnconfig.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)['model_train']
    
    # 初始化训练器
    trainer = SOLTrainer({
        'source': 'local',
        'data_path': config['data_path'],
        'timeframe': '1h',
        'batch_size': config['batch_size'],
        'seq_length': config['seq_length'],
        'existing_model': config.get('existing_model'),
        'cv_config': config.get('cv_config', {}),
        'conservative_rate': 0.01,
        'plot_validation': True,  # 启用绘图功能
    })
    
    # # 修改后的训练循环
    # for symbol in config['symbols']:
    #     print(f"\n=== 开始自动训练 {symbol} ===")
    #     # 调用自动训练方法
    #     model = trainer.auto_train(
    #         symbol=symbol,
    #         initial_epochs=config.get('initial_epochs', 100)
    #     )
        
    #     # 保存最终模型
    #     model_path = os.path.join("model", f"{symbol}_nn_model.pth")
    #     torch.save({
    #         'model_state_dict': model.state_dict(),
    #         'scaler_mean': trainer.scaler.mean_,
    #         'scaler_scale': trainer.scaler.scale_,
    #         'total_epochs': config.get('initial_epochs', 100)
    #     }, model_path)
    #     print(f"\n✅ 最终模型已保存至: {model_path}")


    # 修改后的训练循环
    for symbol in config['symbols']:
        total_epochs = config.get('epochs', 50)
        model = None
        
        # 检查是否存在已有模型
        model_path = os.path.join("model", f"{symbol}_nn_model.pth")
        if os.path.exists(model_path):
            load_choice = input("检测到已有模型，是否加载？(y/n): ")
            if load_choice.lower() == 'y':
                base_model = EnhancedSOLModel(len(get_feature_columns())).to(trainer.device)
                model = trainer.load_model(symbol, base_model)
        
        # 首次训练或需要重新训练
        if model is None:
            print("\n开始全新训练...")
            model = trainer.train(symbol=symbol, epochs=total_epochs, is_save=False)
        else:
            print(f"\n继续训练 (当前总轮次: {total_epochs})")
            additional_epochs = int(input("请输入追加训练次数 (0退出): "))
            if additional_epochs <= 0:
                break
            model = trainer.train(symbol=symbol, epochs=additional_epochs, 
                                    existing_model=model, is_save=False)
            total_epochs += additional_epochs
        
        # 保存模型（新增保存校验）
        if model:
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_mean': trainer.scaler.mean_,
                'scaler_scale': trainer.scaler.scale_,
                'total_epochs': total_epochs
            }, model_path)
            print(f"模型已更新: {model_path}")