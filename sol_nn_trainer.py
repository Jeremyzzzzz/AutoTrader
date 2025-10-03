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
            nn.Linear(64, 1),
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
        self.future_window = 4 #一小时
        self.scaler = StandardScaler()
        # 新增验证结果跟踪
        self.best_val_returns = []
        self.best_epochs = []
        self.loss_weights = {
            'signal': config.get('signal_weight', 1), #1.0
            'return': config.get('return_weight', 200),  # 默认提高收益项权重， #200
            'smooth': config.get('smooth_weight', 30)    # 提高平滑项权重 #5
        }

         # 新增早停配置
        self.early_stop_patience = config.get('early_stop_patience', 300)  # 默认7个epoch无改进停止
        self.no_improve_epochs = 0
        self.best_val_acc = 0
        self.best_val_return = -np.inf
        self.feature_importance_history = []  # 新增特征重要性记录
        self.feature_blacklist = set()  # 新增无效特征黑名单
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

    def _calculate_metrics(self, returns):
        # 添加收益率过滤和止损机制
        returns = returns[~np.isnan(returns)]  # 过滤无效值
        returns = returns[returns > -0.9]     # 添加单次止损（最大亏损90%）
        
        cumulative = np.cumprod(1 + returns)  # 使用累乘计算资金曲线
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return np.mean(returns)/np.std(returns), np.max(drawdown)

    def create_sequences(self, data, labels):
        xs, ys = [], []
        total_length = len(data)
        
        # 直接使用所有有效样本（已过滤非交易信号）
        for i in range(total_length - self.seq_length - self.future_window):
            label_idx = i + self.seq_length + self.future_window
            if label_idx >= len(labels):
                continue
                
            xs.append(data[i:i+self.seq_length])
            ys.append([
                labels[label_idx][0],  # 二分类标签（0或1）
                labels[label_idx][1],
                labels[label_idx][2],
                labels[label_idx][3]   # 新增收益率
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
            btc_symbol='BTC_USDT_USDT'
        )
        val_raw = self.data_adapter.load_data(
            symbol=symbol,
            timeframe=self.timeframe,
            start=self.val_start,
            end=self.val_end,
            btc_symbol='BTC_USDT_USDT'
        )

        print("\n=== 处理滚动窗口 ===")
        train_df = process_rolling_data(train_raw)
        val_df = process_rolling_data(val_raw)

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
        
        print(f"\n特征保存完成: {feature_dir}/{symbol}_[train/val]_{timestamp_str}.csv")
        # 新增标签生成步骤
        self._generate_labels_for_csv(feature_dir, symbol, timestamp_str)
        return train_df, val_df
    def _generate_labels_for_csv(self, feature_dir, symbol, timestamp_str):
        """为已保存的CSV文件生成标签"""
        for mode in ['train', 'val']:
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
    def train_model(self, symbol, epochs=50):
        # 合并加载训练集和验证集
        train_features, train_labels, _ = self.load_features(symbol, 'train')
        val_features, val_labels, _ = self.load_features(symbol, 'val')
        
        # 提前生成所有序列
        X_train, y_train = self.create_sequences(train_features, train_labels)
        X_val, y_val = self.create_sequences(val_features, val_labels)
        
        # 创建持久化的数据加载器
        train_data = SOLDataset(X_train, y_train)
        train_loader = DataLoader(train_data, 
                                 batch_size=self.batch_size, 
                                 shuffle=True, 
                                 drop_last=True)
        val_data = SOLDataset(X_val, y_val)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)
        
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

        train_data = SOLDataset(X_train, y_train)
            # 新增硬件加速配置
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        torch.backends.cudnn.benchmark = True  # 启用cudnn自动优化器
        self.num_workers = min(8, os.cpu_count()//2)  # 根据CPU核心数设置工作进程数
        
        # 修改数据加载部分（约第291行）
        train_loader = DataLoader(train_data, 
                                batch_size=self.batch_size, 
                                shuffle=True,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=2)

        model = EnhancedSOLModel(len(get_feature_columns())).to(self.device)
        optimizer = optim.AdamW([
            {'params': model.lstm.parameters(), 'lr': 0.001, 'weight_decay': 0.01},  # 新增L2正则
            {'params': model.attention.parameters(), 'lr': 0.001, 'weight_decay': 0.01},
            {'params': model.signal_head.parameters(), 'lr': 0.005, 'weight_decay': 0.02}
        ])
        
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

        print("\n=== 开始训练 ===")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            batch_returns = []  # 新增：记录每个batch的收益
            for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
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
                cumulative = torch.cumprod(1 + actual_returns, dim=0)
                peak = torch.cummax(cumulative, dim=0)[0]
                drawdown = (peak - cumulative) / (peak + 1e-6)
                smooth_loss += 0.5 * torch.max(drawdown)  # 系数可调
                
                # 3. 收益波动性惩罚（惩罚方差过大）
                return_variance = torch.var(actual_returns)
                smooth_loss += 0.2 * return_variance  # 系数可调
                
                # 组合损失（调整总损失公式）
                loss = (
                    self.loss_weights['signal'] * signal_loss +
                    self.loss_weights['return'] * return_loss +
                    self.loss_weights['smooth'] * smooth_loss
                )
    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()

                
            # 统一评估训练集和验证集
            train_acc, train_pred_dist, train_return = self._evaluate_model(model, train_loader)
            
            val_loader = DataLoader(SOLDataset(X_val, y_val), batch_size=self.batch_size)
            val_acc, val_pred_dist, val_return = self._evaluate_model(model, val_loader)
            # 在每epoch末尾记录验证结果
            fold_results['val_returns_per_epoch'].append(val_return)
            if val_return > fold_results['best_val_return']:
                fold_results['best_val_return'] = val_return
                fold_results['best_epoch'] = epoch
            # 同时修改报告输出部分（约第301行）

            print(f"\nEpoch {epoch+1} 综合报告:")
            print(f"[训练集] 损失: {total_loss/len(train_loader):.4f} | 准确率: {train_acc:.2%} | 平均收益: {train_return:.2%}")
            print(f"         预测分布: 做多 {train_pred_dist[1]} | 做空 {train_pred_dist[0]}")
            print(f"[验证集] 准确率: {val_acc:.2%} | 平均收益: {val_return:.2%} | 预测分布: 做多 {val_pred_dist[1]} | 做空 {val_pred_dist[0]}")
            # 验证集评估后添加早停判断
            current_acc = val_acc
            current_return = val_return
            # 训练结束后分析特征重要性
    

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
        self._analyze_feature_importance()
        return model

    def _evaluate_model(self, model, data_loader):
        """统一评估函数，新增收益率计算和特征重要性分析"""
        model.eval()
        all_preds = []
        all_targets = []
        total_returns = []
        feature_importance = np.zeros(len(get_feature_columns()))
        total_samples = 0
        
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
                loss = outputs[:, 0].mean()  # 使用更明确的损失计算
                
                # 梯度计算优化
                model.zero_grad()
                loss.backward()
                
                # 获取梯度并标准化
                gradients = inputs.grad.abs().cpu().numpy()
                gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
                
                # 修改聚合方式（按时间步加权）
                time_weights = np.linspace(0.5, 1.5, gradients.shape[1])  # 近期的时序更重要
                weighted_grad = gradients * time_weights[:, np.newaxis]
                feature_importance += weighted_grad.sum(axis=(0,1))
                total_samples += len(inputs)

                # 预测和收益计算部分保持无梯度
                with torch.no_grad():
                    preds = (torch.sigmoid(outputs[:, 0]) > 0.5).long()
                    true_signals = targets[:, 0].long()
                    correct_direction = (preds == true_signals).float() * 2 - 1
                    actual_returns = correct_direction * targets[:, 3]
                    total_returns.extend(actual_returns.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(true_signals.cpu().numpy())

        # # 取消注释并修改特征重要性计算部分
        # feature_names = get_feature_columns()
        # importance_scores = feature_importance / (total_samples + 1e-8)
        # sorted_indices = np.argsort(importance_scores)[::-1]
        
        # # 存储当前epoch的特征重要性
        # self.feature_importance_history.append(importance_scores)
        
        # print("\n=== 特征重要性排行榜 ===")
        # for i, idx in enumerate(sorted_indices[:20]):
        #     print(f"TOP {i+1}: {feature_names[idx]} ({importance_scores[idx]:.4f})")


        # 计算收益指标
        avg_return = np.mean(total_returns) if len(total_returns) > 0 else 0
        acc = (np.array(all_preds) == np.array(all_targets)).mean()
        pred_dist = np.bincount(all_preds, minlength=2)
        return acc, pred_dist, avg_return
    
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

    def validate_model(self, symbol, model):
        """修改后的验证方法"""
        features, labels, df = self.load_features(symbol, 'val')
        X_val, y_val = self.create_sequences(features, labels)
        val_loader = DataLoader(SOLDataset(X_val, y_val), batch_size=self.batch_size)
        
        # 使用统一评估函数
        val_acc, val_pred_dist, val_return = self._evaluate_model(model, val_loader)
        
        # 保留原有收益计算逻辑
        model.eval()
        returns = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                
                # 计算收益
                signals = (outputs[:, 0] > 0.5).long()
                entry_prices = inputs[:, -1, 3]
                future_prices = inputs[:, -1, 3].roll(shifts=-self.future_window)
                
                valid_returns = (future_prices[signals == 1] - entry_prices[signals == 1]) / entry_prices[signals == 1]
                returns.extend(valid_returns.cpu().numpy())

        print(f"\n验证最终报告 - 准确率: {val_acc:.2%}")
        print(f"预测分布: 做多 {val_pred_dist[1]} | 做空 {val_pred_dist[0]}")
        return returns

    def train(self, symbol='SOL_USDT_USDT', epochs=50, is_save=True):
        if is_save:
            self.prepare_and_save_features(symbol)
        model = self.train_model(symbol, epochs)
        # returns = self.validate_model(symbol, model)
        
        model_dir = "model"
        os.makedirs(model_dir, exist_ok=True)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'scaler_mean': self.scaler.mean_,  # 新增
            'scaler_scale': self.scaler.scale_  # 新增
        }
        
        model_path = os.path.join(model_dir, f"{symbol}_nn_model.pth")
        torch.save(checkpoint, model_path)  # 修改保存内容
        print(f"\n模型已保存至: {model_path}")

        # 生成标签逻辑
    def generate_labels(self, symbol):
        """独立标签生成函数"""
        feature_dir = "model/features"
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        
        # 加载最新特征数据
        train_feature = pd.read_csv(
            f"{feature_dir}/{symbol}_train_{timestamp_str}.csv", 
            parse_dates=['timestamp'],
            index_col='timestamp'
        )
        val_feature = pd.read_csv(
            f"{feature_dir}/{symbol}_val_{timestamp_str}.csv",
            parse_dates=['timestamp'],
            index_col='timestamp'
        )
        
        # 生成并保存标签（使用feature_utils中的generate_labels）
        train_labels = generate_labels(train_feature, self.future_window)
        val_labels = generate_labels(val_feature, self.future_window)
    
        train_labels.to_csv(f"{feature_dir}/{symbol}_train_labels_{timestamp_str}.csv")
        val_labels.to_csv(f"{feature_dir}/{symbol}_val_labels_{timestamp_str}.csv")
        print(f"\n标签保存完成: {feature_dir}/{symbol}_[train/val]_labels_{timestamp_str}.csv")

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
        'cv_config': config.get('cv_config', {})
    })
    
    # 交叉验证循环
    for symbol in config['symbols']:
        while True:
            print(f"\n{'='*40}")
            print(f"开始训练 {symbol} 模型 [Fold {trainer.current_fold+1}/{trainer.total_folds}]")
            print(f"训练窗口: {trainer.train_start.date()} ~ {trainer.train_end.date()}")
            print(f"验证窗口: {trainer.val_start.date()} ~ {trainer.val_end.date()}")
            print(f"{'='*40}")
            
            # 训练并保存模型
            trainer.train(symbol=symbol, epochs=config.get('epochs', 50), is_save=True)
            
            
            # 移动到下一个时间窗口
            if not trainer.move_time_window():
                break
                
        # 输出交叉验证汇总报告
        print("\n=== 交叉验证汇总 ===")
        for i, (ret, epoch) in enumerate(zip(trainer.best_val_returns, trainer.best_epochs)):
            print(f"Fold {i+1}: 最佳epoch={epoch} 验证收益={ret:.2%}")