import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def analyze_signal_log(file_path):
    # 读取日志数据
    df = pd.read_excel(file_path)
    
    # 转换时间格式
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # 计算价格变化
    df['price_change'] = df['close'].pct_change() * 100  # 价格百分比变化
    df['price_trend'] = df['close'].rolling(5).mean()     # 5周期均线
    
     # 新增未来时间窗口分析
    periods = {
        '1h': 1,
        '2h': 2,
        '4h': 4,
        '8h': 8
    }
    
     # 修改概率区间定义（统一使用0.995-0.995）
    bins = [round(i*0.005,3) for i in range(190, 200)]  # 0.9950-0.995
    
    # 修改分组逻辑
    df['long_bin'] = pd.cut(df['prob_long'], bins=bins, labels=[f"{x:.3f}-{x+0.005:.3f}" for x in bins[:-1]], right=False)
    df['short_bin'] = pd.cut(df['prob_short'], bins=bins, labels=[f"{x:.3f}-{x+0.005:.3f}" for x in bins[:-1]], right=False)

    # 修改收益率计算（做空方向）
    for name, period in periods.items():
        # 做多收益率保持原样
        df[f'long_{name}'] = (df['close'].shift(-period) - df['close']) / df['close'] * 100
        # 做空收益率调整为：(当前价 - 未来价)/当前价 （正确方向）
        df[f'short_{name}'] = (df['close'] - df['close'].shift(-period)) / df['close'] * 100

    # 生成统计报表（添加做空分析）
    report = []
    for time_window in periods.keys():
        # 做多分析
        long_col = f'long_{time_window}'
        long_group = df.groupby('long_bin')[long_col].agg([
            ('count', 'count'),
            ('mean_return', 'mean'),
            ('median_return', 'median'),
            ('std_dev', 'std'),
            ('win_rate', lambda x: (x > 0).mean() * 100)
        ]).reset_index()
        long_group['signal_type'] = 'LONG'
        long_group['time_window'] = time_window
        
        # 做空分析 
        short_col = f'short_{time_window}'
        short_group = df.groupby('short_bin')[short_col].agg([
            ('count', 'count'),
            ('mean_return', 'mean'),
            ('median_return', 'median'),
            ('std_dev', 'std'),
            ('win_rate', lambda x: (x > 0).mean() * 100)
        ]).reset_index()
        short_group['signal_type'] = 'SHORT'
        short_group['time_window'] = time_window
        
        report.extend([long_group, short_group])
    
    # 合并报表并保存
    final_report = pd.concat(report)
    output_path = file_path.replace('.xlsx', '_analysis.xlsx')
    final_report.to_excel(output_path, index=False)
    print(f"分析报告已保存至：{output_path}")

    # 修改可视化部分
    plt.figure(figsize=(24, 12))
    plt.rcParams['font.size'] = 14

    # 标记做多信号区域（概率>0.995）
    long_mask = df['prob_long'] > 0.995
    plt.scatter(df[long_mask].index, 
                df[long_mask]['close'] * 1.005,  # 价格上方5%位置显示标记
                marker='^', color='green', 
                label='Long Signal (>0.995)')
    
    # 标记做空信号区域（概率>0.995）
    short_mask = df['prob_short'] > 0.995
    plt.scatter(df[short_mask].index,
                df[short_mask]['close'] * 0.995,  # 价格下方5%位置显示标记
                marker='v', color='red',
                label='Short Signal (>0.995)')

    # 新增信号过滤函数
    def filter_consecutive_signals(mask_series):
        filtered = []
        in_sequence = False
        start_idx = None
        
        for i, val in enumerate(mask_series):
            if val:
                if not in_sequence:
                    start_idx = i
                    in_sequence = True
            else:
                if in_sequence:
                    # 添加首尾索引（至少间隔2个以上才记录尾部）
                    filtered.append(start_idx)
                    if i - start_idx > 2:  # 连续3个以上信号才记录尾部
                        filtered.append(i-1)
                    in_sequence = False
                    
        # 处理最后一个未结束的序列
        if in_sequence:
            filtered.append(start_idx)
            if len(mask_series) - start_idx > 2:
                filtered.append(len(mask_series)-1)
        
        # 转换为时间索引并去重
        return mask_series.index[sorted(list(set(filtered)))]

    # 应用过滤
    filtered_long = filter_consecutive_signals(long_mask)
    filtered_short = filter_consecutive_signals(short_mask)

    # 绘制价格走势
    ax = plt.gca()
    ax.plot(df.index, df['close'], label='价格', color='black', alpha=0.7, linewidth=1)

    # 修改后的标记部分
    ax.scatter(filtered_long, 
            df.loc[filtered_long, 'close'] * 1.005,
            marker='^', color='green', s=60,  # 增大标记尺寸
            edgecolors='black', zorder=3,  # 添加黑色边框
            label='做多信号(>0.95)')

    ax.scatter(filtered_short,
            df.loc[filtered_short, 'close'] * 0.995,
            marker='v', color='red', s=60,
            edgecolors='black', zorder=3,
            label='做空信号(>0.95)')

    # 添加网格和细节优化
    ax.grid(which='major', linestyle='--', linewidth=0.5)
    ax.grid(which='minor', linestyle=':', linewidth=0.2)
    plt.xticks(rotation=45)  # 旋转45度避免重叠
    plt.gcf().autofmt_xdate()  # 自动调整布局

    # 保存高清图像（300dpi以上）
    plot_path = file_path.replace('.xlsx', '_probability_plot.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)  # 提高分辨率
    plt.close()
    
    print(f"概率分析图已保存至：{plot_path}")


    # 新增模型效果评估
    def evaluate_model(df):
        # 计算信号有效性
        long_valid = df[df['prob_long'] > 0.95]['long_1h'].mean()
        short_valid = df[df['prob_short'] > 0.95]['short_1h'].mean()
        
        # 计算概率校准误差
        prob_bins = np.arange(0.7, 1.0, 0.05)
        calibration = []
        for b in prob_bins:
            long_winrate = df[df['prob_long'] > b]['long_1h'].apply(lambda x: x > 0).mean()
            short_winrate = df[df['prob_short'] > b]['short_1h'].apply(lambda x: x > 0).mean()
            calibration.append({
                'probability_threshold': b,
                'long_actual_winrate': long_winrate,
                'short_actual_winrate': short_winrate
            })
        
        # 生成评估报告
        report = {
            'long_signal_effectiveness': f"{long_valid:.2%}",
            'short_signal_effectiveness': f"{short_valid:.2%}",
            'calibration_analysis': pd.DataFrame(calibration),
            'return_distribution': df[['long_1h', 'short_1h']].describe()
        }
        return report

    # 执行评估
    model_report = evaluate_model(df)
    
    # 保存评估结果
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
        model_report['calibration_analysis'].to_excel(
            writer, sheet_name='概率校准分析', index=False)
        model_report['return_distribution'].to_excel(
            writer, sheet_name='收益分布统计', index=True)
        
    # 可视化校准曲线
    plt.figure(figsize=(12, 6))
    # 添加这三条plot线
    plt.plot(model_report['calibration_analysis']['probability_threshold'], 
             model_report['calibration_analysis']['long_actual_winrate'],
             label='做多信号校准曲线', marker='o')
    plt.plot(model_report['calibration_analysis']['probability_threshold'],
             model_report['calibration_analysis']['short_actual_winrate'],
             label='做空信号校准曲线', marker='s')
    plt.plot([0.7, 1.0], [0.7, 1.0], '--', color='gray', label='理想校准线')
    
    # 以下设置保留
    plt.title('模型概率校准曲线', fontsize=14)
    plt.xlabel('模型预测概率', fontsize=12)
    plt.ylabel('实际胜率', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    calibration_path = file_path.replace('.xlsx', '_calibration.png')
    plt.savefig(calibration_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"模型校准曲线已保存至：{calibration_path}")

    def generate_conclusion(report):
        # 获取关键指标
        long_effectiveness = float(report['long_signal_effectiveness'].strip('%')) / 100
        short_effectiveness = float(report['short_signal_effectiveness'].strip('%')) / 100
        
        # 校准曲线分析（允许浮点数误差）
        calibration = report['calibration_analysis']
        calib_mask = np.isclose(calibration['probability_threshold'], 0.975, atol=0.025)
        
        if sum(calib_mask) == 0:
            print("警告：未找到0.95概率阈值对应的校准数据，使用最近似值")
            calib_095 = calibration.iloc[-1]  # 取最后一个（最大概率值）
        else:
            calib_095 = calibration[calib_mask].iloc[0]
        
        # 收益风险分析
        returns = report['return_distribution'].loc[['mean', 'std'], :]
        
        # 评估标准
        pass_threshold = all([
            long_effectiveness > 0.005,
            short_effectiveness > 0.005,
            calib_095['long_actual_winrate'] > 0.6,
            calib_095['short_actual_winrate'] > 0.6,
            returns.loc['mean', 'long_1h'] > returns.loc['std', 'long_1h'] * 0.5,
            returns.loc['mean', 'short_1h'] > returns.loc['std', 'short_1h'] * 0.5
        ])
        
        conclusion = {
            '模型可用性': '通过' if pass_threshold else '未通过',
            '主要优势': [],
            '风险提示': [],
            '改进建议': []
        }
        
        # 优势分析
        if long_effectiveness > 0.01:
            conclusion['主要优势'].append(f"做多策略收益显著（{long_effectiveness:.2%}）")
        if short_effectiveness > 0.01:
            conclusion['主要优势'].append(f"做空策略收益显著（{short_effectiveness:.2%}）")
        if calib_095['long_actual_winrate'] > 0.7:
            conclusion['主要优势'].append(f"做多信号预测准确（胜率{calib_095['long_actual_winrate']:.1%}）")
        if calib_095['short_actual_winrate'] > 0.7:
            conclusion['主要优势'].append(f"做空信号预测准确（胜率{calib_095['short_actual_winrate']:.1%}）")
        
        # 风险提示
        if returns.loc['std', 'long_1h'] > 0.03:
            conclusion['风险提示'].append(f"做多策略波动较大（标准差{returns.loc['std', 'long_1h']:.2%}）")
        if returns.loc['std', 'short_1h'] > 0.03:
            conclusion['风险提示'].append(f"做空策略波动较大（标准差{returns.loc['std', 'short_1h']:.2%}）")
        if len(conclusion['主要优势']) < 2:
            conclusion['风险提示'].append("策略优势不明显")
        
        # 改进建议
        if not pass_threshold:
            if long_effectiveness <= 0.005:
                conclusion['改进建议'].append("优化做多信号生成逻辑")
            if short_effectiveness <= 0.005:
                conclusion['改进建议'].append("改进做空信号过滤条件")
            if calib_095['long_actual_winrate'] < 0.6:
                conclusion['改进建议'].append("校准做多概率预测模型")
            if calib_095['short_actual_winrate'] < 0.6:
                conclusion['改进建议'].append("校准做空概率预测模型")
        
        return pd.DataFrame.from_dict(conclusion, orient='index')

    # 在执行评估后添加结论生成
    conclusion_df = generate_conclusion(model_report)
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
        conclusion_df.to_excel(writer, sheet_name='策略评估结论', index=True)
if __name__ == "__main__":
    log_file = "logs/nn_signals_20250628.xlsx" 
    analyze_signal_log(log_file)