import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from datetime import datetime
import os

def generate_summary_report(trades_file):
    # 读取交易数据并按时间排序
    trades_df = pd.read_excel(trades_file, sheet_name='交易明细')
    trades_df = trades_df.sort_values('timestamp')
    
    # 计算基础指标
    initial_capital = 10000  # 从配置获取或根据首笔交易推算
    final_equity = initial_capital + trades_df['收益'].sum()
    total_return = (final_equity - initial_capital) / initial_capital
    
    # 计算资金曲线（修正版本）
    equity_curve = [initial_capital]
    valid_timestamps = [trades_df['timestamp'].iloc[0] - pd.DateOffset(days=1)]  # 初始时间戳
    
    for idx, row in trades_df.iterrows():
        if pd.notnull(row['收益']):
            equity_curve.append(equity_curve[-1] + row['收益'])
            valid_timestamps.append(row['timestamp'])  # 只记录有收益的交易时间戳
    
    # 时间周期计算
    start_time = trades_df['timestamp'].min()
    end_time = trades_df['timestamp'].max()
    days = (end_time - start_time).days or 1
    
    # 关键指标计算
    returns = trades_df['收益'].dropna()
    profitable_trades = len(returns[returns > 0])
    loss_trades = len(returns[returns < 0])
    win_rate = profitable_trades / len(returns) if len(returns) > 0 else 0
    avg_profit = returns.mean()
    
    # 年化收益率
    annualized_return = (1 + total_return) ** (365/days) - 1
    
    # 夏普比率（假设无风险利率为0）
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
    
    # 最大回撤计算
    equity_series = pd.Series(equity_curve)
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min()

    # 生成可视化图表
    plt.figure(figsize=(16, 8))  # 加宽画布
    
    # 获取时间戳列表
    timestamps = pd.to_datetime(trades_df['timestamp']).tolist()
    # 资金曲线时间戳对齐（首日补零）
    timestamps = [timestamps[0] - pd.DateOffset(days=1)] + timestamps
    
    # 绘制资金曲线（时间戳与资金点数量对齐）
    plt.plot(valid_timestamps[:len(equity_curve)], equity_curve, label='资金曲线', linewidth=2)
    
    # 设置时间轴格式
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签
    
    # 添加标签和网格
    plt.title('资金曲线图（按时间序列）', fontsize=14)
    plt.xlabel('交易时间', fontsize=12)
    plt.ylabel('资金量 (USDT)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 限制显示密度：每30天一个刻度
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=1))
    
    # 添加最大回撤标注
    maxdd_idx = drawdown.idxmin()
    plt.annotate(f'最大回撤: {max_drawdown*100:.1f}%', 
                xy=(timestamps[maxdd_idx], equity_curve[maxdd_idx]),
                xytext=(timestamps[maxdd_idx], equity_curve[maxdd_idx]*0.9),
                arrowprops=dict(facecolor='red', shrink=0.05))
    
    # 保存图表
    chart_path = os.path.join(os.path.dirname(trades_file), 'equity_chart.png')
    plt.savefig(chart_path)
    plt.close()

    # 构建汇总数据
    summary_data = {
        '指标': [
            '初始资金 (USDT)', '最终资产 (USDT)', '总收益率 (%)',
            '年化收益率 (%)', '夏普比率', '最大回撤 (%)',
            '总交易次数', '胜率 (%)', '平均每笔收益 (USDT)',
            '盈利交易次数', '亏损交易次数'
        ],
        '数值': [
            initial_capital,
            final_equity,
            total_return * 100,
            annualized_return * 100,
            sharpe_ratio,
            max_drawdown * 100,
            len(trades_df),
            win_rate * 100,
            avg_profit,
            profitable_trades,
            loss_trades
        ]
    }

    # 生成报告文件
    report_path = trades_file.replace('_trades_', '_report_')
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        # 保存汇总数据
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='汇总报告', index=False)
        
        # 保存原始数据
        trades_df.to_excel(writer, sheet_name='交易明细', index=False)
        
        # 添加图表到Excel
        worksheet = writer.sheets['汇总报告']
        img = plt.imread(chart_path)
        # 需要安装openpyxl并添加图片插入逻辑

    return report_path

# 使用示例
if __name__ == "__main__":
    input_file = "回测报告/SOL_trades_20250617_150111.xlsx"
    output_file = generate_summary_report(input_file)
    print(f"报告生成完成：{output_file}")