from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import pandas as pd
def analyze_features(df):
    # 平稳性检验
    results = []
    for col in ['close', 'atr', 'rsi']:
        result = adfuller(df[col].dropna())
        results.append((col, result[1]))  # 保存p-value
    
    # 可视化趋势
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    for idx, col in enumerate(['close', 'macd', 'volume_ratio']):
        axes[idx].plot(df[col])
        axes[idx].set_title(f'{col} Trend Analysis')
    plt.savefig('trend_analysis.png')
    
    return pd.DataFrame(results, columns=['Feature', 'ADF p-value'])