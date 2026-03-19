"""
工具函数模块
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
from config import Config

def load_data(file_path):
    """
    加载数据文件
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        DataFrame: 加载的数据
    """
    try:
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        return df
    except Exception as e:
        print(f"加载数据出错: {e}")
        return None

def preprocess_data(df):
    """
    数据预处理
    
    Args:
        df: 原始数据DataFrame
        
    Returns:
        DataFrame: 预处理后的数据
    """
    # 复制数据，避免修改原始数据
    df_processed = df.copy()
    
    # 处理缺失值
    df_processed = df_processed.dropna()
    
    # 移除重复值
    df_processed = df_processed.drop_duplicates()
    
    # 处理异常值 (使用IQR方法)
    for col in Config.GAIT_FEATURES + Config.SENSOR_FEATURES:
        if col in df_processed.columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 将异常值替换为边界值
            df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
    
    return df_processed

def normalize_features(df, features, method='standard'):
    """
    特征标准化/归一化
    
    Args:
        df: 数据DataFrame
        features: 需要标准化的特征列表
        method: 标准化方法 ('standard' 或 'minmax')
        
    Returns:
        DataFrame: 标准化后的数据
    """
    df_normalized = df.copy()
    
    # 选择标准化方法
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    # 只对指定特征进行标准化
    valid_features = [f for f in features if f in df.columns]
    if valid_features:
        df_normalized[valid_features] = scaler.fit_transform(df[valid_features])
    
    return df_normalized

def extract_time_features(df):
    """
    提取时间序列特征
    
    Args:
        df: 包含时间序列数据的DataFrame
        
    Returns:
        DataFrame: 添加了时间特征的数据
    """
    df_with_features = df.copy()
    
    # 确保数据按时间戳排序
    if 'timestamp' in df.columns:
        df_with_features = df_with_features.sort_values('timestamp')
    
    # 计算移动平均
    window_size = 5
    for col in Config.SENSOR_FEATURES:
        if col in df.columns:
            df_with_features[f'{col}_ma'] = df[col].rolling(window=window_size, min_periods=1).mean()
    
    # 计算变化率
    for col in Config.GAIT_FEATURES:
        if col in df.columns:
            df_with_features[f'{col}_change'] = df[col].pct_change().fillna(0)
    
    return df_with_features

def prepare_chart_data(df, x_col, y_cols, group_col=None):
    """
    准备图表数据
    
    Args:
        df: 数据DataFrame
        x_col: X轴列名
        y_cols: Y轴列名列表
        group_col: 分组列名
        
    Returns:
        dict: 图表数据JSON
    """
    chart_data = {'xAxis': df[x_col].tolist()}
    
    if group_col:
        # 按组准备数据
        groups = df[group_col].unique()
        series = []
        
        for i, group in enumerate(groups):
            group_df = df[df[group_col] == group]
            
            for y_col in y_cols:
                series.append({
                    'name': f'{group} - {y_col}',
                    'type': 'line',
                    'data': group_df[y_col].tolist(),
                    'color': Config.CHART_COLORS[i % len(Config.CHART_COLORS)]
                })
    else:
        # 不分组
        series = []
        for i, y_col in enumerate(y_cols):
            series.append({
                'name': y_col,
                'type': 'line',
                'data': df[y_col].tolist(),
                'color': Config.CHART_COLORS[i % len(Config.CHART_COLORS)]
            })
    
    chart_data['series'] = series
    return json.dumps(chart_data)

def calculate_gait_metrics(df):
    """
    计算步态指标
    
    Args:
        df: 步态数据DataFrame
        
    Returns:
        dict: 计算的指标
    """
    metrics = {}
    
    # 基本统计指标
    for feature in Config.GAIT_FEATURES:
        if feature in df.columns:
            metrics[f'{feature}_mean'] = df[feature].mean()
            metrics[f'{feature}_std'] = df[feature].std()
            metrics[f'{feature}_min'] = df[feature].min()
            metrics[f'{feature}_max'] = df[feature].max()
    
    # 步态类型分布
    if 'gait_type' in df.columns:
        gait_counts = df['gait_type'].value_counts()
        metrics['gait_type_counts'] = {
            Config.GAIT_TYPES.get(gait, gait): count 
            for gait, count in gait_counts.items()
        }
    
    # 左右脚对称性
    if all(col in df.columns for col in ['left_0', 'right_0']):
        left_pressure = df[['left_0', 'left_1', 'left_2', 'left_3', 'left_4']].mean(axis=1)
        right_pressure = df[['right_0', 'right_1', 'right_2', 'right_3', 'right_4']].mean(axis=1)
        
        # 计算左右脚压力差异
        pressure_diff = (left_pressure - right_pressure).abs()
        metrics['pressure_diff_mean'] = pressure_diff.mean()
        metrics['pressure_diff_std'] = pressure_diff.std()
        
        # 计算左右脚压力相关性
        metrics['left_right_correlation'] = left_pressure.corr(right_pressure)
    
    return metrics
