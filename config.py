"""
配置文件
"""

import os

# 基本配置
class Config:
    # 应用配置
    SECRET_KEY = 'your-secret-key'
    DEBUG = True
    
    # 数据文件路径
    ORIGINAL_DATA_PATH = os.path.join(os.getcwd(), '数据.xlsx')
    SYNTHETIC_DATA_PATH = os.path.join(os.getcwd(), 'synthetic_gait_data.xlsx')
    SUBJECTS_DATA_PATH = os.path.join(os.getcwd(), 'subject_profiles.xlsx')
    
    # 模型参数
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # 步态类型映射
    GAIT_TYPES = {
        'walking': '行走',
        'running': '跑步',
        'jumping': '跳跃',
        'stairs_up': '上楼梯',
        'stairs_down': '下楼梯'
    }
    
    # 特征列表
    SENSOR_FEATURES = [
        'acc_x', 'acc_y', 'acc_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'left_0', 'left_1', 'left_2', 'left_3', 'left_4',
        'right_0', 'right_1', 'right_2', 'right_3', 'right_4'
    ]
    
    GAIT_FEATURES = [
        'cadence', 'stride_length', 'speed', 'symmetry'
    ]
    
    DEMOGRAPHIC_FEATURES = [
        'age', 'gender', 'height', 'weight', 
        'medical_condition', 'fitness_level'
    ]
    
    # 图表颜色配置
    CHART_COLORS = [
        '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
        '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc'
    ]
