"""
生成合成步态数据
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os
import sys

# 设置随机种子以确保可重复性
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

# 读取现有数据以了解其结构
try:
    existing_df = pd.read_excel('数据.xlsx')
    print(f"加载原始数据成功，形状: {existing_df.shape}")
except Exception as e:
    print(f"加载原始数据出错: {e}")
    existing_df = None

# 生成合成步态数据的函数
def generate_gait_data(num_samples=1000, start_date=datetime(2023, 1, 1),
                       end_date=datetime(2023, 12, 31), subjects=50):
    """
    生成合成步态数据

    Args:
        num_samples: 要生成的样本数量
        start_date: 数据开始日期
        end_date: 数据结束日期
        subjects: 受试者数量

    Returns:
        tuple: (步态数据DataFrame, 受试者信息DataFrame)
    """
    print(f"开始生成{num_samples}条合成步态数据...")

    # 创建受试者档案
    subject_profiles = []
    for i in range(subjects):
        age = random.randint(18, 80)
        gender = random.choice(['male', 'female'])
        height = np.random.normal(170 if gender == 'male' else 160, 10)
        weight = np.random.normal(75 if gender == 'male' else 60, 15)

        # 可能影响步态的医疗状况
        has_condition = random.random() < 0.3  # 30%的概率有状况
        condition = random.choice(['none', 'arthritis', 'parkinsons', 'stroke_recovery',
                                  'multiple_sclerosis', 'diabetes']) if has_condition else 'none'

        # 健身水平影响步态参数
        fitness_level = random.choice(['sedentary', 'light_active', 'moderate_active', 'very_active'])

        subject_profiles.append({
            'subject_id': f'S{i+1:03d}',
            'age': age,
            'gender': gender,
            'height': height,
            'weight': weight,
            'medical_condition': condition,
            'fitness_level': fitness_level
        })

    # 不同步态类型的参数
    gait_params = {
        'walking': {
            'cadence_range': (85, 120),
            'stride_length_range': (70, 100),
            'speed_range': (2, 5),
            'symmetry_range': (0.85, 1.0),
            'acc_range': (-2, 2),
            'gyro_range': (-50, 50),
            'pressure_range': (0, 100)
        },
        'running': {
            'cadence_range': (150, 200),
            'stride_length_range': (100, 150),
            'speed_range': (8, 15),
            'symmetry_range': (0.8, 1.0),
            'acc_range': (-5, 5),
            'gyro_range': (-100, 100),
            'pressure_range': (20, 150)
        },
        'jumping': {
            'cadence_range': (40, 80),
            'stride_length_range': (50, 120),
            'speed_range': (1, 3),
            'symmetry_range': (0.7, 1.0),
            'acc_range': (-10, 10),
            'gyro_range': (-150, 150),
            'pressure_range': (50, 200)
        },
        'stairs_up': {
            'cadence_range': (70, 100),
            'stride_length_range': (50, 80),
            'speed_range': (1, 3),
            'symmetry_range': (0.75, 0.95),
            'acc_range': (-3, 3),
            'gyro_range': (-70, 70),
            'pressure_range': (30, 120)
        },
        'stairs_down': {
            'cadence_range': (75, 110),
            'stride_length_range': (55, 85),
            'speed_range': (1.5, 3.5),
            'symmetry_range': (0.7, 0.9),
            'acc_range': (-3, 3),
            'gyro_range': (-70, 70),
            'pressure_range': (40, 130)
        }
    }

    # 生成数据
    data = []

    for _ in range(num_samples):
        # 选择随机受试者
        subject = random.choice(subject_profiles)
        subject_id = subject['subject_id']

        # 选择随机日期和时间
        session_date = fake.date_time_between(start_date=start_date, end_date=end_date)

        # 选择随机步态类型（加权概率）
        gait_type = random.choices(
            ['walking', 'running', 'jumping', 'stairs_up', 'stairs_down'],
            weights=[0.5, 0.2, 0.1, 0.1, 0.1],
            k=1
        )[0]

        # 获取此步态类型的参数
        params = gait_params[gait_type]

        # 生成一个时间序列（通常是1-3分钟的数据）
        sequence_length = random.randint(60, 180)  # 1-3分钟

        # 根据受试者特征调整参数
        age_factor = 1.0 - max(0, (subject['age'] - 30) / 100)  # 年龄增加，性能下降
        condition_factor = 0.7 if subject['medical_condition'] != 'none' else 1.0  # 医疗状况影响步态
        fitness_factor = {
            'sedentary': 0.8,
            'light_active': 0.9,
            'moderate_active': 1.0,
            'very_active': 1.1
        }[subject['fitness_level']]

        # 将因素应用于基本参数
        cadence_base = random.uniform(*params['cadence_range']) * age_factor * condition_factor * fitness_factor
        stride_length_base = random.uniform(*params['stride_length_range']) * age_factor * condition_factor * fitness_factor
        speed_base = random.uniform(*params['speed_range']) * age_factor * condition_factor * fitness_factor
        symmetry_base = random.uniform(*params['symmetry_range']) * condition_factor

        # 生成具有一些自然变化的序列
        for i in range(sequence_length):
            timestamp = i / 100  # 100 Hz采样率

            # 添加一些随时间的自然变化
            time_factor = 1.0 + 0.05 * np.sin(timestamp / 10)  # 小的正弦变化

            # 计算具有变化的参数
            cadence = cadence_base * time_factor + np.random.normal(0, 1)
            stride_length = stride_length_base * time_factor + np.random.normal(0, 2)
            speed = speed_base * time_factor + np.random.normal(0, 0.2)
            symmetry = min(1.0, symmetry_base + np.random.normal(0, 0.02))

            # 生成传感器数据
            acc_x = np.random.uniform(*params['acc_range'])
            acc_y = np.random.uniform(*params['acc_range'])
            acc_z = np.random.uniform(*params['acc_range'])

            gyro_x = np.random.uniform(*params['gyro_range'])
            gyro_y = np.random.uniform(*params['gyro_range'])
            gyro_z = np.random.uniform(*params['gyro_range'])

            # 生成压力传感器数据，左右脚之间有相关性
            # 但基于对称性参数有不对称性
            base_pressures = [np.random.uniform(*params['pressure_range']) for _ in range(5)]

            # 左脚压力
            left_pressures = [max(0, p + np.random.normal(0, 5)) for p in base_pressures]

            # 右脚压力（与左脚相关但受对称性影响）
            right_pressures = [max(0, p * symmetry + np.random.normal(0, 10)) for p in base_pressures]

            # 创建记录
            record = {
                'subject_id': subject_id,
                'session_datetime': session_date + timedelta(seconds=timestamp),
                'timestamp': timestamp,
                'cadence': cadence,
                'stride_length': stride_length,
                'speed': speed,
                'symmetry': symmetry,
                'acc_x': acc_x,
                'acc_y': acc_y,
                'acc_z': acc_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z,
                'left_0': left_pressures[0],
                'left_1': left_pressures[1],
                'left_2': left_pressures[2],
                'left_3': left_pressures[3],
                'left_4': left_pressures[4],
                'right_0': right_pressures[0],
                'right_1': right_pressures[1],
                'right_2': right_pressures[2],
                'right_3': right_pressures[3],
                'right_4': right_pressures[4],
                'gait_type': gait_type,
                'age': subject['age'],
                'gender': subject['gender'],
                'height': subject['height'],
                'weight': subject['weight'],
                'medical_condition': subject['medical_condition'],
                'fitness_level': subject['fitness_level']
            }

            data.append(record)

    # 转换为DataFrame
    df = pd.DataFrame(data)

    return df, pd.DataFrame(subject_profiles)

# 生成合成数据
print("开始生成合成数据...")
try:
    # 控制生成的数据量 - 只生成2000条样本
    num_samples = 2000
    print(f"生成 {num_samples} 条合成数据...")
    synthetic_df, subjects_df = generate_gait_data(num_samples=num_samples, subjects=30)

    # 限制每个样本的序列长度，减少总数据量
    max_sequence_length = 20  # 每个样本最多20个时间点
    sample_ids = synthetic_df['subject_id'].unique()
    filtered_rows = []

    print("限制每个样本的序列长度...")
    for sample_id in sample_ids:
        sample_data = synthetic_df[synthetic_df['subject_id'] == sample_id]
        if len(sample_data) > max_sequence_length:
            # 采样数据
            step = len(sample_data) // max_sequence_length
            filtered_rows.append(sample_data.iloc[::step].head(max_sequence_length))
        else:
            filtered_rows.append(sample_data)

    # 合并过滤后的数据
    synthetic_df = pd.concat(filtered_rows, ignore_index=True)
    total_rows = len(synthetic_df)
    print(f"生成合成数据成功，形状: {synthetic_df.shape}")
    print(f"总数据量: {total_rows} 行")

    # 确保数据量不超过5000行
    if total_rows > 5000:
        print(f"数据量超过5000行，进行采样...")
        # 随机采样5000行
        synthetic_df = synthetic_df.sample(n=5000, random_state=42).reset_index(drop=True)
        print(f"采样后数据形状: {synthetic_df.shape}")

    # 保存合成数据
    synthetic_df.to_excel('synthetic_gait_data.xlsx', index=False)
    subjects_df.to_excel('subject_profiles.xlsx', index=False)
    print("合成数据已保存到 synthetic_gait_data.xlsx")
    print("受试者信息已保存到 subject_profiles.xlsx")

    # 与现有数据合并（如果可用）
    if existing_df is not None:
        # 检查现有数据是否具有相同的列
        common_cols = list(set(existing_df.columns).intersection(set(synthetic_df.columns)))
        if common_cols:
            # 仅使用合并的公共列，并限制数据量
            max_rows_per_source = 2500  # 每个数据源最多2500行
            combined_df = pd.concat([
                existing_df[common_cols].head(max_rows_per_source),
                synthetic_df[common_cols].head(max_rows_per_source)
            ], ignore_index=True)
            combined_df.to_excel('combined_gait_data.xlsx', index=False)
            print(f"合并数据已保存，形状: {combined_df.shape}")
except Exception as e:
    print(f"生成合成数据时出错: {e}")

print("数据生成完成!")
