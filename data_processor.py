"""
数据处理模块
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from config import Config
from utils import load_data, preprocess_data, normalize_features, extract_time_features

class DataProcessor:
    def __init__(self):
        """初始化数据处理器"""
        self.original_data = None
        self.synthetic_data = None
        self.subjects_data = None
        self.processed_data = None
        self.train_data = None
        self.test_data = None

    def load_all_data(self, max_samples=1000):
        """
        加载所有数据

        Args:
            max_samples: 最大样本数，用于限制加载的数据量

        Returns:
            bool: 是否成功加载数据
        """
        print(f"加载数据，限制最大样本数为 {max_samples}...")
        data_loaded = False

        # 加载合成数据（优先）
        try:
            synthetic_data_path = 'synthetic_gait_data.xlsx'
            if os.path.exists(synthetic_data_path):
                # 使用pandas读取Excel文件
                print(f"从 {synthetic_data_path} 加载合成数据...")
                self.synthetic_data = pd.read_excel(synthetic_data_path)

                # 如果指定了最大样本数，则限制加载的数据量
                if max_samples and len(self.synthetic_data) > max_samples:
                    print(f"合成数据超过最大样本数限制，进行采样...")
                    self.synthetic_data = self.synthetic_data.sample(n=max_samples, random_state=42)

                print(f"合成数据加载完成，形状: {self.synthetic_data.shape}")
                data_loaded = True
            else:
                print(f"合成数据文件不存在: {synthetic_data_path}")
        except Exception as e:
            print(f"加载合成数据出错: {e}")
            self.synthetic_data = None

        # 如果合成数据加载失败，尝试加载原始数据
        if not data_loaded:
            try:
                if os.path.exists(Config.ORIGINAL_DATA_PATH):
                    print(f"从 {Config.ORIGINAL_DATA_PATH} 加载原始数据...")
                    self.original_data = load_data(Config.ORIGINAL_DATA_PATH)

                    # 如果指定了最大样本数，则限制加载的数据量
                    if self.original_data is not None and max_samples and len(self.original_data) > max_samples:
                        print(f"原始数据超过最大样本数限制，进行采样...")
                        self.original_data = self.original_data.sample(n=max_samples, random_state=42)

                    if self.original_data is not None:
                        print(f"原始数据加载完成，形状: {self.original_data.shape}")
                        data_loaded = True
                    else:
                        print("原始数据加载失败")
                else:
                    print(f"原始数据文件不存在: {Config.ORIGINAL_DATA_PATH}")
            except Exception as e:
                print(f"加载原始数据出错: {e}")
                self.original_data = None

        # 加载受试者信息
        try:
            subject_profiles_path = 'subject_profiles.xlsx'
            if os.path.exists(subject_profiles_path):
                print(f"从 {subject_profiles_path} 加载受试者数据...")
                self.subjects_data = pd.read_excel(subject_profiles_path)
                print(f"受试者数据加载完成，形状: {self.subjects_data.shape}")
            else:
                print(f"受试者数据文件不存在: {subject_profiles_path}")
                self.subjects_data = None
        except Exception as e:
            print(f"加载受试者数据出错: {e}")
            self.subjects_data = None

        # 设置处理后的数据
        if self.synthetic_data is not None:
            print("使用合成数据作为处理数据")
            self.processed_data = self.synthetic_data.copy()
        elif self.original_data is not None:
            print("使用原始数据作为处理数据")
            self.processed_data = self.original_data.copy()
        else:
            print("未能加载任何数据")
            self.processed_data = None
            return False

        # 验证数据格式
        required_columns = ['subject_id', 'timestamp']
        missing_columns = [col for col in required_columns if col not in self.processed_data.columns]
        if missing_columns:
            print(f"数据缺少必要的列: {missing_columns}")
            # 尝试添加缺失的列
            if 'subject_id' not in self.processed_data.columns:
                self.processed_data['subject_id'] = [f'S{i:03d}' for i in range(1, len(self.processed_data) + 1)]
            if 'timestamp' not in self.processed_data.columns:
                self.processed_data['timestamp'] = list(range(len(self.processed_data)))

        print(f"数据加载完成，总行数: {len(self.processed_data)}")
        return True

    def process_data(self):
        """处理数据"""
        if self.processed_data is None:
            if not self.load_all_data():
                return False

        # 数据预处理
        self.processed_data = preprocess_data(self.processed_data)

        # 特征标准化
        features_to_normalize = Config.SENSOR_FEATURES + Config.GAIT_FEATURES
        self.processed_data = normalize_features(
            self.processed_data,
            features_to_normalize,
            method='standard'
        )

        # 提取时间特征
        self.processed_data = extract_time_features(self.processed_data)

        return True

    def split_train_test(self):
        """分割训练集和测试集"""
        if self.processed_data is None:
            if not self.process_data():
                return False

        # 确保存在目标变量
        if 'gait_type' in self.processed_data.columns:
            # 选择特征和目标
            features = Config.SENSOR_FEATURES + Config.GAIT_FEATURES
            valid_features = [f for f in features if f in self.processed_data.columns]

            X = self.processed_data[valid_features]
            y = self.processed_data['gait_type']

            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=Config.TEST_SIZE,
                random_state=Config.RANDOM_STATE,
                stratify=y
            )

            # 保存训练集和测试集
            self.train_data = pd.concat([X_train, y_train], axis=1)
            self.test_data = pd.concat([X_test, y_test], axis=1)

            return True
        else:
            print("数据中没有目标变量 'gait_type'")
            return False

    def get_data_summary(self):
        """获取数据摘要"""
        if self.processed_data is None:
            if not self.process_data():
                return {}

        summary = {
            'total_samples': len(self.processed_data),
            'features': list(self.processed_data.columns),
            'missing_values': self.processed_data.isnull().sum().sum(),
        }

        # 步态类型分布
        if 'gait_type' in self.processed_data.columns:
            gait_counts = self.processed_data['gait_type'].value_counts().to_dict()
            summary['gait_type_distribution'] = {
                Config.GAIT_TYPES.get(gait, gait): count
                for gait, count in gait_counts.items()
            }

        # 受试者信息
        if 'subject_id' in self.processed_data.columns:
            summary['subject_count'] = self.processed_data['subject_id'].nunique()

        # 基本统计信息
        for feature in Config.GAIT_FEATURES:
            if feature in self.processed_data.columns:
                summary[f'{feature}_stats'] = {
                    'mean': self.processed_data[feature].mean(),
                    'std': self.processed_data[feature].std(),
                    'min': self.processed_data[feature].min(),
                    'max': self.processed_data[feature].max()
                }

        return summary

    def get_data_by_gait_type(self, gait_type):
        """获取特定步态类型的数据"""
        if self.processed_data is None:
            if not self.process_data():
                return None

        if 'gait_type' in self.processed_data.columns:
            return self.processed_data[self.processed_data['gait_type'] == gait_type]
        else:
            return None

    def get_data_by_subject(self, subject_id):
        """获取特定受试者的数据"""
        if self.processed_data is None:
            if not self.process_data():
                return None

        if 'subject_id' in self.processed_data.columns:
            subject_data = self.processed_data[self.processed_data['subject_id'] == subject_id]
            if not subject_data.empty:
                return subject_data

        # 如果没有找到数据，生成模拟数据
        print(f"未找到用户 {subject_id} 的数据，生成模拟数据")
        return self.generate_mock_user_data(subject_id)

    def generate_mock_data(self, sample_size=1000):
        """
        生成模拟数据用于测试和演示

        Args:
            sample_size: 生成的样本数量

        Returns:
            DataFrame: 模拟数据
        """
        print(f"生成 {sample_size} 条模拟数据...")

        # 创建受试者ID
        subject_ids = [f'S{i:03d}' for i in range(1, 31)]

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

        for _ in range(sample_size):
            # 选择随机受试者
            subject_id = random.choice(subject_ids)

            # 选择随机步态类型（加权概率）
            gait_type = random.choices(
                ['walking', 'running', 'jumping', 'stairs_up', 'stairs_down'],
                weights=[0.5, 0.2, 0.1, 0.1, 0.1],
                k=1
            )[0]

            # 获取此步态类型的参数
            params = gait_params[gait_type]

            # 生成用户信息
            age = random.randint(18, 80)
            gender = random.choice(['male', 'female'])
            height = round(np.random.normal(170 if gender == 'male' else 160, 10), 1)
            weight = round(np.random.normal(75 if gender == 'male' else 60, 15), 1)
            fitness_level = random.choice(['sedentary', 'light_active', 'moderate_active', 'very_active'])
            medical_condition = random.choice(['none', 'arthritis', 'parkinsons', 'stroke_recovery',
                                              'multiple_sclerosis', 'diabetes']) if random.random() < 0.3 else 'none'

            # 生成步态参数
            cadence = random.uniform(*params['cadence_range'])
            stride_length = random.uniform(*params['stride_length_range'])
            speed = random.uniform(*params['speed_range'])
            symmetry = random.uniform(*params['symmetry_range'])

            # 生成传感器数据
            acc_x = random.uniform(*params['acc_range'])
            acc_y = random.uniform(*params['acc_range'])
            acc_z = random.uniform(*params['acc_range'])

            gyro_x = random.uniform(*params['gyro_range'])
            gyro_y = random.uniform(*params['gyro_range'])
            gyro_z = random.uniform(*params['gyro_range'])

            # 生成压力传感器数据
            left_pressures = [max(0, random.uniform(*params['pressure_range']) + random.normalvariate(0, 5)) for _ in range(5)]
            right_pressures = [max(0, p * symmetry + random.normalvariate(0, 10)) for p in left_pressures]

            # 创建记录
            record = {
                'subject_id': subject_id,
                'timestamp': random.randint(0, 100),
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
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'fitness_level': fitness_level,
                'medical_condition': medical_condition
            }

            data.append(record)

        # 转换为DataFrame
        mock_df = pd.DataFrame(data)
        print(f"模拟数据生成完成，形状: {mock_df.shape}")

        return mock_df

    def generate_mock_user_data(self, subject_id):
        """
        为特定用户生成模拟数据

        Args:
            subject_id: 用户ID

        Returns:
            DataFrame: 用户的模拟数据
        """
        # 生成100个样本点
        data = []

        # 随机选择步态类型
        gait_type = random.choice(['walking', 'running', 'jumping', 'stairs_up', 'stairs_down'])

        # 用户信息
        age = random.randint(18, 80)
        gender = random.choice(['male', 'female'])
        height = round(np.random.normal(170 if gender == 'male' else 160, 10), 1)
        weight = round(np.random.normal(75 if gender == 'male' else 60, 15), 1)
        fitness_level = random.choice(['sedentary', 'light_active', 'moderate_active', 'very_active'])
        medical_condition = 'none'

        for i in range(100):
            # 加速度计数据
            acc_x = np.sin(i/10) + np.random.normal(0, 0.2)
            acc_y = np.cos(i/10) + np.random.normal(0, 0.2)
            acc_z = 0.5 * np.sin(i/5) + np.random.normal(0, 0.2)

            # 陀螺仪数据
            gyro_x = 20 * np.sin(i/8) + np.random.normal(0, 3)
            gyro_y = 15 * np.cos(i/7) + np.random.normal(0, 3)
            gyro_z = 25 * np.sin(i/10) + np.random.normal(0, 3)

            # 足底压力数据
            left_0 = 70 + 10 * np.sin(i/15) + np.random.normal(0, 5)
            left_1 = 80 + 10 * np.sin(i/12) + np.random.normal(0, 5)
            left_2 = 75 + 10 * np.sin(i/10) + np.random.normal(0, 5)
            left_3 = 85 + 10 * np.sin(i/8) + np.random.normal(0, 5)
            left_4 = 65 + 10 * np.sin(i/6) + np.random.normal(0, 5)

            right_0 = 75 + 10 * np.cos(i/15) + np.random.normal(0, 5)
            right_1 = 85 + 10 * np.cos(i/12) + np.random.normal(0, 5)
            right_2 = 80 + 10 * np.cos(i/10) + np.random.normal(0, 5)
            right_3 = 90 + 10 * np.cos(i/8) + np.random.normal(0, 5)
            right_4 = 70 + 10 * np.cos(i/6) + np.random.normal(0, 5)

            # 步态参数
            cadence = 90 + 5 * np.sin(i/20) + np.random.normal(0, 1)
            stride_length = 80 + 10 * np.sin(i/25) + np.random.normal(0, 2)
            speed = 4 + np.sin(i/30) + np.random.normal(0, 0.2)
            symmetry = 0.9 + 0.05 * np.sin(i/35) + np.random.normal(0, 0.01)

            # 创建记录
            record = {
                'subject_id': subject_id,
                'timestamp': i,
                'acc_x': acc_x,
                'acc_y': acc_y,
                'acc_z': acc_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z,
                'left_0': left_0,
                'left_1': left_1,
                'left_2': left_2,
                'left_3': left_3,
                'left_4': left_4,
                'right_0': right_0,
                'right_1': right_1,
                'right_2': right_2,
                'right_3': right_3,
                'right_4': right_4,
                'cadence': cadence,
                'stride_length': stride_length,
                'speed': speed,
                'symmetry': symmetry,
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'fitness_level': fitness_level,
                'medical_condition': medical_condition,
                'gait_type': gait_type
            }

            data.append(record)

        # 转换为DataFrame
        return pd.DataFrame(data)
