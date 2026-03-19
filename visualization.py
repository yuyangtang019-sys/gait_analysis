"""
可视化模块
"""

import json
import numpy as np
import pandas as pd
import random
import os
from config import Config

class GaitVisualizer:
    def __init__(self):
        """初始化可视化器"""
        self.excel_data_cache = {}  # 用于缓存从Excel读取的数据

    def load_data_from_excel(self, file_path, sheet_name=0, max_rows=None):
        """
        从Excel文件中加载数据

        Args:
            file_path: Excel文件路径
            sheet_name: 工作表名称或索引
            max_rows: 最大加载行数

        Returns:
            DataFrame: 加载的数据
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None

        # 检查缓存
        cache_key = f"{file_path}_{sheet_name}_{max_rows}"
        if cache_key in self.excel_data_cache:
            return self.excel_data_cache[cache_key]

        try:
            # 读取Excel文件
            if max_rows:
                df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=max_rows)
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name)

            # 缓存数据
            self.excel_data_cache[cache_key] = df

            print(f"从 {file_path} 加载了 {len(df)} 行数据")
            return df
        except Exception as e:
            print(f"加载Excel数据出错: {e}")
            return None

    def create_gait_parameters_chart(self, data, gait_type=None):
        """
        创建步态参数图表

        Args:
            data: 步态数据
            gait_type: 步态类型，如果指定则只显示该类型的数据

        Returns:
            dict: 图表配置
        """
        # 检查数据是否为空
        if data is None or len(data) == 0:
            # 返回模拟数据
            return self._create_mock_gait_params_chart()

        if gait_type:
            filtered_data = data[data['gait_type'] == gait_type]
        else:
            filtered_data = data

        if len(filtered_data) == 0:
            # 返回模拟数据
            return self._create_mock_gait_params_chart()

        # 选择要显示的参数
        params = [col for col in Config.GAIT_FEATURES if col in filtered_data.columns]

        if not params:
            # 返回模拟数据
            return self._create_mock_gait_params_chart()

        # 限制数据点数量以提高性能
        max_points = 100
        if len(filtered_data) > max_points:
            # 采样数据
            step = len(filtered_data) // max_points
            filtered_data = filtered_data.iloc[::step].copy()

        # 创建 x 轴数据 - 使用样本索引而不是 DataFrame 索引
        x_data = list(range(len(filtered_data)))

        # 准备数据
        chart_data = []
        for param in params:
            chart_data.append({
                'name': param,
                'type': 'line',
                'data': filtered_data[param].tolist()
            })

        # 创建图表配置
        chart_config = {
            'title': {
                'text': f'步态参数分析 {Config.GAIT_TYPES.get(gait_type, gait_type) if gait_type else "所有类型"}'
            },
            'tooltip': {
                'trigger': 'axis'
            },
            'legend': {
                'data': params
            },
            'grid': {
                'left': '3%',
                'right': '4%',
                'bottom': '3%',
                'containLabel': True
            },
            'xAxis': {
                'type': 'category',
                'boundaryGap': False,
                'data': x_data
            },
            'yAxis': {
                'type': 'value'
            },
            'series': chart_data
        }

        return chart_config

    def _create_mock_gait_params_chart(self):
        """创建模拟步态参数图表"""
        # 使用 Config 中定义的步态特征
        params = Config.GAIT_FEATURES

        # 创建模拟数据
        x_data = list(range(100))
        series_data = []

        for i, param in enumerate(params):
            # 生成不同的正弦波形
            if param == 'cadence':
                data = [90 + 10 * np.sin(i/10) + np.random.normal(0, 1) for i in range(100)]
            elif param == 'stride_length':
                data = [80 + 15 * np.sin(i/12) + np.random.normal(0, 2) for i in range(100)]
            elif param == 'speed':
                data = [4 + 1.5 * np.sin(i/15) + np.random.normal(0, 0.3) for i in range(100)]
            elif param == 'symmetry':
                data = [0.9 + 0.05 * np.sin(i/20) + np.random.normal(0, 0.01) for i in range(100)]
            else:
                data = [np.random.normal(50, 10) for _ in range(100)]

            series_data.append({
                'name': param,
                'type': 'line',
                'data': data
            })

        # 创建图表配置
        chart_config = {
            'title': {
                'text': '步态参数分析 (模拟数据)'
            },
            'tooltip': {
                'trigger': 'axis'
            },
            'legend': {
                'data': params
            },
            'grid': {
                'left': '3%',
                'right': '4%',
                'bottom': '3%',
                'containLabel': True
            },
            'xAxis': {
                'type': 'category',
                'boundaryGap': False,
                'data': x_data
            },
            'yAxis': {
                'type': 'value'
            },
            'series': series_data
        }

        return chart_config

    def create_gait_type_distribution_chart(self, data):
        """
        创建步态类型分布图表

        Args:
            data: 步态数据

        Returns:
            dict: 图表配置
        """
        if 'gait_type' not in data.columns:
            return {}

        # 计算步态类型分布
        gait_counts = data['gait_type'].value_counts()

        # 准备数据
        chart_data = []
        for gait_type, count in gait_counts.items():
            chart_data.append({
                'name': Config.GAIT_TYPES.get(gait_type, gait_type),
                'value': int(count)
            })

        # 创建图表配置
        chart_config = {
            'title': {
                'text': '步态类型分布',
                'left': 'center'
            },
            'tooltip': {
                'trigger': 'item',
                'formatter': '{a} <br/>{b}: {c} ({d}%)'
            },
            'legend': {
                'orient': 'vertical',
                'left': 'left',
                'data': [item['name'] for item in chart_data]
            },
            'series': [
                {
                    'name': '步态类型',
                    'type': 'pie',
                    'radius': ['50%', '70%'],
                    'avoidLabelOverlap': False,
                    'label': {
                        'show': False,
                        'position': 'center'
                    },
                    'emphasis': {
                        'label': {
                            'show': True,
                            'fontSize': '30',
                            'fontWeight': 'bold'
                        }
                    },
                    'labelLine': {
                        'show': False
                    },
                    'data': chart_data
                }
            ]
        }

        return chart_config

    def create_sensor_data_chart(self, data, sensor_type='acc'):
        """
        创建传感器数据图表

        Args:
            data: 步态数据
            sensor_type: 传感器类型 ('acc' 或 'gyro')

        Returns:
            dict: 图表配置
        """
        if sensor_type == 'acc':
            sensor_cols = ['acc_x', 'acc_y', 'acc_z']
            title = '加速度计数据'
        elif sensor_type == 'gyro':
            sensor_cols = ['gyro_x', 'gyro_y', 'gyro_z']
            title = '陀螺仪数据'
        else:
            return {}

        # 检查列是否存在
        valid_cols = [col for col in sensor_cols if col in data.columns]

        if not valid_cols:
            return {}

        # 准备数据
        chart_data = []
        for col in valid_cols:
            chart_data.append({
                'name': col,
                'type': 'line',
                'data': data[col].tolist()
            })

        # 创建图表配置
        chart_config = {
            'title': {
                'text': title
            },
            'tooltip': {
                'trigger': 'axis'
            },
            'legend': {
                'data': valid_cols
            },
            'grid': {
                'left': '3%',
                'right': '4%',
                'bottom': '3%',
                'containLabel': True
            },
            'xAxis': {
                'type': 'category',
                'boundaryGap': False,
                'data': data.index.tolist()
            },
            'yAxis': {
                'type': 'value'
            },
            'series': chart_data
        }

        return chart_config

    def create_sensor_chart_from_excel(self, file_path, sensor_type='acc', max_rows=100):
        """
        从Excel文件中读取数据并创建传感器数据图表

        Args:
            file_path: Excel文件路径
            sensor_type: 传感器类型 ('acc' 或 'gyro')
            max_rows: 最大加载行数

        Returns:
            dict: 图表配置
        """
        # 加载数据
        data = self.load_data_from_excel(file_path, max_rows=max_rows)
        if data is None or len(data) == 0:
            print(f"无法从 {file_path} 加载数据")
            return {}

        # 确定传感器列
        if sensor_type == 'acc':
            sensor_cols = ['acc_x', 'acc_y', 'acc_z']
            title = '加速度计数据'
        elif sensor_type == 'gyro':
            sensor_cols = ['gyro_x', 'gyro_y', 'gyro_z']
            title = '陀螺仪数据'
        else:
            return {}

        # 检查列是否存在
        valid_cols = [col for col in sensor_cols if col in data.columns]
        if not valid_cols:
            print(f"在 {file_path} 中未找到有效的 {sensor_type} 列")
            return {}

        # 准备数据
        chart_data = []
        for col in valid_cols:
            chart_data.append({
                'name': col,
                'type': 'line',
                'data': data[col].tolist()
            })

        # 创建 x 轴数据
        x_data = list(range(len(data)))

        # 创建图表配置
        chart_config = {
            'title': {
                'text': title
            },
            'tooltip': {
                'trigger': 'axis'
            },
            'legend': {
                'data': valid_cols
            },
            'grid': {
                'left': '3%',
                'right': '4%',
                'bottom': '3%',
                'containLabel': True
            },
            'xAxis': {
                'type': 'category',
                'boundaryGap': False,
                'data': x_data
            },
            'yAxis': {
                'type': 'value'
            },
            'series': chart_data
        }

        return chart_config

    def create_pressure_distribution_chart(self, data, foot='both'):
        """
        创建足底压力分布图表

        Args:
            data: 步态数据
            foot: 脚部选择 ('left', 'right', 或 'both')

        Returns:
            dict: 图表配置
        """
        # 检查数据是否为空
        if data is None or len(data) == 0:
            return self._create_mock_pressure_chart(foot)

        left_cols = ['left_0', 'left_1', 'left_2', 'left_3', 'left_4']
        right_cols = ['right_0', 'right_1', 'right_2', 'right_3', 'right_4']

        # 检查列是否存在
        valid_left_cols = [col for col in left_cols if col in data.columns]
        valid_right_cols = [col for col in right_cols if col in data.columns]

        if (foot == 'left' and not valid_left_cols) or \
           (foot == 'right' and not valid_right_cols) or \
           (foot == 'both' and (not valid_left_cols or not valid_right_cols)):
            return self._create_mock_pressure_chart(foot)

        # 计算平均压力
        left_avg = None
        right_avg = None

        if foot == 'left' or foot == 'both':
            if valid_left_cols:
                left_avg = data[valid_left_cols].mean().tolist()
                # 确保有5个值
                if len(left_avg) < 5:
                    left_avg.extend([0] * (5 - len(left_avg)))
            else:
                left_avg = [70, 85, 65, 75, 80]  # 模拟数据

        if foot == 'right' or foot == 'both':
            if valid_right_cols:
                right_avg = data[valid_right_cols].mean().tolist()
                # 确保有5个值
                if len(right_avg) < 5:
                    right_avg.extend([0] * (5 - len(right_avg)))
            else:
                right_avg = [75, 80, 90, 70, 85]  # 模拟数据

        # 准备数据
        chart_data = []

        if foot == 'left' or foot == 'both':
            chart_data.append({
                'name': '左脚压力',
                'type': 'radar',
                'data': [
                    {
                        'value': left_avg,
                        'name': '左脚压力'
                    }
                ]
            })

        if foot == 'right' or foot == 'both':
            chart_data.append({
                'name': '右脚压力',
                'type': 'radar',
                'data': [
                    {
                        'value': right_avg,
                        'name': '右脚压力'
                    }
                ]
            })

        # 找出最大压力值，用于设置雷达图的最大值
        max_pressure = 100
        if left_avg:
            max_pressure = max(max_pressure, max(left_avg) * 1.2)
        if right_avg:
            max_pressure = max(max_pressure, max(right_avg) * 1.2)

        # 创建图表配置
        chart_config = {
            'title': {
                'text': '足底压力分布'
            },
            'tooltip': {
                'trigger': 'item'
            },
            'legend': {
                'data': ['左脚压力', '右脚压力'] if foot == 'both' else
                        ['左脚压力'] if foot == 'left' else ['右脚压力']
            },
            'radar': {
                'shape': 'circle',
                'indicator': [
                    {'name': '前脚掌内侧', 'max': max_pressure},
                    {'name': '前脚掌外侧', 'max': max_pressure},
                    {'name': '脚掌中部', 'max': max_pressure},
                    {'name': '脚跟内侧', 'max': max_pressure},
                    {'name': '脚跟外侧', 'max': max_pressure}
                ]
            },
            'series': chart_data
        }

        return chart_config

    def _create_mock_pressure_chart(self, foot='both'):
        """创建模拟足底压力分布图表"""
        # 生成模拟数据
        left_avg = [70 + random.randint(-10, 10) for _ in range(5)]
        right_avg = [75 + random.randint(-10, 10) for _ in range(5)]

        # 准备数据
        chart_data = []

        if foot == 'left' or foot == 'both':
            chart_data.append({
                'name': '左脚压力',
                'type': 'radar',
                'data': [
                    {
                        'value': left_avg,
                        'name': '左脚压力',
                        'areaStyle': {
                            'color': 'rgba(52, 152, 219, 0.6)'
                        }
                    }
                ]
            })

        if foot == 'right' or foot == 'both':
            chart_data.append({
                'name': '右脚压力',
                'type': 'radar',
                'data': [
                    {
                        'value': right_avg,
                        'name': '右脚压力',
                        'areaStyle': {
                            'color': 'rgba(46, 204, 113, 0.6)'
                        }
                    }
                ]
            })

        # 创建图表配置
        chart_config = {
            'title': {
                'text': '足底压力分布 (模拟数据)'
            },
            'tooltip': {
                'trigger': 'item'
            },
            'legend': {
                'data': ['左脚压力', '右脚压力'] if foot == 'both' else
                        ['左脚压力'] if foot == 'left' else ['右脚压力']
            },
            'radar': {
                'shape': 'circle',
                'indicator': [
                    {'name': '前脚掌内侧', 'max': 100},
                    {'name': '前脚掌外侧', 'max': 100},
                    {'name': '脚掌中部', 'max': 100},
                    {'name': '脚跟内侧', 'max': 100},
                    {'name': '脚跟外侧', 'max': 100}
                ]
            },
            'series': chart_data
        }

        return chart_config

    def create_pressure_chart_from_excel(self, file_path, foot='both', max_rows=100):
        """
        从Excel文件中读取数据并创建足底压力分布图表

        Args:
            file_path: Excel文件路径
            foot: 脚部选择 ('left', 'right', 或 'both')
            max_rows: 最大加载行数

        Returns:
            dict: 图表配置
        """
        # 加载数据
        data = self.load_data_from_excel(file_path, max_rows=max_rows)
        if data is None or len(data) == 0:
            print(f"无法从 {file_path} 加载数据")
            return self._create_mock_pressure_chart(foot)

        # 确定压力列
        left_cols = ['left_0', 'left_1', 'left_2', 'left_3', 'left_4']
        right_cols = ['right_0', 'right_1', 'right_2', 'right_3', 'right_4']

        # 检查列是否存在
        valid_left_cols = [col for col in left_cols if col in data.columns]
        valid_right_cols = [col for col in right_cols if col in data.columns]

        if (foot == 'left' and not valid_left_cols) or \
           (foot == 'right' and not valid_right_cols) or \
           (foot == 'both' and (not valid_left_cols or not valid_right_cols)):
            print(f"在 {file_path} 中未找到有效的足底压力列")
            return self._create_mock_pressure_chart(foot)

        # 计算平均压力
        left_avg = None
        right_avg = None

        if foot == 'left' or foot == 'both':
            if valid_left_cols:
                left_avg = data[valid_left_cols].mean().tolist()
                # 确保有5个值
                if len(left_avg) < 5:
                    left_avg.extend([0] * (5 - len(left_avg)))
            else:
                left_avg = [70, 85, 65, 75, 80]  # 模拟数据

        if foot == 'right' or foot == 'both':
            if valid_right_cols:
                right_avg = data[valid_right_cols].mean().tolist()
                # 确保有5个值
                if len(right_avg) < 5:
                    right_avg.extend([0] * (5 - len(right_avg)))
            else:
                right_avg = [75, 80, 90, 70, 85]  # 模拟数据

        # 准备数据
        chart_data = []

        if foot == 'left' or foot == 'both':
            chart_data.append({
                'name': '左脚压力',
                'type': 'radar',
                'data': [
                    {
                        'value': left_avg,
                        'name': '左脚压力',
                        'areaStyle': {
                            'color': 'rgba(52, 152, 219, 0.6)'
                        }
                    }
                ]
            })

        if foot == 'right' or foot == 'both':
            chart_data.append({
                'name': '右脚压力',
                'type': 'radar',
                'data': [
                    {
                        'value': right_avg,
                        'name': '右脚压力',
                        'areaStyle': {
                            'color': 'rgba(46, 204, 113, 0.6)'
                        }
                    }
                ]
            })

        # 找出最大压力值，用于设置雷达图的最大值
        max_pressure = 100
        if left_avg:
            max_pressure = max(max_pressure, max(left_avg) * 1.2)
        if right_avg:
            max_pressure = max(max_pressure, max(right_avg) * 1.2)

        # 创建图表配置
        chart_config = {
            'title': {
                'text': '足底压力分布'
            },
            'tooltip': {
                'trigger': 'item'
            },
            'legend': {
                'data': ['左脚压力', '右脚压力'] if foot == 'both' else
                        ['左脚压力'] if foot == 'left' else ['右脚压力']
            },
            'radar': {
                'shape': 'circle',
                'indicator': [
                    {'name': '前脚掌内侧', 'max': max_pressure},
                    {'name': '前脚掌外侧', 'max': max_pressure},
                    {'name': '脚掌中部', 'max': max_pressure},
                    {'name': '脚跟内侧', 'max': max_pressure},
                    {'name': '脚跟外侧', 'max': max_pressure}
                ]
            },
            'series': chart_data
        }

        return chart_config

    def create_correlation_heatmap(self, corr_data):
        """
        创建相关性热力图

        Args:
            corr_data: 相关性数据

        Returns:
            dict: 图表配置
        """
        if not corr_data or 'features' not in corr_data or 'data' not in corr_data:
            return {}

        # 创建图表配置
        chart_config = {
            'title': {
                'text': '特征相关性热力图'
            },
            'tooltip': {
                'position': 'top'
            },
            'grid': {
                'height': '50%',
                'top': '10%'
            },
            'xAxis': {
                'type': 'category',
                'data': corr_data['features'],
                'splitArea': {
                    'show': True
                }
            },
            'yAxis': {
                'type': 'category',
                'data': corr_data['features'],
                'splitArea': {
                    'show': True
                }
            },
            'visualMap': {
                'min': -1,
                'max': 1,
                'calculable': True,
                'orient': 'horizontal',
                'left': 'center',
                'bottom': '15%',
                'inRange': {
                    'color': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8',
                             '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
                }
            },
            'series': [
                {
                    'name': '相关系数',
                    'type': 'heatmap',
                    'data': corr_data['data'],
                    'label': {
                        'show': False
                    },
                    'emphasis': {
                        'itemStyle': {
                            'shadowBlur': 10,
                            'shadowColor': 'rgba(0, 0, 0, 0.5)'
                        }
                    }
                }
            ]
        }

        return chart_config

    def create_pca_scatter_chart(self, pca_data):
        """
        创建PCA散点图

        Args:
            pca_data: PCA结果数据

        Returns:
            dict: 图表配置
        """
        if not pca_data or 'components' not in pca_data:
            return {}

        # 准备数据
        series_data = []

        for component in pca_data['components']:
            data_points = []
            for point in component['data']:
                data_points.append([point[0], point[1]])

            series_data.append({
                'name': component['name'],
                'type': 'scatter',
                'data': data_points
            })

        # 创建图表配置
        chart_config = {
            'title': {
                'text': 'PCA降维分析'
            },
            'tooltip': {
                'trigger': 'item',
                'formatter': '{a} <br/>(PC1: {c[0]}, PC2: {c[1]})'
            },
            'legend': {
                'data': [component['name'] for component in pca_data['components']]
            },
            'xAxis': {
                'type': 'value',
                'name': '主成分1',
                'splitLine': {
                    'show': False
                }
            },
            'yAxis': {
                'type': 'value',
                'name': '主成分2',
                'splitLine': {
                    'show': False
                }
            },
            'series': series_data
        }

        return chart_config

    def create_gait_comparison_radar_chart(self, comparison_data):
        """
        创建步态类型比较雷达图

        Args:
            comparison_data: 步态类型比较数据

        Returns:
            dict: 图表配置
        """
        if not comparison_data or 'features' not in comparison_data or 'gait_types' not in comparison_data:
            return {}

        # 准备指标
        indicators = []
        for feature in comparison_data['features']:
            # 找出该特征的最大值
            max_value = 0
            for gait_data in comparison_data['data']:
                if feature in gait_data and gait_data[feature] > max_value:
                    max_value = gait_data[feature]

            # 设置最大值的1.2倍作为雷达图的最大值
            indicators.append({
                'name': feature,
                'max': max_value * 1.2
            })

        # 准备数据
        series_data = []
        for i, gait_type in enumerate(comparison_data['gait_types']):
            values = []
            for feature in comparison_data['features']:
                values.append(comparison_data['data'][i].get(feature, 0))

            series_data.append({
                'value': values,
                'name': gait_type
            })

        # 创建图表配置
        chart_config = {
            'title': {
                'text': '步态类型特征比较'
            },
            'tooltip': {},
            'legend': {
                'data': comparison_data['gait_types']
            },
            'radar': {
                'indicator': indicators
            },
            'series': [
                {
                    'type': 'radar',
                    'data': series_data
                }
            ]
        }

        return chart_config

    def create_prediction_results_chart(self, predictions, prediction_type):
        """
        创建预测结果图表

        Args:
            predictions: 预测结果
            prediction_type: 预测类型 ('gait', 'fatigue', 'fall_risk', 'health')

        Returns:
            dict: 图表配置
        """
        if not predictions or 'success' not in predictions or not predictions['success']:
            return {}

        if prediction_type == 'gait':
            # 步态类型预测结果
            if 'predictions' not in predictions:
                return {}

            # 统计各类型的数量
            class_counts = {}
            for pred in predictions['predictions']:
                class_name = pred['predicted_class_name']
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1

            # 准备数据
            chart_data = []
            for class_name, count in class_counts.items():
                chart_data.append({
                    'name': class_name,
                    'value': count
                })

            # 创建图表配置
            chart_config = {
                'title': {
                    'text': '步态类型预测结果',
                    'left': 'center'
                },
                'tooltip': {
                    'trigger': 'item',
                    'formatter': '{a} <br/>{b}: {c} ({d}%)'
                },
                'legend': {
                    'orient': 'vertical',
                    'left': 'left',
                    'data': [item['name'] for item in chart_data]
                },
                'series': [
                    {
                        'name': '步态类型',
                        'type': 'pie',
                        'radius': '55%',
                        'center': ['50%', '60%'],
                        'data': chart_data,
                        'emphasis': {
                            'itemStyle': {
                                'shadowBlur': 10,
                                'shadowOffsetX': 0,
                                'shadowColor': 'rgba(0, 0, 0, 0.5)'
                            }
                        }
                    }
                ]
            }

            return chart_config

        elif prediction_type in ['fatigue', 'fall_risk', 'health']:
            # 疲劳/跌倒风险/健康状况预测结果
            if 'predictions' not in predictions:
                return {}

            # 确定指标名称和级别
            if prediction_type == 'fatigue':
                index_name = 'fatigue_index'
                level_name = 'fatigue_level'
                title = '疲劳预测结果'
                color_range = ['#91cc75', '#fac858', '#ee6666']  # 绿-黄-红
            elif prediction_type == 'fall_risk':
                index_name = 'fall_risk'
                level_name = 'risk_level'
                title = '跌倒风险预测结果'
                color_range = ['#91cc75', '#fac858', '#ee6666']  # 绿-黄-红
            else:  # health
                index_name = 'health_index'
                level_name = 'health_status'
                title = '健康状况预测结果'
                color_range = ['#ee6666', '#fac858', '#91cc75']  # 红-黄-绿

            # 统计各级别的数量
            level_counts = {}
            for pred in predictions['predictions']:
                level = pred[level_name]
                if level in level_counts:
                    level_counts[level] += 1
                else:
                    level_counts[level] = 1

            # 准备数据
            chart_data = []
            for level, count in level_counts.items():
                chart_data.append({
                    'name': level,
                    'value': count
                })

            # 创建图表配置
            chart_config = {
                'title': {
                    'text': title,
                    'left': 'center'
                },
                'tooltip': {
                    'trigger': 'item',
                    'formatter': '{a} <br/>{b}: {c} ({d}%)'
                },
                'legend': {
                    'orient': 'vertical',
                    'left': 'left',
                    'data': [item['name'] for item in chart_data]
                },
                'series': [
                    {
                        'name': level_name.replace('_', ' ').title(),
                        'type': 'pie',
                        'radius': '55%',
                        'center': ['50%', '60%'],
                        'data': chart_data,
                        'emphasis': {
                            'itemStyle': {
                                'shadowBlur': 10,
                                'shadowOffsetX': 0,
                                'shadowColor': 'rgba(0, 0, 0, 0.5)'
                            }
                        }
                    }
                ]
            }

            return chart_config

        else:
            return {}
