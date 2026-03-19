"""
数据分析模块
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import json
from config import Config
from utils import calculate_gait_metrics

class GaitAnalyzer:
    def __init__(self, data_processor):
        """
        初始化步态分析器
        
        Args:
            data_processor: 数据处理器实例
        """
        self.data_processor = data_processor
        self.data = None
        self.pca_model = None
        self.kmeans_model = Noneanalysis.py
        self.anomaly_detector = None
    
    def prepare_data(self):
        """准备分析数据"""
        if self.data_processor.processed_data is None:
            if not self.data_processor.process_data():
                return False
        
        self.data = self.data_processor.processed_data
        return True
    
    def get_correlation_matrix(self, features=None):
        """
        计算相关性矩阵
        
        Args:
            features: 要计算相关性的特征列表，默认为None（使用所有特征）
            
        Returns:
            dict: 相关性矩阵数据
        """
        if self.data is None:
            if not self.prepare_data():
                return {}
        
        if features is None:
            features = Config.SENSOR_FEATURES + Config.GAIT_FEATURES
        
        # 筛选有效特征
        valid_features = [f for f in features if f in self.data.columns]
        
        if not valid_features:
            return {}
        
        # 计算相关性矩阵
        corr_matrix = self.data[valid_features].corr()
        
        # 转换为前端可用的格式
        result = {
            'features': valid_features,
            'data': []
        }
        
        for i, feature1 in enumerate(valid_features):
            for j, feature2 in enumerate(valid_features):
                result['data'].append([i, j, round(corr_matrix.loc[feature1, feature2], 2)])
        
        return result
    
    def perform_pca(self, n_components=2):
        """
        执行PCA降维
        
        Args:
            n_components: 主成分数量
            
        Returns:
            dict: PCA结果
        """
        if self.data is None:
            if not self.prepare_data():
                return {}
        
        # 选择特征
        features = Config.SENSOR_FEATURES + Config.GAIT_FEATURES
        valid_features = [f for f in features if f in self.data.columns]
        
        if not valid_features:
            return {}
        
        # 执行PCA
        self.pca_model = PCA(n_components=n_components)
        pca_result = self.pca_model.fit_transform(self.data[valid_features])
        
        # 准备结果
        result = {
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_.tolist(),
            'components': []
        }
        
        # 如果有步态类型，按类型分组
        if 'gait_type' in self.data.columns:
            for gait_type in self.data['gait_type'].unique():
                mask = self.data['gait_type'] == gait_type
                gait_data = pca_result[mask]
                
                if len(gait_data) > 0:
                    result['components'].append({
                        'name': Config.GAIT_TYPES.get(gait_type, gait_type),
                        'data': gait_data.tolist()
                    })
        else:
            # 不分组
            result['components'].append({
                'name': 'All Data',
                'data': pca_result.tolist()
            })
        
        return result
    
    def perform_clustering(self, n_clusters=3):
        """
        执行聚类分析
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            dict: 聚类结果
        """
        if self.data is None:
            if not self.prepare_data():
                return {}
        
        # 选择特征
        features = Config.SENSOR_FEATURES + Config.GAIT_FEATURES
        valid_features = [f for f in features if f in self.data.columns]
        
        if not valid_features:
            return {}
        
        # 执行聚类
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=Config.RANDOM_STATE)
        clusters = self.kmeans_model.fit_predict(self.data[valid_features])
        
        # 添加聚类标签
        data_with_clusters = self.data.copy()
        data_with_clusters['cluster'] = clusters
        
        # 准备结果
        result = {
            'cluster_centers': self.kmeans_model.cluster_centers_.tolist(),
            'cluster_distribution': data_with_clusters['cluster'].value_counts().to_dict(),
            'clusters': []
        }
        
        # 按聚类分组
        for cluster_id in range(n_clusters):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
            
            # 计算每个聚类的特征均值
            cluster_means = {}
            for feature in valid_features:
                cluster_means[feature] = cluster_data[feature].mean()
            
            # 如果有步态类型，计算每个聚类中的步态类型分布
            gait_distribution = {}
            if 'gait_type' in cluster_data.columns:
                gait_counts = cluster_data['gait_type'].value_counts()
                for gait_type, count in gait_counts.items():
                    gait_distribution[Config.GAIT_TYPES.get(gait_type, gait_type)] = count
            
            result['clusters'].append({
                'id': cluster_id,
                'size': len(cluster_data),
                'feature_means': cluster_means,
                'gait_distribution': gait_distribution
            })
        
        return result
    
    def detect_anomalies(self, contamination=0.05):
        """
        检测异常步态
        
        Args:
            contamination: 预期的异常比例
            
        Returns:
            dict: 异常检测结果
        """
        if self.data is None:
            if not self.prepare_data():
                return {}
        
        # 选择特征
        features = Config.SENSOR_FEATURES + Config.GAIT_FEATURES
        valid_features = [f for f in features if f in self.data.columns]
        
        if not valid_features:
            return {}
        
        # 执行异常检测
        self.anomaly_detector = IsolationForest(
            contamination=contamination,
            random_state=Config.RANDOM_STATE
        )
        
        # 预测异常 (-1表示异常，1表示正常)
        anomaly_labels = self.anomaly_detector.fit_predict(self.data[valid_features])
        
        # 添加异常标签
        data_with_anomalies = self.data.copy()
        data_with_anomalies['is_anomaly'] = anomaly_labels == -1
        
        # 准备结果
        anomalies = data_with_anomalies[data_with_anomalies['is_anomaly']]
        normal = data_with_anomalies[~data_with_anomalies['is_anomaly']]
        
        result = {
            'anomaly_count': len(anomalies),
            'normal_count': len(normal),
            'anomaly_percentage': len(anomalies) / len(data_with_anomalies) * 100,
        }
        
        # 计算异常和正常数据的特征统计
        anomaly_stats = {}
        normal_stats = {}
        
        for feature in valid_features:
            anomaly_stats[feature] = {
                'mean': anomalies[feature].mean(),
                'std': anomalies[feature].std(),
                'min': anomalies[feature].min(),
                'max': anomalies[feature].max()
            }
            
            normal_stats[feature] = {
                'mean': normal[feature].mean(),
                'std': normal[feature].std(),
                'min': normal[feature].min(),
                'max': normal[feature].max()
            }
        
        result['anomaly_stats'] = anomaly_stats
        result['normal_stats'] = normal_stats
        
        # 如果有步态类型，计算每种步态类型中的异常比例
        if 'gait_type' in self.data.columns:
            gait_anomaly_rates = {}
            
            for gait_type in self.data['gait_type'].unique():
                gait_data = data_with_anomalies[data_with_anomalies['gait_type'] == gait_type]
                if len(gait_data) > 0:
                    anomaly_rate = gait_data['is_anomaly'].mean() * 100
                    gait_anomaly_rates[Config.GAIT_TYPES.get(gait_type, gait_type)] = anomaly_rate
            
            result['gait_anomaly_rates'] = gait_anomaly_rates
        
        return result
    
    def compare_gait_types(self):
        """
        比较不同步态类型的特征
        
        Returns:
            dict: 步态类型比较结果
        """
        if self.data is None:
            if not self.prepare_data():
                return {}
        
        if 'gait_type' not in self.data.columns:
            return {}
        
        # 选择特征
        features = Config.GAIT_FEATURES
        valid_features = [f for f in features if f in self.data.columns]
        
        if not valid_features:
            return {}
        
        # 按步态类型分组
        result = {
            'features': valid_features,
            'gait_types': [],
            'data': []
        }
        
        for gait_type in self.data['gait_type'].unique():
            gait_data = self.data[self.data['gait_type'] == gait_type]
            
            # 计算每种步态类型的特征均值
            gait_means = {}
            for feature in valid_features:
                gait_means[feature] = gait_data[feature].mean()
            
            result['gait_types'].append(Config.GAIT_TYPES.get(gait_type, gait_type))
            result['data'].append(gait_means)
        
        return result
    
    def analyze_demographic_impact(self, demographic_feature):
        """
        分析人口统计学特征对步态的影响
        
        Args:
            demographic_feature: 人口统计学特征 (如 'age', 'gender')
            
        Returns:
            dict: 分析结果
        """
        if self.data is None:
            if not self.prepare_data():
                return {}
        
        if demographic_feature not in self.data.columns:
            return {}
        
        # 选择步态特征
        gait_features = Config.GAIT_FEATURES
        valid_features = [f for f in gait_features if f in self.data.columns]
        
        if not valid_features:
            return {}
        
        # 按人口统计学特征分组
        groups = []
        
        # 对于连续变量（如年龄），进行分箱
        if demographic_feature == 'age' and 'age' in self.data.columns:
            # 创建年龄组
            self.data['age_group'] = pd.cut(
                self.data['age'],
                bins=[0, 30, 50, 70, 100],
                labels=['<30', '30-50', '50-70', '>70']
            )
            demographic_values = self.data['age_group'].unique()
            group_col = 'age_group'
        else:
            demographic_values = self.data[demographic_feature].unique()
            group_col = demographic_feature
        
        # 计算每个组的步态特征均值
        for value in demographic_values:
            if pd.isna(value):
                continue
                
            group_data = self.data[self.data[group_col] == value]
            
            if len(group_data) == 0:
                continue
            
            group_means = {}
            for feature in valid_features:
                group_means[feature] = group_data[feature].mean()
            
            groups.append({
                'name': str(value),
                'count': len(group_data),
                'feature_means': group_means
            })
        
        result = {
            'demographic_feature': demographic_feature,
            'features': valid_features,
            'groups': groups
        }
        
        return result
    
    def get_gait_metrics_by_type(self):
        """
        获取各步态类型的指标
        
        Returns:
            dict: 各步态类型的指标
        """
        if self.data is None:
            if not self.prepare_data():
                return {}
        
        if 'gait_type' not in self.data.columns:
            return {}
        
        result = {}
        
        for gait_type in self.data['gait_type'].unique():
            gait_data = self.data[self.data['gait_type'] == gait_type]
            metrics = calculate_gait_metrics(gait_data)
            result[Config.GAIT_TYPES.get(gait_type, gait_type)] = metrics
        
        return result
