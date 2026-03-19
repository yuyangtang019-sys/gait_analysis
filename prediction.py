"""
预测模块
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import json
from config import Config

class GaitPredictor:
    def __init__(self, data_processor):
        """
        初始化步态预测器

        Args:
            data_processor: 数据处理器实例
        """
        self.data_processor = data_processor
        self.gait_classifier = None
        self.fatigue_predictor = None
        self.fall_risk_predictor = None
        self.health_predictor = None

    def prepare_data(self):
        """准备训练数据"""
        # 修复 DataFrame 的 truth value 错误
        train_data_exists = self.data_processor.train_data is not None and not self.data_processor.train_data.empty
        test_data_exists = self.data_processor.test_data is not None and not self.data_processor.test_data.empty

        if not train_data_exists or not test_data_exists:
            print("训练数据或测试数据不存在，尝试分割数据...")
            if not self.data_processor.split_train_test():
                print("数据分割失败")
                return False
            else:
                print("数据分割成功")

        return True

    def train_gait_classifier(self):
        """
        训练步态类型分类器

        Returns:
            dict: 训练结果
        """
        if not self.prepare_data():
            return {'success': False, 'message': '数据准备失败'}

        # 选择特征和目标
        features = Config.SENSOR_FEATURES + Config.GAIT_FEATURES
        valid_features = [f for f in features if f in self.data_processor.train_data.columns]

        if not valid_features:
            return {'success': False, 'message': '没有有效特征'}

        if 'gait_type' not in self.data_processor.train_data.columns:
            return {'success': False, 'message': '数据中没有目标变量 gait_type'}

        # 准备训练数据
        X_train = self.data_processor.train_data[valid_features]
        y_train = self.data_processor.train_data['gait_type']

        X_test = self.data_processor.test_data[valid_features]
        y_test = self.data_processor.test_data['gait_type']

        # 训练模型
        self.gait_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=Config.RANDOM_STATE
        )

        self.gait_classifier.fit(X_train, y_train)

        # 评估模型
        y_pred = self.gait_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # 交叉验证
        cv_scores = cross_val_score(
            self.gait_classifier,
            X_train, y_train,
            cv=5,
            scoring='accuracy'
        )

        # 特征重要性
        feature_importance = dict(zip(valid_features, self.gait_classifier.feature_importances_))

        # 分类报告
        report = classification_report(y_test, y_pred, output_dict=True)

        # 保存模型
        joblib.dump(self.gait_classifier, 'data/models/gait_classifier.pkl')

        return {
            'success': True,
            'accuracy': accuracy,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'feature_importance': feature_importance,
            'classification_report': report
        }

    def predict_gait_type(self, input_data):
        """
        预测步态类型

        Args:
            input_data: 输入数据

        Returns:
            dict: 预测结果
        """
        if self.gait_classifier is None:
            try:
                self.gait_classifier = joblib.load('data/models/gait_classifier.pkl')
            except:
                result = self.train_gait_classifier()
                if not result['success']:
                    return {'success': False, 'message': '模型训练失败'}

        # 选择特征
        features = Config.SENSOR_FEATURES + Config.GAIT_FEATURES
        valid_features = [f for f in features if f in input_data.columns]

        if not valid_features:
            return {'success': False, 'message': '输入数据没有有效特征'}

        # 预测
        predictions = self.gait_classifier.predict(input_data[valid_features])
        probabilities = self.gait_classifier.predict_proba(input_data[valid_features])

        # 获取类别名称
        class_names = self.gait_classifier.classes_

        # 准备结果
        results = []
        for i, pred in enumerate(predictions):
            probs = probabilities[i]
            class_probs = {class_names[j]: float(prob) for j, prob in enumerate(probs)}

            results.append({
                'predicted_class': pred,
                'predicted_class_name': Config.GAIT_TYPES.get(pred, pred),
                'probabilities': class_probs
            })

        return {
            'success': True,
            'predictions': results
        }

    def train_fatigue_predictor(self):
        """
        训练疲劳预测模型

        Returns:
            dict: 训练结果
        """
        if not self.prepare_data():
            return {'success': False, 'message': '数据准备失败'}

        # 创建疲劳指标（模拟）
        # 在实际应用中，这应该是从数据中获取的真实疲劳指标
        # 这里我们基于步态参数的变化来模拟疲劳程度
        train_data = self.data_processor.train_data.copy()
        test_data = self.data_processor.test_data.copy()

        # 模拟疲劳指标：基于步频和步长的变化
        if 'cadence' in train_data.columns and 'stride_length' in train_data.columns:
            # 标准化步频和步长
            cadence_mean = train_data['cadence'].mean()
            cadence_std = train_data['cadence'].std()
            stride_mean = train_data['stride_length'].mean()
            stride_std = train_data['stride_length'].std()

            # 计算疲劳指标：步频下降和步长缩短表示疲劳增加
            train_data['fatigue_index'] = (
                (cadence_mean - train_data['cadence']) / cadence_std +
                (stride_mean - train_data['stride_length']) / stride_std
            ) / 2

            test_data['fatigue_index'] = (
                (cadence_mean - test_data['cadence']) / cadence_std +
                (stride_mean - test_data['stride_length']) / stride_std
            ) / 2

            # 将指标归一化到0-100范围
            min_fatigue = min(train_data['fatigue_index'].min(), test_data['fatigue_index'].min())
            max_fatigue = max(train_data['fatigue_index'].max(), test_data['fatigue_index'].max())

            train_data['fatigue_index'] = ((train_data['fatigue_index'] - min_fatigue) /
                                          (max_fatigue - min_fatigue)) * 100

            test_data['fatigue_index'] = ((test_data['fatigue_index'] - min_fatigue) /
                                         (max_fatigue - min_fatigue)) * 100

            # 选择特征和目标
            features = Config.SENSOR_FEATURES
            valid_features = [f for f in features if f in train_data.columns]

            if not valid_features:
                return {'success': False, 'message': '没有有效特征'}

            # 准备训练数据
            X_train = train_data[valid_features]
            y_train = train_data['fatigue_index']

            X_test = test_data[valid_features]
            y_test = test_data['fatigue_index']

            # 训练模型
            self.fatigue_predictor = GradientBoostingRegressor(
                n_estimators=100,
                random_state=Config.RANDOM_STATE
            )

            self.fatigue_predictor.fit(X_train, y_train)

            # 评估模型
            y_pred = self.fatigue_predictor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # 特征重要性
            feature_importance = dict(zip(valid_features, self.fatigue_predictor.feature_importances_))

            # 保存模型
            joblib.dump(self.fatigue_predictor, 'data/models/fatigue_predictor.pkl')

            return {
                'success': True,
                'mse': mse,
                'r2': r2,
                'feature_importance': feature_importance
            }
        else:
            return {'success': False, 'message': '数据中没有必要的特征 (cadence, stride_length)'}

    def predict_fatigue(self, input_data):
        """
        预测疲劳程度

        Args:
            input_data: 输入数据

        Returns:
            dict: 预测结果
        """
        if self.fatigue_predictor is None:
            try:
                self.fatigue_predictor = joblib.load('data/models/fatigue_predictor.pkl')
            except:
                result = self.train_fatigue_predictor()
                if not result['success']:
                    return {'success': False, 'message': '模型训练失败'}

        # 选择特征
        features = Config.SENSOR_FEATURES
        valid_features = [f for f in features if f in input_data.columns]

        if not valid_features:
            return {'success': False, 'message': '输入数据没有有效特征'}

        # 预测
        predictions = self.fatigue_predictor.predict(input_data[valid_features])

        # 准备结果
        results = []
        for i, pred in enumerate(predictions):
            fatigue_level = 'Low'
            if pred > 70:
                fatigue_level = 'High'
            elif pred > 30:
                fatigue_level = 'Medium'

            results.append({
                'fatigue_index': float(pred),
                'fatigue_level': fatigue_level
            })

        return {
            'success': True,
            'predictions': results
        }

    def train_fall_risk_predictor(self):
        """
        训练跌倒风险预测模型

        Returns:
            dict: 训练结果
        """
        if not self.prepare_data():
            return {'success': False, 'message': '数据准备失败'}

        # 创建跌倒风险指标（模拟）
        # 在实际应用中，这应该是从数据中获取的真实跌倒风险指标
        train_data = self.data_processor.train_data.copy()
        test_data = self.data_processor.test_data.copy()

        # 模拟跌倒风险指标：基于步态对称性和传感器数据的波动
        if 'symmetry' in train_data.columns:
            # 计算传感器数据的波动性
            sensor_cols = [col for col in Config.SENSOR_FEATURES if col in train_data.columns]

            if sensor_cols:
                # 计算传感器数据的标准差作为波动性指标
                train_data['sensor_variability'] = train_data[sensor_cols].std(axis=1)
                test_data['sensor_variability'] = test_data[sensor_cols].std(axis=1)

                # 计算跌倒风险指标：对称性越低，传感器波动越大，风险越高
                train_data['fall_risk'] = (
                    (1 - train_data['symmetry']) * 0.7 +
                    (train_data['sensor_variability'] / train_data['sensor_variability'].max()) * 0.3
                ) * 100

                test_data['fall_risk'] = (
                    (1 - test_data['symmetry']) * 0.7 +
                    (test_data['sensor_variability'] / test_data['sensor_variability'].max()) * 0.3
                ) * 100

                # 选择特征和目标
                features = Config.GAIT_FEATURES + ['sensor_variability']
                valid_features = [f for f in features if f in train_data.columns]

                if not valid_features:
                    return {'success': False, 'message': '没有有效特征'}

                # 准备训练数据
                X_train = train_data[valid_features]
                y_train = train_data['fall_risk']

                X_test = test_data[valid_features]
                y_test = test_data['fall_risk']

                # 训练模型
                self.fall_risk_predictor = GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=Config.RANDOM_STATE
                )

                self.fall_risk_predictor.fit(X_train, y_train)

                # 评估模型
                y_pred = self.fall_risk_predictor.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # 特征重要性
                feature_importance = dict(zip(valid_features, self.fall_risk_predictor.feature_importances_))

                # 保存模型
                joblib.dump(self.fall_risk_predictor, 'data/models/fall_risk_predictor.pkl')

                return {
                    'success': True,
                    'mse': mse,
                    'r2': r2,
                    'feature_importance': feature_importance
                }
            else:
                return {'success': False, 'message': '数据中没有传感器特征'}
        else:
            return {'success': False, 'message': '数据中没有必要的特征 (symmetry)'}

    def predict_fall_risk(self, input_data):
        """
        预测跌倒风险

        Args:
            input_data: 输入数据

        Returns:
            dict: 预测结果
        """
        if self.fall_risk_predictor is None:
            try:
                self.fall_risk_predictor = joblib.load('data/models/fall_risk_predictor.pkl')
            except:
                result = self.train_fall_risk_predictor()
                if not result['success']:
                    return {'success': False, 'message': '模型训练失败'}

        # 计算传感器数据的波动性
        input_data = input_data.copy()
        sensor_cols = [col for col in Config.SENSOR_FEATURES if col in input_data.columns]

        if sensor_cols:
            input_data['sensor_variability'] = input_data[sensor_cols].std(axis=1)

            # 选择特征
            features = Config.GAIT_FEATURES + ['sensor_variability']
            valid_features = [f for f in features if f in input_data.columns]

            if not valid_features:
                return {'success': False, 'message': '输入数据没有有效特征'}

            # 预测
            predictions = self.fall_risk_predictor.predict(input_data[valid_features])

            # 准备结果
            results = []
            for i, pred in enumerate(predictions):
                risk_level = 'Low'
                if pred > 70:
                    risk_level = 'High'
                elif pred > 30:
                    risk_level = 'Medium'

                results.append({
                    'fall_risk': float(pred),
                    'risk_level': risk_level
                })

            return {
                'success': True,
                'predictions': results
            }
        else:
            return {'success': False, 'message': '输入数据没有传感器特征'}

    def train_health_predictor(self):
        """
        训练健康状况预测模型

        Returns:
            dict: 训练结果
        """
        if not self.prepare_data():
            return {'success': False, 'message': '数据准备失败'}

        # 创建健康指数（模拟）
        # 在实际应用中，这应该是从数据中获取的真实健康指标
        train_data = self.data_processor.train_data.copy()
        test_data = self.data_processor.test_data.copy()

        # 模拟健康指数：基于步态参数和人口统计学特征
        if all(f in train_data.columns for f in ['cadence', 'stride_length', 'symmetry']):
            # 如果有医疗状况信息，使用它来调整健康指数
            if 'medical_condition' in train_data.columns:
                # 创建医疗状况的数值映射
                condition_map = {
                    'none': 1.0,
                    'arthritis': 0.7,
                    'parkinsons': 0.5,
                    'stroke_recovery': 0.6,
                    'multiple_sclerosis': 0.4,
                    'diabetes': 0.8
                }

                train_data['condition_factor'] = train_data['medical_condition'].map(
                    lambda x: condition_map.get(x, 0.9)
                )

                test_data['condition_factor'] = test_data['medical_condition'].map(
                    lambda x: condition_map.get(x, 0.9)
                )
            else:
                train_data['condition_factor'] = 1.0
                test_data['condition_factor'] = 1.0

            # 标准化步态参数
            for col in ['cadence', 'stride_length', 'symmetry']:
                col_mean = train_data[col].mean()
                col_std = train_data[col].std()

                train_data[f'{col}_norm'] = (train_data[col] - col_mean) / col_std
                test_data[f'{col}_norm'] = (test_data[col] - col_mean) / col_std

            # 计算健康指数
            train_data['health_index'] = (
                (train_data['cadence_norm'] + 3) / 6 * 0.3 +
                (train_data['stride_length_norm'] + 3) / 6 * 0.3 +
                train_data['symmetry'] * 0.4
            ) * train_data['condition_factor'] * 100

            test_data['health_index'] = (
                (test_data['cadence_norm'] + 3) / 6 * 0.3 +
                (test_data['stride_length_norm'] + 3) / 6 * 0.3 +
                test_data['symmetry'] * 0.4
            ) * test_data['condition_factor'] * 100

            # 选择特征和目标
            features = Config.GAIT_FEATURES + Config.SENSOR_FEATURES
            if 'age' in train_data.columns:
                features.append('age')

            valid_features = [f for f in features if f in train_data.columns]

            if not valid_features:
                return {'success': False, 'message': '没有有效特征'}

            # 准备训练数据
            X_train = train_data[valid_features]
            y_train = train_data['health_index']

            X_test = test_data[valid_features]
            y_test = test_data['health_index']

            # 训练模型
            self.health_predictor = GradientBoostingRegressor(
                n_estimators=100,
                random_state=Config.RANDOM_STATE
            )

            self.health_predictor.fit(X_train, y_train)

            # 评估模型
            y_pred = self.health_predictor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # 特征重要性
            feature_importance = dict(zip(valid_features, self.health_predictor.feature_importances_))

            # 保存模型
            joblib.dump(self.health_predictor, 'data/models/health_predictor.pkl')

            return {
                'success': True,
                'mse': mse,
                'r2': r2,
                'feature_importance': feature_importance
            }
        else:
            return {'success': False, 'message': '数据中没有必要的特征 (cadence, stride_length, symmetry)'}

    def predict_health_status(self, input_data):
        """
        预测健康状况

        Args:
            input_data: 输入数据

        Returns:
            dict: 预测结果
        """
        if self.health_predictor is None:
            try:
                self.health_predictor = joblib.load('data/models/health_predictor.pkl')
            except:
                result = self.train_health_predictor()
                if not result['success']:
                    return {'success': False, 'message': '模型训练失败'}

        # 选择特征
        features = Config.GAIT_FEATURES + Config.SENSOR_FEATURES
        if 'age' in input_data.columns:
            features.append('age')

        valid_features = [f for f in features if f in input_data.columns]

        if not valid_features:
            return {'success': False, 'message': '输入数据没有有效特征'}

        # 预测
        predictions = self.health_predictor.predict(input_data[valid_features])

        # 准备结果
        results = []
        for i, pred in enumerate(predictions):
            health_status = 'Good'
            if pred < 30:
                health_status = 'Poor'
            elif pred < 70:
                health_status = 'Fair'

            results.append({
                'health_index': float(pred),
                'health_status': health_status
            })

        return {
            'success': True,
            'predictions': results
        }
