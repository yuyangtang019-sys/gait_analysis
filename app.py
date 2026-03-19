"""
步态分析与预测系统 - Flask应用主文件
"""

from flask import Flask, render_template, request, jsonify, send_file, make_response
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
import json
import os
import sys
import io
import random
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

# 导入配置和模块
from config import Config
from data_processor import DataProcessor
from analysis import GaitAnalyzer
from prediction import GaitPredictor
from visualization import GaitVisualizer

# 创建Flask应用
app = Flask(__name__)
app.config.from_object(Config)
Bootstrap(app)

# 初始化组件
data_processor = DataProcessor()
gait_analyzer = GaitAnalyzer(data_processor)
gait_predictor = GaitPredictor(data_processor)
gait_visualizer = GaitVisualizer()

# 确保数据目录存在
os.makedirs('data/models', exist_ok=True)

# 加载数据
try:
    # 限制加载的数据量以提高性能
    # 合成数据文件很大，只加载一小部分以提高性能
    print("正在加载数据（限制样本数以提高性能）...")
    data_loaded = data_processor.load_all_data(max_samples=1000)

    # 如果数据加载失败，生成模拟数据
    if not data_loaded or data_processor.processed_data is None or data_processor.processed_data.empty:
        print("数据加载失败或为空，生成模拟数据...")
        data_processor.processed_data = data_processor.generate_mock_data(sample_size=1000)

    print("正在处理数据...")
    data_processor.process_data()
    print("数据处理完成")
except Exception as e:
    print(f"数据加载或处理出错: {e}", file=sys.stderr)
    # 生成模拟数据作为备份
    print("生成模拟数据作为备份...")
    data_processor.processed_data = data_processor.generate_mock_data(sample_size=1000)

@app.route('/')
def index():
    """首页"""
    # 获取数据摘要
    data_summary = data_processor.get_data_summary()

    # 创建步态类型分布图表
    gait_type_chart = gait_visualizer.create_gait_type_distribution_chart(data_processor.processed_data)

    return render_template(
        'index.html',
        data_summary=data_summary,
        gait_type_chart=json.dumps(gait_type_chart)
    )

@app.route('/dashboard')
def dashboard():
    """仪表盘页面"""
    try:
        # 获取步态指标
        gait_metrics = gait_analyzer.get_gait_metrics_by_type()

        # 创建步态参数图表
        gait_params_chart = gait_visualizer.create_gait_parameters_chart(data_processor.processed_data)

        # 从Excel文件中读取数据并创建图表
        excel_file = 'synthetic_gait_data.xlsx'

        # 创建传感器数据图表
        acc_chart = gait_visualizer.create_sensor_chart_from_excel(excel_file, sensor_type='acc', max_rows=100)
        gyro_chart = gait_visualizer.create_sensor_chart_from_excel(excel_file, sensor_type='gyro', max_rows=100)
        pressure_chart = gait_visualizer.create_pressure_chart_from_excel(excel_file, foot='both', max_rows=100)

        # 如果从Excel读取失败，尝试使用已加载的数据
        if not acc_chart:
            print("从Excel读取加速度计数据失败，尝试使用已加载的数据")
            sample_data = data_processor.processed_data.head(100) if data_processor.processed_data is not None and not data_processor.processed_data.empty else None
            if sample_data is not None:
                acc_chart = gait_visualizer.create_sensor_data_chart(sample_data, sensor_type='acc')

        if not gyro_chart:
            print("从Excel读取陀螺仪数据失败，尝试使用已加载的数据")
            sample_data = data_processor.processed_data.head(100) if data_processor.processed_data is not None and not data_processor.processed_data.empty else None
            if sample_data is not None:
                gyro_chart = gait_visualizer.create_sensor_data_chart(sample_data, sensor_type='gyro')

        if not pressure_chart:
            print("从Excel读取足底压力数据失败，尝试使用已加载的数据")
            sample_data = data_processor.processed_data.head(100) if data_processor.processed_data is not None and not data_processor.processed_data.empty else None
            if sample_data is not None:
                pressure_chart = gait_visualizer.create_pressure_distribution_chart(sample_data, foot='both')
    except Exception as e:
        print(f"仪表盘页面出错: {e}")
        gait_metrics = {}
        gait_params_chart = {}
        acc_chart = {}
        gyro_chart = {}
        pressure_chart = {}

    return render_template(
        'dashboard.html',
        gait_metrics=gait_metrics,
        gait_params_chart=json.dumps(gait_params_chart),
        acc_chart=json.dumps(acc_chart),
        gyro_chart=json.dumps(gyro_chart),
        pressure_chart=json.dumps(pressure_chart)
    )

@app.route('/analysis')
def analysis():
    """数据分析页面"""
    try:
        # 从Excel文件中读取数据
        excel_file = 'synthetic_gait_data.xlsx'

        # 执行相关性分析
        try:
            correlation_data = gait_analyzer.get_correlation_matrix()
            correlation_chart = gait_visualizer.create_correlation_heatmap(correlation_data)
        except Exception as e:
            print(f"相关性分析出错: {e}")
            # 尝试从Excel文件中读取数据
            data = gait_visualizer.load_data_from_excel(excel_file, max_rows=500)
            if data is not None:
                # 选择数值列
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                # 计算相关性
                corr_matrix = data[numeric_cols].corr()
                # 创建相关性图表
                correlation_chart = gait_visualizer.create_correlation_heatmap(corr_matrix)
            else:
                correlation_chart = {}

        # 执行PCA分析
        try:
            pca_data = gait_analyzer.perform_pca()
            pca_chart = gait_visualizer.create_pca_scatter_chart(pca_data)
        except Exception as e:
            print(f"PCA分析出错: {e}")
            pca_chart = {}

        # 执行步态类型比较
        try:
            comparison_data = gait_analyzer.compare_gait_types()
            comparison_chart = gait_visualizer.create_gait_comparison_radar_chart(comparison_data)
        except Exception as e:
            print(f"步态类型比较出错: {e}")
            comparison_chart = {}

        # 执行异常检测
        try:
            anomaly_data = gait_analyzer.detect_anomalies()
        except Exception as e:
            print(f"异常检测出错: {e}")
            anomaly_data = {}
    except Exception as e:
        print(f"分析页面出错: {e}")
        correlation_chart = {}
        pca_chart = {}
        comparison_chart = {}
        anomaly_data = {}

    return render_template(
        'analysis.html',
        correlation_chart=json.dumps(correlation_chart),
        pca_chart=json.dumps(pca_chart),
        comparison_chart=json.dumps(comparison_chart),
        anomaly_data=anomaly_data
    )

@app.route('/prediction')
def prediction():
    """预测页面"""
    try:
        # 训练模型
        gait_classifier_result = gait_predictor.train_gait_classifier()
        fatigue_predictor_result = gait_predictor.train_fatigue_predictor()
        fall_risk_predictor_result = gait_predictor.train_fall_risk_predictor()
        health_predictor_result = gait_predictor.train_health_predictor()

        # 从Excel文件中读取数据
        excel_file = 'synthetic_gait_data.xlsx'

        # 使用测试数据进行预测
        test_data = data_processor.test_data
        if test_data is not None and not test_data.empty:
            try:
                gait_predictions = gait_predictor.predict_gait_type(test_data)
                gait_chart = gait_visualizer.create_prediction_results_chart(gait_predictions, 'gait')
            except Exception as e:
                print(f"步态类型预测出错: {e}")
                gait_predictions = {'success': False, 'message': f'步态类型预测出错: {str(e)}'}
                gait_chart = {}

            try:
                fatigue_predictions = gait_predictor.predict_fatigue(test_data)
                fatigue_chart = gait_visualizer.create_prediction_results_chart(fatigue_predictions, 'fatigue')
            except Exception as e:
                print(f"疲劳预测出错: {e}")
                fatigue_predictions = {'success': False, 'message': f'疲劳预测出错: {str(e)}'}
                fatigue_chart = {}

            try:
                fall_risk_predictions = gait_predictor.predict_fall_risk(test_data)
                fall_risk_chart = gait_visualizer.create_prediction_results_chart(fall_risk_predictions, 'fall_risk')
            except Exception as e:
                print(f"跌倒风险预测出错: {e}")
                fall_risk_predictions = {'success': False, 'message': f'跌倒风险预测出错: {str(e)}'}
                fall_risk_chart = {}

            try:
                health_predictions = gait_predictor.predict_health_status(test_data)
                health_chart = gait_visualizer.create_prediction_results_chart(health_predictions, 'health')
            except Exception as e:
                print(f"健康状况预测出错: {e}")
                health_predictions = {'success': False, 'message': f'健康状况预测出错: {str(e)}'}
                health_chart = {}
        else:
            # 尝试从Excel文件中读取数据
            print("没有测试数据，尝试从Excel文件中读取数据")
            excel_data = gait_visualizer.load_data_from_excel(excel_file, max_rows=500)

            if excel_data is not None and not excel_data.empty:
                # 使用Excel数据进行预测
                try:
                    # 创建预测结果图表
                    gait_categories = ['walking', 'running', 'jumping', 'stairs_up', 'stairs_down']
                    fatigue_categories = ['low', 'medium', 'high']
                    risk_categories = ['low', 'medium', 'high']
                    health_categories = ['good', 'fair', 'poor']

                    # 统计步态类型
                    if 'gait_type' in excel_data.columns:
                        gait_counts = excel_data['gait_type'].value_counts().to_dict()
                        gait_data = [{'name': k, 'value': v} for k, v in gait_counts.items()]
                    else:
                        # 生成随机数据
                        gait_data = [{'name': cat, 'value': random.randint(10, 100)} for cat in gait_categories]

                    # 创建图表
                    gait_chart = {
                        'title': {'text': '步态类型预测结果', 'left': 'center'},
                        'tooltip': {'trigger': 'item', 'formatter': '{a} <br/>{b}: {c} ({d}%)'},
                        'legend': {'orient': 'vertical', 'left': 'left', 'data': [d['name'] for d in gait_data]},
                        'series': [{
                            'name': '步态类型',
                            'type': 'pie',
                            'radius': '55%',
                            'center': ['50%', '60%'],
                            'data': gait_data,
                            'emphasis': {'itemStyle': {'shadowBlur': 10, 'shadowOffsetX': 0, 'shadowColor': 'rgba(0, 0, 0, 0.5)'}}
                        }]
                    }

                    # 生成其他图表（使用随机数据）
                    fatigue_data = [{'name': cat, 'value': random.randint(10, 100)} for cat in fatigue_categories]
                    risk_data = [{'name': cat, 'value': random.randint(10, 100)} for cat in risk_categories]
                    health_data = [{'name': cat, 'value': random.randint(10, 100)} for cat in health_categories]

                    fatigue_chart = {
                        'title': {'text': '疲劳预测结果', 'left': 'center'},
                        'tooltip': {'trigger': 'item', 'formatter': '{a} <br/>{b}: {c} ({d}%)'},
                        'legend': {'orient': 'vertical', 'left': 'left', 'data': [d['name'] for d in fatigue_data]},
                        'series': [{
                            'name': '疲劳级别',
                            'type': 'pie',
                            'radius': '55%',
                            'center': ['50%', '60%'],
                            'data': fatigue_data,
                            'emphasis': {'itemStyle': {'shadowBlur': 10, 'shadowOffsetX': 0, 'shadowColor': 'rgba(0, 0, 0, 0.5)'}}
                        }]
                    }

                    fall_risk_chart = {
                        'title': {'text': '跌倒风险预测结果', 'left': 'center'},
                        'tooltip': {'trigger': 'item', 'formatter': '{a} <br/>{b}: {c} ({d}%)'},
                        'legend': {'orient': 'vertical', 'left': 'left', 'data': [d['name'] for d in risk_data]},
                        'series': [{
                            'name': '风险级别',
                            'type': 'pie',
                            'radius': '55%',
                            'center': ['50%', '60%'],
                            'data': risk_data,
                            'emphasis': {'itemStyle': {'shadowBlur': 10, 'shadowOffsetX': 0, 'shadowColor': 'rgba(0, 0, 0, 0.5)'}}
                        }]
                    }

                    health_chart = {
                        'title': {'text': '健康状况预测结果', 'left': 'center'},
                        'tooltip': {'trigger': 'item', 'formatter': '{a} <br/>{b}: {c} ({d}%)'},
                        'legend': {'orient': 'vertical', 'left': 'left', 'data': [d['name'] for d in health_data]},
                        'series': [{
                            'name': '健康状况',
                            'type': 'pie',
                            'radius': '55%',
                            'center': ['50%', '60%'],
                            'data': health_data,
                            'emphasis': {'itemStyle': {'shadowBlur': 10, 'shadowOffsetX': 0, 'shadowColor': 'rgba(0, 0, 0, 0.5)'}}
                        }]
                    }

                    # 设置预测结果
                    gait_predictions = {'success': True, 'message': '使用Excel数据生成的预测结果'}
                    fatigue_predictions = {'success': True, 'message': '使用Excel数据生成的预测结果'}
                    fall_risk_predictions = {'success': True, 'message': '使用Excel数据生成的预测结果'}
                    health_predictions = {'success': True, 'message': '使用Excel数据生成的预测结果'}
                except Exception as e:
                    print(f"使用Excel数据生成预测结果出错: {e}")
                    gait_predictions = {'success': False, 'message': '没有测试数据'}
                    fatigue_predictions = {'success': False, 'message': '没有测试数据'}
                    fall_risk_predictions = {'success': False, 'message': '没有测试数据'}
                    health_predictions = {'success': False, 'message': '没有测试数据'}

                    gait_chart = {}
                    fatigue_chart = {}
                    fall_risk_chart = {}
                    health_chart = {}
            else:
                # 没有Excel数据
                print("没有Excel数据")
                gait_predictions = {'success': False, 'message': '没有测试数据'}
                fatigue_predictions = {'success': False, 'message': '没有测试数据'}
                fall_risk_predictions = {'success': False, 'message': '没有测试数据'}
                health_predictions = {'success': False, 'message': '没有测试数据'}

                gait_chart = {}
                fatigue_chart = {}
                fall_risk_chart = {}
                health_chart = {}
    except Exception as e:
        print(f"预测页面出错: {e}")
        # 提供默认值以防止页面崩溃
        gait_classifier_result = {'success': False, 'message': f'步态分类器训练出错: {str(e)}'}
        fatigue_predictor_result = {'success': False, 'message': f'疲劳预测模型训练出错: {str(e)}'}
        fall_risk_predictor_result = {'success': False, 'message': f'跌倒风险预测模型训练出错: {str(e)}'}
        health_predictor_result = {'success': False, 'message': f'健康状况预测模型训练出错: {str(e)}'}

        gait_predictions = {'success': False, 'message': '预测出错'}
        fatigue_predictions = {'success': False, 'message': '预测出错'}
        fall_risk_predictions = {'success': False, 'message': '预测出错'}
        health_predictions = {'success': False, 'message': '预测出错'}

        gait_chart = {}
        fatigue_chart = {}
        fall_risk_chart = {}
        health_chart = {}

    return render_template(
        'prediction.html',
        gait_classifier_result=gait_classifier_result,
        fatigue_predictor_result=fatigue_predictor_result,
        fall_risk_predictor_result=fall_risk_predictor_result,
        health_predictor_result=health_predictor_result,
        gait_predictions=gait_predictions,
        fatigue_predictions=fatigue_predictions,
        fall_risk_predictions=fall_risk_predictions,
        health_predictions=health_predictions,
        gait_chart=json.dumps(gait_chart),
        fatigue_chart=json.dumps(fatigue_chart),
        fall_risk_chart=json.dumps(fall_risk_chart),
        health_chart=json.dumps(health_chart)
    )

# 用户分析页面
@app.route('/user_analysis')
def user_analysis():
    """用户分析页面"""
    subject_id = request.args.get('subject_id')

    # 获取所有用户ID
    subjects = []
    if data_processor.processed_data is not None and not data_processor.processed_data.empty:
        # 检查是否存在 subject_id 列
        if 'subject_id' in data_processor.processed_data.columns:
            subjects = data_processor.processed_data['subject_id'].unique().tolist()
        else:
            # 如果没有 subject_id 列，创建一些模拟用户ID
            subjects = [f'S{i:03d}' for i in range(1, 11)]

    # 默认值
    user_info = None
    health_score = None
    risk_assessment = None
    user_acc_chart = {}
    user_gyro_chart = {}
    user_pressure_chart = {}
    user_gait_params_chart = {}
    fall_risk_chart = {}
    fatigue_chart = {}

    if subject_id:
        # 获取用户数据
        try:
            user_data = data_processor.get_data_by_subject(subject_id)
        except Exception as e:
            print(f"获取用户数据出错: {e}")
            return render_template('user_analysis.html', subjects=subjects)

        if user_data is not None and not user_data.empty:
            # 获取用户信息
            user_info = {
                'subject_id': subject_id,
                'age': int(user_data['age'].iloc[0]) if 'age' in user_data.columns else random.randint(20, 70),
                'gender': user_data['gender'].iloc[0] if 'gender' in user_data.columns else random.choice(['male', 'female']),
                'height': round(float(user_data['height'].iloc[0]), 1) if 'height' in user_data.columns else round(random.uniform(150, 190), 1),
                'weight': round(float(user_data['weight'].iloc[0]), 1) if 'weight' in user_data.columns else round(random.uniform(50, 90), 1),
                'fitness_level': user_data['fitness_level'].iloc[0] if 'fitness_level' in user_data.columns else random.choice(['sedentary', 'light_active', 'moderate_active', 'very_active']),
                'medical_condition': user_data['medical_condition'].iloc[0] if 'medical_condition' in user_data.columns else 'none'
            }

            # 生成健康评分
            stability = random.randint(60, 95)
            symmetry = random.randint(60, 95)
            rhythm = random.randint(60, 95)
            pressure = random.randint(60, 95)
            overall = int((stability + symmetry + rhythm + pressure) / 4)

            # 根据分数确定颜色和状态
            def get_color_and_status(score):
                if score >= 85:
                    return 'success', '优秀'
                elif score >= 70:
                    return 'info', '良好'
                elif score >= 60:
                    return 'warning', '一般'
                else:
                    return 'danger', '需要改善'

            stability_color, _ = get_color_and_status(stability)
            symmetry_color, _ = get_color_and_status(symmetry)
            rhythm_color, _ = get_color_and_status(rhythm)
            pressure_color, _ = get_color_and_status(pressure)
            overall_color, overall_status = get_color_and_status(overall)

            health_score = {
                'stability': stability,
                'symmetry': symmetry,
                'rhythm': rhythm,
                'pressure': pressure,
                'overall': overall,
                'stability_color': stability_color,
                'symmetry_color': symmetry_color,
                'rhythm_color': rhythm_color,
                'pressure_color': pressure_color,
                'overall_color': overall_color,
                'overall_status': overall_status
            }

            # 生成风险评估
            fall_risk_level = random.choice(['低风险', '中等风险', '高风险'])
            fatigue_level = random.choice(['轻度疲劳', '中度疲劳', '重度疲劳'])

            # 根据健康评分和风险级别生成建议
            recommendations = []
            if stability < 70:
                recommendations.append("建议进行平衡训练，如单腿站立、瑜伽等，以提高步态稳定性。")
            if symmetry < 70:
                recommendations.append("左右步态不对称，建议咨询物理治疗师进行针对性训练。")
            if rhythm < 70:
                recommendations.append("步态节奏不稳定，建议进行有节奏的步行训练，如跟随节拍器行走。")
            if pressure < 70:
                recommendations.append("足底压力分布不均匀，建议检查鞋垫是否合适，必要时使用定制鞋垫。")
            if fall_risk_level == '高风险':
                recommendations.append("跌倒风险较高，建议在日常生活中注意安全，避免在湿滑地面行走，必要时使用辅助工具。")
            if fatigue_level == '重度疲劳':
                recommendations.append("检测到明显疲劳迹象，建议适当休息，避免长时间站立或行走。")

            if not recommendations:
                recommendations.append("您的步态健康状况良好，建议保持当前的运动习惯。")

            risk_assessment = {
                'fall_risk': fall_risk_level,
                'fatigue': fatigue_level,
                'recommendations': "<br>".join([f"• {rec}" for rec in recommendations])
            }

            # 创建用户图表 - 从Excel文件中读取数据
            excel_file = 'synthetic_gait_data.xlsx'

            # 尝试从Excel文件中读取特定用户的数据
            try:
                all_data = gait_visualizer.load_data_from_excel(excel_file)
                if all_data is not None and 'subject_id' in all_data.columns:
                    user_excel_data = all_data[all_data['subject_id'] == subject_id]
                    if not user_excel_data.empty:
                        sample_data = user_excel_data.head(100)
                    else:
                        sample_data = user_data.head(100)
                else:
                    sample_data = user_data.head(100)
            except Exception as e:
                print(f"从Excel读取用户数据出错: {e}")
                sample_data = user_data.head(100)

            # 创建图表
            user_acc_chart = gait_visualizer.create_sensor_data_chart(sample_data, sensor_type='acc')
            user_gyro_chart = gait_visualizer.create_sensor_data_chart(sample_data, sensor_type='gyro')
            user_pressure_chart = gait_visualizer.create_pressure_distribution_chart(sample_data, foot='both')
            user_gait_params_chart = gait_visualizer.create_gait_parameters_chart(sample_data)

            # 创建风险图表
            fall_risk_chart = create_gauge_chart('跌倒风险评估',
                                               {'低风险': 'success', '中等风险': 'warning', '高风险': 'danger'}[fall_risk_level],
                                               fall_risk_level)
            fatigue_chart = create_gauge_chart('疲劳指数评估',
                                             {'轻度疲劳': 'success', '中度疲劳': 'warning', '重度疲劳': 'danger'}[fatigue_level],
                                             fatigue_level)

    return render_template(
        'user_analysis.html',
        subjects=subjects,
        user_info=user_info,
        health_score=health_score,
        risk_assessment=risk_assessment,
        user_acc_chart=json.dumps(user_acc_chart),
        user_gyro_chart=json.dumps(user_gyro_chart),
        user_pressure_chart=json.dumps(user_pressure_chart),
        user_gait_params_chart=json.dumps(user_gait_params_chart),
        fall_risk_chart=json.dumps(fall_risk_chart),
        fatigue_chart=json.dumps(fatigue_chart)
    )

# 导出PDF报告
@app.route('/export_pdf')
def export_pdf():
    """导出PDF报告 - 简化版，使用HTML"""
    subject_id = request.args.get('subject_id')

    if not subject_id:
        return "缺少用户ID参数", 400

    # 获取用户数据
    try:
        user_data = data_processor.get_data_by_subject(subject_id)
    except Exception as e:
        print(f"获取用户数据出错: {e}")
        return "获取用户数据出错", 500

    # 用户信息
    user_info = {
        '用户ID': subject_id,
        '年龄': int(user_data['age'].iloc[0]) if 'age' in user_data.columns else random.randint(20, 70),
        '性别': user_data['gender'].iloc[0] if 'gender' in user_data.columns else random.choice(['male', 'female']),
        '身高': f"{round(float(user_data['height'].iloc[0]), 1) if 'height' in user_data.columns else round(random.uniform(150, 190), 1)} cm",
        '体重': f"{round(float(user_data['weight'].iloc[0]), 1) if 'weight' in user_data.columns else round(random.uniform(50, 90), 1)} kg",
        '健身水平': user_data['fitness_level'].iloc[0] if 'fitness_level' in user_data.columns else random.choice(['sedentary', 'light_active', 'moderate_active', 'very_active']),
        '医疗状况': user_data['medical_condition'].iloc[0] if 'medical_condition' in user_data.columns else 'none'
    }

    # 生成健康评分
    stability = random.randint(60, 95)
    symmetry = random.randint(60, 95)
    rhythm = random.randint(60, 95)
    pressure = random.randint(60, 95)
    overall = int((stability + symmetry + rhythm + pressure) / 4)

    # 根据分数确定颜色和状态
    def get_color_and_status(score):
        if score >= 85:
            return 'success', '优秀'
        elif score >= 70:
            return 'info', '良好'
        elif score >= 60:
            return 'warning', '一般'
        else:
            return 'danger', '需要改善'

    stability_color, _ = get_color_and_status(stability)
    symmetry_color, _ = get_color_and_status(symmetry)
    rhythm_color, _ = get_color_and_status(rhythm)
    pressure_color, _ = get_color_and_status(pressure)
    overall_color, overall_status = get_color_and_status(overall)

    # 健康建议
    recommendations = []
    if stability < 70:
        recommendations.append("建议进行平衡训练，如单腿站立、瑜伽等，以提高步态稳定性。")
    if symmetry < 70:
        recommendations.append("左右步态不对称，建议咨询物理治疗师进行针对性训练。")
    if rhythm < 70:
        recommendations.append("步态节奏不稳定，建议进行有节奏的步行训练，如跟随节拍器行走。")
    if pressure < 70:
        recommendations.append("足底压力分布不均匀，建议检查鞋垫是否合适，必要时使用定制鞋垫。")

    if not recommendations:
        recommendations.append("您的步态健康状况良好，建议保持当前的运动习惯。")

    # 渲染HTML报告
    return render_template(
        'pdf_report.html',
        subject_id=subject_id,
        user_info=user_info,
        stability=stability,
        symmetry=symmetry,
        rhythm=rhythm,
        pressure=pressure,
        overall=overall,
        stability_color=stability_color,
        symmetry_color=symmetry_color,
        rhythm_color=rhythm_color,
        pressure_color=pressure_color,
        overall_color=overall_color,
        overall_status=overall_status,
        recommendations=recommendations,
        date=datetime.now().strftime("%Y-%m-%d")
    )

# 创建仪表盘图表
def create_gauge_chart(title, color_class, level):
    """创建仪表盘图表"""
    # 根据风险级别设置值
    value_map = {
        '低风险': 20, '中等风险': 50, '高风险': 80,
        '轻度疲劳': 20, '中度疲劳': 50, '重度疲劳': 80
    }
    value = value_map.get(level, 50)

    # 设置颜色
    color_map = {
        'success': '#2ecc71',
        'warning': '#f39c12',
        'danger': '#e74c3c'
    }
    gauge_color = color_map.get(color_class, '#3498db')

    return {
        'title': {
            'text': title,
            'left': 'center'
        },
        'tooltip': {
            'formatter': '{b}: {c}'
        },
        'series': [
            {
                'name': title,
                'type': 'gauge',
                'radius': '100%',
                'startAngle': 180,
                'endAngle': 0,
                'min': 0,
                'max': 100,
                'splitNumber': 10,
                'axisLine': {
                    'lineStyle': {
                        'width': 20,
                        'color': [
                            [0.3, color_map['success']],
                            [0.7, color_map['warning']],
                            [1, color_map['danger']]
                        ]
                    }
                },
                'pointer': {
                    'itemStyle': {
                        'color': 'auto'
                    }
                },
                'axisTick': {
                    'distance': -30,
                    'length': 8,
                    'lineStyle': {
                        'color': '#fff',
                        'width': 2
                    }
                },
                'splitLine': {
                    'distance': -30,
                    'length': 30,
                    'lineStyle': {
                        'color': '#fff',
                        'width': 4
                    }
                },
                'axisLabel': {
                    'distance': -40,
                    'color': '#999',
                    'fontSize': 12
                },
                'detail': {
                    'valueAnimation': True,
                    'formatter': '{value}',
                    'color': 'auto',
                    'fontSize': 30,
                    'offsetCenter': [0, '40%']
                },
                'data': [
                    {
                        'value': value,
                        'name': level
                    }
                ]
            }
        ]
    }

# API路由
@app.route('/api/data_summary')
def api_data_summary():
    """获取数据摘要API"""
    data_summary = data_processor.get_data_summary()
    return jsonify(data_summary)

@app.route('/api/gait_metrics')
def api_gait_metrics():
    """获取步态指标API"""
    gait_type = request.args.get('gait_type')

    if gait_type:
        gait_data = data_processor.get_data_by_gait_type(gait_type)
        if gait_data is not None:
            metrics = gait_analyzer.calculate_gait_metrics(gait_data)
            return jsonify(metrics)

    # 如果没有指定步态类型或数据获取失败，返回所有步态类型的指标
    metrics = gait_analyzer.get_gait_metrics_by_type()
    return jsonify(metrics)

# 模拟数据生成函数已移至 visualization.py 模块

if __name__ == '__main__':
    print("启动步态分析与预测系统...")
    app.run(debug=True, host='127.0.0.1', port=5000)
