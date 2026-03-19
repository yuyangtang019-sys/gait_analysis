# 步态分析与预测系统

## 项目概述

步态分析与预测系统是一个基于Python、Flask和ECharts的Web应用程序，用于分析和可视化步态数据，并提供预测功能。该系统可以处理来自可穿戴传感器的步态数据，进行数据分析、可视化和预测，帮助用户了解步态健康状况。

![系统首页](static/img/screenshot_dashboard.png)

## 功能特点

- **数据可视化**：使用ECharts创建交互式图表，展示步态数据
- **数据分析**：提供相关性分析、PCA降维分析、步态类型比较等功能
- **预测功能**：预测步态类型、疲劳程度、跌倒风险和健康状况
- **用户分析**：针对特定用户进行步态健康评估和风险分析
- **PDF导出**：生成用户步态健康报告，可以导出为PDF格式
- **响应式设计**：适配不同设备屏幕大小的用户界面

## 技术栈

- **后端**：Python、Flask
- **前端**：HTML、CSS、JavaScript、Bootstrap 4
- **数据处理**：Pandas、NumPy、Scikit-learn
- **数据可视化**：ECharts、Matplotlib
- **机器学习**：Scikit-learn

## 安装说明

### 环境要求

- Python 3.8+
- pip (Python包管理器)

### 安装步骤

1. 克隆或下载项目代码

2. 创建并激活虚拟环境（可选但推荐）
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. 安装依赖包
   ```bash
   pip install -r requirements.txt
   ```

4. 生成合成数据（如果没有真实数据）
   ```bash
   python generate_synthetic_data.py
   ```

5. 启动应用程序
   ```bash
   python app.py
   ```

6. 在浏览器中访问应用程序
   ```
   http://127.0.0.1:5000/
   ```

## 项目结构

```
步态分析与预测系统/
├── app.py                  # 主应用程序文件
├── config.py               # 配置文件
├── data_processor.py       # 数据处理模块
├── analysis.py             # 数据分析模块
├── prediction.py           # 预测模块
├── visualization.py        # 可视化模块
├── generate_synthetic_data.py  # 合成数据生成脚本
├── requirements.txt        # 依赖包列表
├── static/                 # 静态文件
│   ├── css/                # CSS样式文件
│   ├── js/                 # JavaScript文件
│   └── img/                # 图片文件
├── templates/              # HTML模板
│   ├── index.html          # 首页模板
│   ├── dashboard.html      # 仪表盘页面模板
│   ├── analysis.html       # 数据分析页面模板
│   ├── prediction.html     # 预测页面模板
│   ├── user_analysis.html  # 用户分析页面模板
│   └── pdf_report.html     # PDF报告模板
└── data/                   # 数据文件
    ├── models/             # 保存训练好的模型
    ├── synthetic_gait_data.xlsx  # 合成步态数据
    └── subject_profiles.xlsx     # 受试者信息数据
```

## 使用指南

### 首页

首页提供系统概述和主要功能的入口，包括数据摘要和步态类型分布图表。

### 仪表盘

仪表盘页面展示步态数据的实时可视化，包括：
- 步态参数图表
- 加速度计数据
- 陀螺仪数据
- 足底压力分布

### 数据分析

数据分析页面提供深入的数据分析功能，包括：
- 特征相关性热力图
- PCA降维分析
- 步态类型比较
- 异常检测

### 预测

预测页面展示基于机器学习模型的预测结果，包括：
- 步态类型预测
- 疲劳程度预测
- 跌倒风险预测
- 健康状况预测

### 用户分析

用户分析页面针对特定用户进行步态健康评估，包括：
- 用户基本信息
- 步态健康评分
- 风险评估
- 健康建议

### PDF导出

用户可以通过用户分析页面导出PDF格式的步态健康报告，包含用户信息、健康评分和健康建议。

## 数据来源

系统使用以下数据源：
1. **合成数据**：使用`generate_synthetic_data.py`脚本生成的模拟步态数据
2. **Excel文件**：从`synthetic_gait_data.xlsx`和`subject_profiles.xlsx`读取数据

## 开发说明

### 数据处理

数据处理模块(`data_processor.py`)负责加载、处理和管理步态数据，主要功能包括：
- 从Excel文件加载数据
- 数据清洗和预处理
- 特征提取和归一化

### 可视化

可视化模块(`visualization.py`)负责创建各种图表，主要功能包括：
- 从Excel文件读取数据并创建图表
- 创建传感器数据图表
- 创建足底压力分布图表
- 创建相关性热力图
- 创建PCA散点图
- 创建步态类型比较雷达图
- 创建预测结果图表

### 预测

预测模块(`prediction.py`)负责训练机器学习模型并进行预测，主要功能包括：
- 训练步态分类器
- 训练疲劳预测模型
- 训练跌倒风险预测模型
- 训练健康状况预测模型

## 注意事项

- 系统目前使用合成数据进行演示，实际应用中可以替换为真实的步态数据
- 预测模型的准确性取决于训练数据的质量和数量
- 系统生成的健康建议仅供参考，不应替代专业医疗建议

## 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。

## 联系方式

如有任何问题或建议，请联系项目维护者。

---

© 2025 步态分析与预测系统 | 毕业设计项目
