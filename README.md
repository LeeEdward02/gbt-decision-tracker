# 基于梯度提升树的模型决策链全程可解释性追踪系统

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1.2-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.50.0-green.svg)](https://shap.readthedocs.io/)

本项目实现了一个完整的机器学习模型可解释性分析框架，专门针对梯度提升树模型（如XGBoost），提供从预测到决策路径的全链路可解释性分析。

## 🌟 主要功能

1. **模型加载与推理** - 支持加载预训练的XGBoost模型并进行预测
2. **SHAP可解释性分析** - 计算特征重要性，提供全局和局部解释
3. **决策路径追踪** - 提取并分析每个样本在模型中的完整决策路径
4. **可视化展示** - 生成多种直观的可视化图表
5. **特征依赖分析** - 展示特征值与预测结果之间的关系

## 📁 项目结构

```
explainable_trace/
├── main.py                    # 主入口文件
├── config.py                  # 全局配置文件
├── generate_sample_model.py   # 生成示例数据和模型
├── requirements.txt           # 项目依赖
├── data/                      # 数据目录
│   ├── train.csv             # 训练数据（用于SHAP背景数据）
│   └── test.csv              # 测试数据（用于推理和解释）
├── models/                    # 模型目录
│   ├── xgb_model.json        # 预训练XGBoost模型
│   └── model_info.json       # 模型信息
├── explainer/                 # 核心解释模块
│   ├── model_loader.py       # 模型加载器
│   ├── inference.py          # 模型推理
│   ├── shap_explainer.py     # SHAP解释器
│   ├── decision_trace.py     # 决策路径追踪
│   └── visualizer.py         # 可视化工具
└── results/                   # 结果输出目录
    ├── predictions.csv       # 预测结果
    ├── shap_values.npy       # SHAP值
    ├── decision_paths.csv    # 决策路径
    └── *.png                 # 可视化图表
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/LeeEdward02/Fully-Explainable-Tracking-System-for-Model-Decision-Chains-Based-on-Gradient-Boosting-Trees.git
cd explainable_trace

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据和模型

如果您没有现成的数据和模型，可以运行示例生成脚本：

```bash
python generate_sample_model.py
```

这将：
- 生成合成分类数据集（2000个样本，20个特征）
- 训练一个XGBoost分类模型
- 将数据保存到 `data/` 目录
- 将模型保存到 `models/` 目录

### 3. 运行可解释性分析

```bash
python main.py
```

程序将自动执行以下步骤：
1. 加载数据和预训练模型
2. 执行模型推理
3. 计算SHAP值
4. 提取决策路径
5. 生成可视化图表

## 📊 输出说明

运行完成后，结果将保存在 `results/` 目录中：

- **predictions.csv** - 模型预测结果，包含预测值和各类别概率
- **shap_values.npy** - 每个样本每个特征的SHAP值
- **decision_paths.csv** - 每个样本在各棵树中的决策路径
- **global_importance_bar.png** - 全局特征重要性柱状图
- **global_importance_beeswarm.png** - 全局特征重要性蜂群图
- **local_explanation_sample_*.png** - 单个样本的局部解释图
- **decision_path_heatmap.png** - 决策路径热力图
- **prediction_confidence.png** - 预测置信度分布（分类任务）
- **decision_chain_sample_*.png** - 决策链可视化图
- **shap_dependence_*.png** - 重要特征的SHAP依赖图

## ⚙️ 配置说明

主要配置项在 `config.py` 文件中：

```python
# 模型和数据路径
MODEL_PATH = "models/xgb_model.json"      # 预训练模型路径
DATA_PATH = "data/test.csv"               # 测试数据路径
TRAIN_DATA_PATH = "data/train.csv"        # 训练数据路径

# 模型配置
MODEL_TYPE = "xgboost"                    # 模型类型
TASK_TYPE = "classification"              # 任务类型：classification/regression

# 数据配置
TARGET_COL = "target"                     # 目标列名
FEATURE_COLS = None                       # 特征列名列表

# SHAP参数
BACKGROUND_SIZE = 100                     # SHAP背景数据样本数
MAX_DISPLAY = 10                          # 图表中显示的最大特征数
```

## 🔧 核心模块介绍

### model_loader.py
- 负责加载预训练模型
- 支持XGBoost模型格式
- 提取模型基本信息（树的数量、特征数等）

### inference.py
- 执行模型预测
- 支持分类和回归任务
- 返回预测值和概率（分类任务）

### shap_explainer.py
- 构建SHAP解释器
- 计算SHAP值
- 提供特征重要性排序
- 生成单样本解释

### decision_trace.py
- 提取决策路径
- 分析特征使用频率
- 构建完整决策链

### visualizer.py
- 生成各类可视化图表
- 支持全局和局部解释图
- 创建决策路径热力图
- 绘制特征依赖关系