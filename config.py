"""
全局配置文件 - 基于梯度提升树的模型决策链可解释性追踪项目
"""

# 模型和数据路径配置
MODEL_PATH = "models/xgb_model.json"      # 预训练模型路径
DATA_PATH = "data/test.csv"               # 测试数据路径（用于推理和解释）
TRAIN_DATA_PATH = "data/train.csv"        # 训练数据路径（用于SHAP背景数据）

# 模型配置
MODEL_TYPE = "xgboost"                    # 模型类型
TASK_TYPE = "classification"              # 任务类型：分类或回归

# 数据列配置
TARGET_COL = "target"                     # 目标列名
FEATURE_COLS = None                       # 特征列名列表，None表示使用除目标列外的所有列

# SHAP解释参数
BACKGROUND_SIZE = 100                     # SHAP背景数据样本数
MAX_DISPLAY = 10                          # 图表中显示的最大特征数

# 输出路径配置
RESULTS_DIR = "results"                   # 结果输出目录
DECISION_PATHS_OUTPUT = "results/decision_paths.csv"    # 决策路径输出文件
SHAP_VALUES_OUTPUT = "results/shap_values.npy"          # SHAP值输出文件
PREDICTIONS_OUTPUT = "results/predictions.csv"          # 预测结果输出文件

# 可视化配置
FIGURE_SIZE = (10, 6)                     # 图形尺寸
DPI = 300                                 # 图形分辨率
SAVE_PLOTS = True                         # 是否保存图形