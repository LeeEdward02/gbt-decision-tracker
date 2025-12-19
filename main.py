"""
主入口文件 - 基于梯度提升树的模型决策链全程可解释性追踪
本项目实现了一个完整的模型解释框架，包括：
1. 预训练模型加载
2. 前向推理
3. SHAP值计算
4. 决策路径追踪
5. 可视化解释
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 导入项目模块
from config import *
from explainer.model_loader import load_model, get_model_info
from explainer.inference import run_inference
from explainer.shap_explainer import build_shap_explainer, compute_shap_values, \
    get_feature_importance, explain_single_sample
from explainer.decision_trace import extract_decision_paths, get_full_decision_chain, \
    analyze_feature_usage_in_paths
from explainer.visualizer import plot_local_explanation, plot_global_explanation, \
    plot_decision_path_heatmap, plot_prediction_confidence, \
    plot_decision_chain_tree, plot_feature_shap_dependence


def setup_directories():
    """创建必要的目录"""
    directories = [RESULTS_DIR]
    for dir_path in directories:
        Path(dir_path).mkdir(exist_ok=True)
    print("目录结构检查完成！")


def load_data():
    """加载数据"""
    print(f"\n{'=' * 50}")
    print("1. 加载数据")
    print(f"{'=' * 50}")

    # 加载测试数据
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"测试数据文件不存在: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    print(f"加载数据成功！形状: {data.shape}")

    # 分离特征和目标
    if TARGET_COL in data.columns:
        X = data.drop(columns=[TARGET_COL])
        y = data[TARGET_COL]
        print(f"特征数量: {len(X.columns)}")
        print(f"目标列: {TARGET_COL}")
    else:
        X = data
        y = None
        print(f"特征数量: {len(X.columns)}")
        print("未检测到目标列，将仅使用特征进行解释")

    # 如果指定了特征列，则只使用这些列
    if FEATURE_COLS is not None:
        X = X[FEATURE_COLS]
        print(f"使用指定特征列，数量: {len(FEATURE_COLS)}")

    return X, y, data


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("基于梯度提升树的模型决策链全程可解释性追踪系统")
    print("=" * 60)

    # 1. 创建目录
    setup_directories()

    # 2. 加载数据
    X, y, data = load_data()

    # 3. 加载模型
    print(f"\n{'=' * 50}")
    print("2. 加载预训练模型")
    print(f"{'=' * 50}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

    model = load_model(MODEL_PATH, MODEL_TYPE)

    # 获取模型信息
    model_info = get_model_info(model)
    print(f"\n模型信息:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # 4. 模型推理
    print(f"\n{'=' * 50}")
    print("3. 执行模型推理")
    print(f"{'=' * 50}")

    inference_result = run_inference(model, X, TASK_TYPE)

    # 保存预测结果
    predictions_df = pd.DataFrame({
        'prediction': inference_result['prediction']
    })
    if TASK_TYPE.lower() == 'classification':
        for i, cls in enumerate(inference_result['classes']):
            predictions_df[f'prob_class_{cls}'] = inference_result['probabilities'][f'prob_class_{cls}']

    predictions_df.to_csv(PREDICTIONS_OUTPUT, index=False)
    print(f"\n预测结果已保存至: {PREDICTIONS_OUTPUT}")

    # 5. SHAP解释分析
    print(f"\n{'=' * 50}")
    print("4. SHAP可解释性分析")
    print(f"{'=' * 50}")

    # 加载背景数据（如果存在）
    if os.path.exists(TRAIN_DATA_PATH):
        print("使用训练数据作为背景数据...")
        background_data = pd.read_csv(TRAIN_DATA_PATH)
        if TARGET_COL in background_data.columns:
            background_data = background_data.drop(columns=[TARGET_COL])
    else:
        print("使用测试数据的子集作为背景数据...")
        background_data = X.sample(n=min(BACKGROUND_SIZE, len(X)), random_state=42)

    # 构建SHAP解释器
    explainer = build_shap_explainer(model, background_data, MODEL_TYPE)

    # 计算SHAP值
    shap_values = compute_shap_values(explainer, X)

    # 保存SHAP值
    if hasattr(shap_values, 'values'):
        np.save(SHAP_VALUES_OUTPUT, shap_values.values)
        print(f"\nSHAP值已保存至: {SHAP_VALUES_OUTPUT}")

    # 获取全局特征重要性
    feature_importance = get_feature_importance(shap_values, X)
    print(f"\nTop 10 重要特征:")
    print(feature_importance.head(10))

    # 6. 决策路径追踪
    print(f"\n{'=' * 50}")
    print("5. 决策路径追踪")
    print(f"{'=' * 50}")

    # 提取所有样本的决策路径
    decision_paths = extract_decision_paths(model, X)
    print(f"\n决策路径矩阵形状: {decision_paths.shape}")
    print(f"前5个样本的决策路径:")
    print(decision_paths.head())

    # 保存决策路径
    decision_paths.to_csv(DECISION_PATHS_OUTPUT, index=False)
    print(f"\n决策路径已保存至: {DECISION_PATHS_OUTPUT}")

    # 分析特征在决策路径中的使用频率
    feature_usage = analyze_feature_usage_in_paths(model, X, sample_indices=range(min(100, len(X))))
    print(f"\n决策路径中使用频率最高的Top 10特征:")
    print(feature_usage.head(10))

    # 7. 可视化展示
    print(f"\n{'=' * 50}")
    print("6. 生成可视化图表")
    print(f"{'=' * 50}")

    # 全局解释图
    print("\n生成全局特征重要性图...")
    plot_global_explanation(
        shap_values, X,
        plot_type="bar",
        save_path=os.path.join(RESULTS_DIR, "global_importance_bar.png")
    )

    plot_global_explanation(
        shap_values, X,
        plot_type="beeswarm",
        save_path=os.path.join(RESULTS_DIR, "global_importance_beeswarm.png")
    )

    # 局部解释图（前3个样本）
    for i in range(min(3, len(X))):
        print(f"\n生成样本 {i} 的局部解释图...")
        plot_local_explanation(
            shap_values, i,
            max_display=MAX_DISPLAY,
            save_path=os.path.join(RESULTS_DIR, f"local_explanation_sample_{i}.png")
        )

        # 获取单样本的详细解释
        sample_explanation = explain_single_sample(shap_values, X, i, MAX_DISPLAY)
        print(f"\n样本 {i} 的解释摘要:")
        print(f"  基准值: {sample_explanation['base_value']:.4f}")
        print(f"  预测值: {sample_explanation['prediction']:.4f}")
        print(f"  总SHAP值: {sample_explanation['total_shap']:.4f}")

    # 决策路径热力图（前20个样本）
    print("\n生成决策路径热力图...")
    plot_decision_path_heatmap(
        decision_paths.iloc[:20],
        save_path=os.path.join(RESULTS_DIR, "decision_path_heatmap.png")
    )

    # 预测置信度分布（分类任务）
    if TASK_TYPE.lower() == 'classification':
        print("\n生成预测置信度分布图...")
        plot_prediction_confidence(
            inference_result['prediction'],
            inference_result['probability'],
            save_path=os.path.join(RESULTS_DIR, "prediction_confidence.png")
        )

    # 决策链可视化（第一个样本的前5棵树）
    print("\n生成决策链可视化...")
    decision_chain = get_full_decision_chain(model, X, sample_idx=0)
    plot_decision_chain_tree(
        decision_chain,
        max_trees=5,
        save_path=os.path.join(RESULTS_DIR, "decision_chain_sample_0.png")
    )

    # 特征SHAP依赖图（最重要的3个特征）
    print("\n生成特征依赖图...")
    top_features = feature_importance['feature'].head(3).tolist()
    for feature in top_features:
        print(f"生成特征 '{feature}' 的SHAP依赖图...")
        plot_feature_shap_dependence(
            shap_values, X, feature,
            save_path=os.path.join(RESULTS_DIR, f"shap_dependence_{feature}.png")
        )

    # 8. 总结报告
    print(f"\n{'=' * 60}")
    print("分析完成！总结报告")
    print(f"{'=' * 60}")
    print(f"1. 数据集: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
    print(f"2. 模型类型: {model_info['model_type']}")
    if 'n_trees' in model_info:
        print(f"   树的数量: {model_info['n_trees']}")
    print(f"3. 任务类型: {TASK_TYPE}")
    print(f"4. SHAP解释: 完成")
    print(f"5. 决策路径: 完成")
    print(f"6. 生成图表: {len(os.listdir(RESULTS_DIR))} 个文件")
    print(f"\n所有结果已保存至 '{RESULTS_DIR}' 目录")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
