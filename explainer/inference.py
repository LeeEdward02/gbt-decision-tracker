"""
推理模块
负责使用预训练模型进行前向推理，不涉及任何训练过程
"""

import time
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb


def run_inference(model, X: pd.DataFrame, task_type: str = "classification") -> Dict:
    """
    对输入样本进行前向推理

    parameters:
        model: 预训练模型
        X: 特征数据
        task_type: 任务类型，'classification' 或 'regression'

    returns:
        包含预测结果的字典
    """
    print(f"开始对 {len(X)} 个样本进行推理...")
    start_time = time.time()

    # 确保输入是DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # 执行预测
    if isinstance(model, xgb.Booster):
        # XGBoost Booster对象
        dmatrix = xgb.DMatrix(X)
        y_pred_raw = model.predict(dmatrix)

        if task_type.lower() == "classification":
            # 对于分类任务，XGBoost返回原始分数或概率
            if len(y_pred_raw.shape) == 1:
                # 二分类，返回概率
                y_prob = np.vstack([1 - y_pred_raw, y_pred_raw]).T
                y_pred = (y_pred_raw > 0.5).astype(int)
            else:
                # 多分类
                y_prob = y_pred_raw
                y_pred = np.argmax(y_pred_raw, axis=1)

            classes = [0, 1] if len(y_prob[0]) == 2 else list(range(y_prob.shape[1]))

            # 构建概率字典
            prob_dict = {}
            for i, cls in enumerate(classes):
                prob_dict[f"prob_class_{cls}"] = y_prob[:, i]

            result = {
                "prediction": y_pred,
                "probability": y_prob,
                "probabilities": prob_dict,
                "classes": classes
            }
        else:
            # 回归任务
            result = {
                "prediction": y_pred_raw,
                "values": y_pred_raw
            }
    else:
        # 其他模型类型（sklearn等）
        if task_type.lower() == "classification":
            # 分类任务
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)

            # 获取类别标签
            if hasattr(model, 'classes_'):
                classes = model.classes_
            else:
                # 如果没有classes_属性，假设是二分类（0, 1）
                classes = [0, 1]

            # 构建概率字典
            prob_dict = {}
            for i, cls in enumerate(classes):
                prob_dict[f"prob_class_{cls}"] = y_prob[:, i]

            result = {
                "prediction": y_pred,
                "probability": y_prob,
                "probabilities": prob_dict,
                "classes": classes
            }

        elif task_type.lower() == "regression":
            # 回归任务
            y_pred = model.predict(X)
            result = {
                "prediction": y_pred,
                "values": y_pred
            }
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

    # 记录推理时间
    inference_time = time.time() - start_time
    result["inference_time"] = inference_time
    result["n_samples"] = len(X)

    print(f"推理完成！耗时: {inference_time:.4f}秒")
    print(f"平均每个样本推理时间: {inference_time/len(X)*1000:.4f}毫秒")

    return result


def batch_inference(model, X: pd.DataFrame, batch_size: int = 1000,
                   task_type: str = "classification") -> Dict:
    """
    批量推理，适用于大数据集

    Parameters:
        model: 预训练模型
        X: 特征数据
        batch_size: 批次大小
        task_type: 任务类型

    Returns:
        包含所有预测结果的字典
    """
    print(f"开始批量推理，总样本数: {len(X)}, 批次大小: {batch_size}")

    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size

    # 初始化结果存储
    all_predictions = []
    all_probabilities = []
    all_values = []

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_X = X.iloc[start_idx:end_idx]

        print(f"处理批次 {i+1}/{n_batches} (样本 {start_idx} 到 {end_idx})")

        # 对当前批次进行推理
        batch_result = run_inference(model, batch_X, task_type)

        # 存储结果
        all_predictions.extend(batch_result["prediction"])
        if task_type.lower() == "classification":
            all_probabilities.extend(batch_result["probability"])
        else:
            all_values.extend(batch_result["values"])

    # 构建最终结果
    final_result = {
        "prediction": np.array(all_predictions),
        "n_samples": n_samples,
        "n_batches": n_batches
    }

    if task_type.lower() == "classification":
        final_result["probability"] = np.array(all_probabilities)
    else:
        final_result["values"] = np.array(all_values)

    print(f"批量推理完成！共处理 {n_samples} 个样本")

    return final_result


def get_feature_contributions(model, X: pd.DataFrame, sample_idx: int = 0) -> Dict:
    """
    获取单个样本的特征贡献（适用于XGBoost）

    Parameters:
        model: XGBoost模型
        X: 特征数据
        sample_idx: 样本索引

    Returns:
        特征贡献信息
    """
    if not hasattr(model, 'get_booster'):
        raise ValueError("此函数仅适用于XGBoost模型")

    # 获取单样本
    sample = X.iloc[[sample_idx]]

    # 使用XGBoost的predict方法获取贡献值
    if hasattr(model, 'predict'):
        try:
            # 尝试获取贡献值
            contributions = model.predict(sample, pred_contribs=True)
            feature_contributions = contributions[0, :-1]  # 最后一项是偏置值
            bias = contributions[0, -1]

            return {
                "feature_contributions": dict(zip(X.columns, feature_contributions)),
                "bias": bias,
                "total_contribution": np.sum(feature_contributions) + bias
            }
        except:
            print("无法获取特征贡献值，可能需要更新XGBoost版本")
            return None

    return None