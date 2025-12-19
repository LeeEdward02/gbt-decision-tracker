"""
SHAP解释模块
使用SHAP（TreeSHAP）对模型预测进行可解释性分析
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import shap


def build_shap_explainer(model, background_data: pd.DataFrame, model_type: str = "xgboost"):
    """
    构建SHAP Explainer（使用TreeSHAP）

    Parameters:
        model: 预训练模型
        background_data: 背景数据（用于计算期望值）
        model_type: 模型类型

    Returns:
        SHAP解释器
    """
    print("正在构建SHAP解释器...")

    # 对背景数据进行采样（如果数据量太大）
    if len(background_data) > 100:
        print(f"背景数据样本数: {len(background_data)}，随机采样100个样本")
        background_data = background_data.sample(n=100, random_state=42)

    # 根据模型类型选择解释器
    if model_type.lower() == "xgboost":
        # TreeSHAP专为树模型设计，计算速度快且精确
        explainer = shap.Explainer(model, background_data)
    elif model_type.lower() in ["sklearn", "lightgbm", "catboost"]:
        # 其他树模型也使用TreeExplainer
        explainer = shap.TreeExplainer(model, background_data)
    else:
        # 通用解释器（速度较慢）
        explainer = shap.Explainer(model, background_data)

    print("SHAP解释器构建成功！")
    print(f"解释器类型: {type(explainer).__name__}")

    return explainer


def compute_shap_values(explainer, X: pd.DataFrame, batch_size: Optional[int] = None) -> shap.Explanation:
    """
    计算SHAP值

    Parameters:
        explainer: SHAP解释器
        X: 要解释的数据
        batch_size: 批处理大小，用于大数据集

    Returns:
        SHAP值对象
    """
    print(f"开始计算 {len(X)} 个样本的SHAP值...")

    if batch_size is None or len(X) <= batch_size:
        # 一次性计算所有样本
        shap_values = explainer(X)
    else:
        # 批量计算（适用于大数据集）
        print(f"使用批处理模式，批次大小: {batch_size}")
        shap_values_list = []

        for i in range(0, len(X), batch_size):
            batch_X = X.iloc[i:i+batch_size]
            print(f"处理批次: {i//batch_size + 1}/{(len(X)-1)//batch_size + 1}")
            batch_shap = explainer(batch_X)
            shap_values_list.append(batch_shap)

        # 合并结果
        shap_values = shap.Explanation(
            values=np.concatenate([sv.values for sv in shap_values_list]),
            base_values=np.concatenate([sv.base_values for sv in shap_values_list]),
            data=np.concatenate([sv.data for sv in shap_values_list]),
            feature_names=shap_values_list[0].feature_names
        )

    print("SHAP值计算完成！")
    return shap_values


def get_feature_importance(shap_values: shap.Explanation, X: pd.DataFrame) -> pd.DataFrame:
    """
    获取特征重要性（基于SHAP值的绝对值）

    Parameters:
        shap_values: SHAP值对象
        X: 特征数据

    Returns:
        包含特征重要性的DataFrame
    """
    # 计算每个特征的平均绝对SHAP值
    if len(shap_values.values.shape) == 3:  # 多分类情况
        # 对所有类别取平均
        importance = np.mean(np.abs(shap_values.values), axis=(0, 2))
    else:  # 二分类或回归
        importance = np.mean(np.abs(shap_values.values), axis=0)

    # 创建重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    })

    # 按重要性排序
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    return importance_df


def explain_single_sample(shap_values: shap.Explanation, X: pd.DataFrame,
                         sample_idx: int, max_display: int = 10) -> Dict:
    """
    解释单个样本的预测

    Parameters:
        shap_values: SHAP值对象
        X: 特征数据
        sample_idx: 样本索引
        max_display: 最大显示特征数

    Returns:
        单样本解释结果
    """
    sample_shap = shap_values[sample_idx]
    sample_data = X.iloc[sample_idx]

    # 获取SHAP值和特征值
    if len(sample_shap.values.shape) > 1:
        # 多分类情况，使用第一个类别的SHAP值
        shap_vals = sample_shap.values[:, 0] if sample_shap.values.shape[1] > 1 else sample_shap.values.flatten()
    else:
        shap_vals = sample_shap.values

    # 创建特征贡献DataFrame
    contribution_df = pd.DataFrame({
        'feature': X.columns,
        'value': sample_data.values,
        'shap_value': shap_vals,
        'abs_shap': np.abs(shap_vals)
    })

    # 按绝对SHAP值排序
    contribution_df = contribution_df.sort_values('abs_shap', ascending=False).reset_index(drop=True)

    # 分离正向和负向贡献
    positive_contrib = contribution_df[contribution_df['shap_value'] > 0].head(max_display//2)
    negative_contrib = contribution_df[contribution_df['shap_value'] < 0].head(max_display//2)

    # 处理base_value
    base_val = sample_shap.base_values
    if isinstance(base_val, np.ndarray):
        if base_val.size == 1:
            base_val = base_val.item()
        elif base_val.size > 1:
            base_val = base_val[0]

    return {
        'sample_index': sample_idx,
        'base_value': base_val,
        'prediction': base_val + np.sum(shap_vals),
        'total_shap': np.sum(shap_vals),
        'positive_contributions': positive_contrib.to_dict('records'),
        'negative_contributions': negative_contrib.to_dict('records'),
        'top_features': contribution_df.head(max_display).to_dict('records')
    }


def save_shap_values(shap_values: shap.Explanation, filepath: str):
    """
    保存SHAP值到文件

    Parameters:
        shap_values: SHAP值对象
        filepath: 保存路径
    """
    print(f"保存SHAP值到: {filepath}")
    np.save(filepath, {
        'values': shap_values.values,
        'base_values': shap_values.base_values,
        'data': shap_values.data,
        'feature_names': shap_values.feature_names
    })
    print("SHAP值保存成功！")


def load_shap_values(filepath: str, X: pd.DataFrame) -> shap.Explanation:
    """
    从文件加载SHAP值

    Parameters:
        filepath: 文件路径
        X: 特征数据（用于获取特征名称）

    Returns:
        SHAP值对象
    """
    print(f"从文件加载SHAP值: {filepath}")
    loaded = np.load(filepath, allow_pickle=True).item()

    return shap.Explanation(
        values=loaded['values'],
        base_values=loaded['base_values'],
        data=loaded['data'],
        feature_names=loaded['feature_names']
    )