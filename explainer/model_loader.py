"""
模型加载器模块
负责加载预训练的XGBoost模型，不进行任何训练操作
"""

import pickle
from pathlib import Path

import joblib
import xgboost as xgb


def load_model(model_path: str, model_type: str = "xgboost"):
    """
    加载预训练模型

    Parameters:
        model_path: 模型文件路径
        model_type: 模型类型，支持 'xgboost', 'sklearn', 'pickle'

    Returns:
        加载的模型对象
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"正在加载模型: {model_path}")

    if model_type.lower() == "xgboost":
        # 加载XGBoost模型
        if model_path.suffix == '.json':
            # XGBoost JSON格式 - 直接加载Booster对象
            model = xgb.Booster()
            model.load_model(str(model_path))
        elif model_path.suffix in ['.pkl', '.pickle', '.joblib']:
            # pickle或joblib格式
            model = joblib.load(model_path)
        else:
            raise ValueError(f"不支持的XGBoost模型格式: {model_path.suffix}")

    elif model_type.lower() == "sklearn":
        # 加载scikit-learn模型
        if model_path.suffix in ['.pkl', '.pickle']:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        elif model_path.suffix == '.joblib':
            model = joblib.load(model_path)
        else:
            raise ValueError(f"不支持的sklearn模型格式: {model_path.suffix}")

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    print(f"模型加载成功！")
    print(f"模型类型: {type(model)}")

    # 对于XGBoost模型，打印基本信息
    if hasattr(model, 'n_features_in_'):
        print(f"特征数量: {model.n_features_in_}")
    if hasattr(model, 'n_classes_'):
        print(f"类别数量: {model.n_classes_}")
    if hasattr(model, 'feature_names_in_'):
        print(f"特征名称: {list(model.feature_names_in_)}")

    return model


def get_model_info(model):
    """
    获取模型基本信息

    Parameters:
        model: 模型对象

    Returns:
        包含模型信息的字典
    """
    info = {
        'model_type': type(model).__name__,
    }

    # XGBoost特有信息
    if hasattr(model, 'n_features_in_'):
        info['n_features'] = model.n_features_in_
    if hasattr(model, 'n_classes_'):
        info['n_classes'] = model.n_classes_
    if hasattr(model, 'feature_names_in_'):
        info['feature_names'] = list(model.feature_names_in_)
    # 处理XGBoost Booster对象
    if isinstance(model, xgb.Booster):
        # 获取树的棵数
        try:
            info['n_trees'] = len(model.get_dump())
        except:
            info['n_trees'] = 'Unknown'
    elif hasattr(model, 'get_booster'):
        # 获取Booster对象
        booster = model.get_booster()
        # 获取树的棵数
        try:
            info['n_trees'] = booster.best_iteration + 1 if booster.best_iteration > 0 else len(booster.get_dump())
        except:
            info['n_trees'] = len(booster.get_dump())
        # 获取树的深度
        if hasattr(model, 'max_depth'):
            info['max_depth'] = model.max_depth

    return info