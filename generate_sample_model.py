"""
生成示例数据和预训练模型
用于演示系统的可解释性分析功能
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def generate_synthetic_data(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, random_state=42):
    """
    生成合成分类数据集

    Parameters:
        n_samples: 样本数
        n_features: 特征数
        n_informative: 有用特征数
        n_redundant: 冗余特征数
        random_state: 随机种子

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print("正在生成合成数据集...")

    # 生成数据
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=2,
        random_state=random_state
    )

    # 创建特征名称
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # 转换为DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    # 分割训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=random_state, stratify=y
    )

    print(f"数据生成完成！")
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    print(f"正负样本比例 - 训练集: {y_train.value_counts().to_dict()}")
    print(f"正负样本比例 - 测试集: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test, feature_names


def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    训练XGBoost模型并保存

    Parameters:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签

    Returns:
        训练好的模型
    """
    print("\n正在训练XGBoost模型...")

    # 初始化模型
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # 评估模型
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"模型训练完成！")
    print(f"训练集准确率: {train_score:.4f}")
    print(f"测试集准确率: {test_score:.4f}")

    return model


def save_data_and_model(X_train, X_test, y_train, y_test, model):
    """
    保存数据集和模型

    Parameters:
        X_train, X_test, y_train, y_test: 数据集
        model: 训练好的模型
    """
    print("\n正在保存数据和模型...")

    # 创建目录
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    # 保存训练数据
    train_data = pd.concat([X_train, y_train], axis=1)
    train_data.to_csv("data/train.csv", index=False)
    print(f"训练数据已保存至: data/train.csv")

    # 保存测试数据
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv("data/test.csv", index=False)
    print(f"测试数据已保存至: data/test.csv")

    # 保存模型
    # 使用Booster对象来保存模型
    booster = model.get_booster()
    booster.save_model("models/xgb_model.json")
    print(f"模型已保存至: models/xgb_model.json")

    # 保存模型信息
    model_info = {
        "n_features": len(X_train.columns),
        "n_classes": len(np.unique(y_train)),
        "feature_names": list(X_train.columns),
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "train_accuracy": float(model.score(X_train, y_train)),
        "test_accuracy": float(model.score(X_test, y_test))
    }

    with open("models/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"模型信息已保存至: models/model_info.json")


def main():
    """主函数"""
    print("="*60)
    print("生成示例数据和预训练模型")
    print("="*60)

    # 生成数据
    X_train, X_test, y_train, y_test, feature_names = generate_synthetic_data(
        n_samples=2000,
        n_features=20,
        n_informative=12,
        n_redundant=5
    )

    # 训练模型
    model = train_xgboost_model(X_train, y_train, X_test, y_test)

    # 保存数据和模型
    save_data_and_model(X_train, X_test, y_train, y_test, model)

    print("\n" + "="*60)
    print("生成完成！")
    print("\n现在可以运行 main.py 来进行可解释性分析：")
    print("python main.py")
    print("="*60)


if __name__ == "__main__":
    main()