"""
决策路径追踪模块
追踪样本在梯度提升树中每棵树的决策路径，实现完整的决策链可解释性
"""

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb


def extract_decision_paths(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    提取每个样本在每棵树上的叶节点索引

    Parameters:
        model: XGBoost模型
        X: 输入特征

    Returns:
        包含决策路径的DataFrame
    """
    print("正在提取决策路径...")
    print(f"样本数量: {len(X)}")

    # 根据模型类型获取叶节点索引
    if isinstance(model, xgb.Booster):
        # XGBoost Booster对象
        dmatrix = xgb.DMatrix(X)
        # 使用predict获取叶节点索引
        leaf_indices = model.predict(dmatrix, pred_leaf=True)
    else:
        # XGBClassifier或其他模型
        leaf_indices = model.apply(X)

    # 创建列名（tree_0, tree_1, ...）
    n_trees = leaf_indices.shape[1]
    column_names = [f"tree_{i}" for i in range(n_trees)]

    # 构建DataFrame
    leaf_df = pd.DataFrame(
        leaf_indices,
        columns=column_names,
        index=X.index
    )

    print(f"决策路径提取完成！共 {n_trees} 棵树")
    return leaf_df


def get_tree_structure(model, tree_idx: int) -> Dict:
    """
    获取指定树的结构信息

    Parameters:
        model: XGBoost模型
        tree_idx: 树的索引

    Returns:
        树结构信息
    """
    # 获取Booster对象
    if isinstance(model, xgb.Booster):
        booster = model
    else:
        booster = model.get_booster()

    dump = booster.get_dump(dump_format='json')

    if tree_idx >= len(dump):
        raise ValueError(f"树索引 {tree_idx} 超出范围，总共有 {len(dump)} 棵树")

    tree_json = json.loads(dump[tree_idx])
    return tree_json


def extract_decision_rules(model, X: pd.DataFrame, sample_idx: int,
                           tree_idx: int) -> List[Dict]:
    """
    提取单个样本在指定树中的决策规则

    Parameters:
        model: XGBoost模型
        X: 输入特征
        sample_idx: 样本索引
        tree_idx: 树索引

    Returns:
        决策规则列表
    """
    # 获取树结构
    tree_structure = get_tree_structure(model, tree_idx)

    # 获取样本数据
    sample = X.iloc[sample_idx]

    # 递归提取决策路径
    rules = []
    current_node = tree_structure

    def traverse_tree(node, depth=0):
        """递归遍历树，提取决策规则"""
        if 'leaf' in node:
            # 到达叶节点
            rules.append({
                'type': 'leaf',
                'depth': depth,
                'value': node['leaf'],
                'rule': f"到达叶节点，值={node['leaf']:.4f}"
            })
            return True

        # 检查分裂条件
        feature_name = node['split']
        threshold = node['split_condition']
        feature_value = sample[feature_name]

        # 判断走向
        if feature_value < threshold:
            # 走向左子树
            rules.append({
                'type': 'decision',
                'depth': depth,
                'feature': feature_name,
                'threshold': threshold,
                'value': feature_value,
                'comparison': '<',
                'direction': 'left',
                'rule': f"如果 {feature_name}={feature_value:.4f} < {threshold:.4f}，向左"
            })
            return traverse_tree(node['children'][0], depth + 1)
        else:
            # 走向右子树
            rules.append({
                'type': 'decision',
                'depth': depth,
                'feature': feature_name,
                'threshold': threshold,
                'value': feature_value,
                'comparison': '>=',
                'direction': 'right',
                'rule': f"如果 {feature_name}={feature_value:.4f} >= {threshold:.4f}，向右"
            })
            return traverse_tree(node['children'][1], depth + 1)

    traverse_tree(current_node)
    return rules


def get_full_decision_chain(model, X: pd.DataFrame, sample_idx: int) -> Dict:
    """
    获取单个样本在所有树中的完整决策链

    Parameters:
        model: XGBoost模型
        X: 输入特征
        sample_idx: 样本索引

    Returns:
        完整的决策链信息
    """
    print(f"正在追踪样本 {sample_idx} 的决策链...")

    # 获取叶节点索引
    if isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(X)
        leaf_indices = model.predict(dmatrix, pred_leaf=True)
    else:
        leaf_indices = model.apply(X)
    sample_leaves = leaf_indices[sample_idx]

    # 获取每棵树的决策规则
    n_trees = len(sample_leaves)
    decision_chain = {
        'sample_index': sample_idx,
        'n_trees': n_trees,
        'trees': [],
        'leaf_nodes': sample_leaves.tolist()
    }

    # 收集每棵树的决策信息
    for i in range(n_trees):
        rules = extract_decision_rules(model, X, sample_idx, i)
        tree_info = {
            'tree_index': i,
            'leaf_index': sample_leaves[i],
            'decision_rules': rules,
            'depth': len(rules)
        }
        decision_chain['trees'].append(tree_info)

    # 计算统计信息
    depths = [tree['depth'] for tree in decision_chain['trees']]
    decision_chain['stats'] = {
        'min_depth': min(depths),
        'max_depth': max(depths),
        'avg_depth': np.mean(depths),
        'total_decisions': sum(depths)
    }

    print(f"决策链追踪完成！共 {n_trees} 棵树，平均深度: {decision_chain['stats']['avg_depth']:.2f}")

    return decision_chain


def analyze_feature_usage_in_paths(model, X: pd.DataFrame,
                                   sample_indices: Optional[List[int]] = None) -> pd.DataFrame:
    """
    分析特征在决策路径中的使用频率

    Parameters:
        model: XGBoost模型
        X: 输入特征
        sample_indices: 要分析的样本索引列表，None表示分析所有样本

    Returns:
        特征使用频率统计
    """
    if sample_indices is None:
        sample_indices = range(len(X))

    print(f"分析 {len(sample_indices)} 个样本的决策路径中的特征使用...")

    feature_usage = {feature: 0 for feature in X.columns}
    total_decisions = 0

    # 获取所有样本的叶节点索引（一次性获取以提高效率）
    if isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(X)
        leaf_indices = model.predict(dmatrix, pred_leaf=True)
    else:
        leaf_indices = model.apply(X)

    for sample_idx in sample_indices:
        sample_leaves = leaf_indices[sample_idx]

        for tree_idx in range(len(sample_leaves)):
            rules = extract_decision_rules(model, X, sample_idx, tree_idx)
            for rule in rules:
                if rule['type'] == 'decision':
                    feature_usage[rule['feature']] += 1
                    total_decisions += 1

    # 计算使用频率
    usage_df = pd.DataFrame([
        {
            'feature': feature,
            'usage_count': count,
            'usage_ratio': count / total_decisions if total_decisions > 0 else 0
        }
        for feature, count in feature_usage.items()
    ])

    # 按使用频率排序
    usage_df = usage_df.sort_values('usage_count', ascending=False).reset_index(drop=True)

    print("特征使用频率分析完成！")
    return usage_df


def save_decision_paths(decision_paths: pd.DataFrame, filepath: str):
    """
    保存决策路径到文件

    Parameters:
        decision_paths: 决策路径DataFrame
        filepath: 保存路径
    """
    print(f"保存决策路径到: {filepath}")
    decision_paths.to_csv(filepath, index=False)
    print("决策路径保存成功！")


def load_decision_paths(filepath: str) -> pd.DataFrame:
    """
    从文件加载决策路径

    Parameters:
        filepath: 文件路径

    Returns:
        决策路径DataFrame
    """
    print(f"从文件加载决策路径: {filepath}")
    return pd.read_csv(filepath)


def visualize_tree_decision_path(model, X: pd.DataFrame, sample_idx: int,
                                 tree_idx: int, max_depth: int = 3):
    """
    可视化单个样本在单棵树中的决策路径（文本形式）

    Parameters:
        model: XGBoost模型
        X: 输入特征
        sample_idx: 样本索引
        tree_idx: 树索引
        max_depth: 最大显示深度
    """
    rules = extract_decision_rules(model, X, sample_idx, tree_idx)
    sample = X.iloc[sample_idx]

    print(f"\n=== 样本 {sample_idx} 在树 {tree_idx} 中的决策路径 ===")
    print(f"样本特征值:")
    for feature, value in sample.head(5).items():  # 只显示前5个特征
        print(f"  {feature}: {value:.4f}")

    print(f"\n决策路径:")
    for i, rule in enumerate(rules[:max_depth]):
        if rule['type'] == 'decision':
            indent = "  " * rule['depth']
            print(f"{indent}├─ {rule['rule']}")
        else:
            indent = "  " * rule['depth']
            print(f"{indent}└─ {rule['rule']}")

    if len(rules) > max_depth:
        print(f"  ... (还有 {len(rules) - max_depth} 个决策步骤)")
