"""
可视化模块
提供模型解释的各种可视化功能，包括SHAP图、决策路径图等
"""

import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
from matplotlib import font_manager

# 添加中文字体支持
try:
    # Windows系统字体
    font_list = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
    # 尝试找到可用字体
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    selected_font = 'Arial'  # 默认
    for font in font_list:
        if font in available_fonts:
            selected_font = font
            break

    # 设置matplotlib字体
    matplotlib.rcParams['font.sans-serif'] = [selected_font] + font_list
    matplotlib.rcParams['axes.unicode_minus'] = True  # 允许负号显示
    matplotlib.rcParams['font.family'] = 'sans-serif'

    # 设置matplotlib的负号显示
    plt.rcParams['axes.unicode_minus'] = True
    plt.rcParams['font.sans-serif'] = [selected_font] + font_list

except:
    # 如果设置失败，使用默认配置
    matplotlib.rcParams['axes.unicode_minus'] = True
    plt.rcParams['axes.unicode_minus'] = True

# 设置seaborn样式
sns.set_style("whitegrid", {'font.sans-serif': [selected_font] if 'selected_font' in locals() else 'Arial'})


def plot_local_explanation(shap_values: shap.Explanation, index: int,
                          max_display: int = 10, save_path: Optional[str] = None):
    """
    单样本解释（瀑布图）

    Parameters:
        shap_values: SHAP值对象
        index: 样本索引
        max_display: 最大显示特征数
        save_path: 保存路径
    """
    # 创建新图形
    plt.figure(figsize=(12, 8))

    # 使用SHAP的瀑布图，但是保存到当前图形
    shap.plots.waterfall(
        shap_values[index],
        max_display=max_display,
        show=False
    )

    # 重新设置标题以确保正确显示
    plt.title(f'SHAP Local Explanation for Sample {index} (Waterfall Plot)',
              fontsize=16, pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Local explanation plot saved to: {save_path}")

    plt.show()


def plot_global_explanation(shap_values: shap.Explanation, X: pd.DataFrame,
                           plot_type: str = "bar", save_path: Optional[str] = None):
    """
    全局特征重要性图

    Parameters:
        shap_values: SHAP值对象
        X: 特征数据
        plot_type: 图形类型：'bar'（条形图）或 'beeswarm'（蜂群图）
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))

    if plot_type == "bar":
        # 条形图 - 显示特征重要性排序
        shap.plots.bar(shap_values, show=False)
        plt.title('Global Feature Importance (SHAP Bar Plot)', fontsize=16, pad=20)
    elif plot_type == "beeswarm":
        # 蜂群图 - 显示特征值与SHAP值的关系
        shap.plots.beeswarm(shap_values, show=False)
        plt.title('Feature Values vs SHAP Values (Beeswarm Plot)', fontsize=16, pad=20)
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Global explanation plot saved to: {save_path}")

    plt.show()


def plot_decision_path_heatmap(decision_paths: pd.DataFrame,
                              sample_indices: Optional[List[int]] = None,
                              save_path: Optional[str] = None):
    """
    决策路径热力图

    Parameters:
        decision_paths: 决策路径DataFrame
        sample_indices: 要显示的样本索引
        save_path: 保存路径
    """
    if sample_indices is not None:
        decision_paths = decision_paths.iloc[sample_indices]

    plt.figure(figsize=(15, min(10, len(decision_paths) * 0.5)))

    # 创建热力图
    sns.heatmap(
        decision_paths,
        cmap='viridis',
        cbar_kws={'label': '叶节点索引'},
        xticklabels=5,  # 每5棵树显示一个标签
        yticklabels=True
    )

    plt.title('Decision Path Heatmap (Leaf Nodes per Tree)', fontsize=16, pad=20)
    plt.xlabel('Tree Index', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"决策路径热力图已保存至: {save_path}")

    plt.show()


def plot_feature_shap_dependence(shap_values: shap.Explanation, X: pd.DataFrame,
                                feature_name: str, save_path: Optional[str] = None):
    """
    单个特征的SHAP依赖图

    Parameters:
        shap_values: SHAP值对象
        X: 特征数据
        feature_name: 特征名称
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))

    # 创建依赖图
    shap.plots.scatter(
        shap_values[:, feature_name],
        color=shap_values,
        show=False
    )

    plt.title(f'SHAP Dependence Plot for Feature "{feature_name}"', fontsize=16, pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP依赖图已保存至: {save_path}")

    plt.show()


def plot_prediction_confidence(predictions: np.ndarray, probabilities: np.ndarray,
                               save_path: Optional[str] = None):
    """
    预测置信度分布图

    Parameters:
        predictions: 预测结果
        probabilities: 预测概率
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 5))

    # 子图1：预测类别分布
    plt.subplot(1, 2, 1)
    unique, counts = np.unique(predictions, return_counts=True)
    plt.bar(unique, counts, alpha=0.7)
    plt.xlabel('预测类别')
    plt.ylabel('样本数')
    plt.title('预测类别分布')
    plt.xticks(unique)

    # 子图2：预测置信度分布
    plt.subplot(1, 2, 2)
    max_probs = np.max(probabilities, axis=1)
    plt.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('最大预测概率')
    plt.ylabel('频次')
    plt.title('预测置信度分布')
    plt.axvline(max_probs.mean(), color='red', linestyle='--',
                label=f'平均置信度: {max_probs.mean():.3f}')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测置信度图已保存至: {save_path}")

    plt.show()


def plot_decision_chain_tree(decision_chain: Dict, max_trees: int = 10,
                            save_path: Optional[str] = None):
    """
    决策链树形图（文本可视化）

    Parameters:
        decision_chain: 决策链字典
        max_trees: 最大显示树数
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(15, max_trees * 2))
    ax.axis('off')

    y_pos = 0
    n_trees = min(len(decision_chain['trees']), max_trees)

    text_lines = [f"Sample {decision_chain['sample_index']} Decision Chain (showing first {n_trees} trees)"]
    text_lines.append("=" * 60)
    text_lines.append(f"Average Decision Depth: {decision_chain['stats']['avg_depth']:.2f}")
    text_lines.append("")

    for i in range(n_trees):
        tree_info = decision_chain['trees'][i]
        text_lines.append(f"Tree {tree_info['tree_index']} (depth: {tree_info['depth']}):")

        for rule in tree_info['decision_rules'][:3]:  # 只显示前3个规则
            indent = "  " * rule['depth']
            if rule['type'] == 'decision':
                # 使用更简单的字符
                symbol = "->" if rule['direction'] == 'right' else "->"
                text_lines.append(f"{indent}{symbol} {rule['rule']}")
            else:
                text_lines.append(f"{indent}=> {rule['rule']}")

        if tree_info['depth'] > 3:
            text_lines.append(f"{indent}  ... ({tree_info['depth'] - 3} more steps)")
        text_lines.append("")

    # 添加文本，使用支持更多字符的字体
    ax.text(0.05, 0.95, '\n'.join(text_lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace')

    plt.title('Decision Chain Visualization', fontsize=16, pad=20)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Decision chain plot saved to: {save_path}")

    plt.show()


def create_interactive_shap_plot(shap_values: shap.Explanation, X: pd.DataFrame):
    """
    创建交互式SHAP图（使用Plotly）

    Parameters:
        shap_values: SHAP值对象
        X: 特征数据
    """
    # 计算特征重要性
    if len(shap_values.values.shape) == 3:
        importance = np.mean(np.abs(shap_values.values), axis=(0, 2))
    else:
        importance = np.mean(np.abs(shap_values.values), axis=0)

    # 创建重要性条形图
    fig = go.Figure(data=[
        go.Bar(
            x=X.columns,
            y=importance,
            marker_color='lightblue'
        )
    ])

    fig.update_layout(
        title='交互式特征重要性（SHAP值）',
        xaxis_title='特征',
        yaxis_title='平均|SHAP值|',
        hovermode='x',
        showlegend=False
    )

    fig.show()


def plot_explanation_comparison(shap_values_1: shap.Explanation,
                               shap_values_2: shap.Explanation,
                               X: pd.DataFrame,
                               labels: List[str] = ["模型1", "模型2"],
                               save_path: Optional[str] = None):
    """
    比较两个模型的解释结果

    Args：
        shap_values_1: 第一个模型的SHAP值
        shap_values_2: 第二个模型的SHAP值
        X: 特征数据
        labels: 模型标签
        save_path: 保存路径
    """
    # 计算特征重要性
    if len(shap_values_1.values.shape) == 3:
        importance_1 = np.mean(np.abs(shap_values_1.values), axis=(0, 2))
        importance_2 = np.mean(np.abs(shap_values_2.values), axis=(0, 2))
    else:
        importance_1 = np.mean(np.abs(shap_values_1.values), axis=0)
        importance_2 = np.mean(np.abs(shap_values_2.values), axis=0)

    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 模型1
    indices_1 = np.argsort(importance_1)[::-1][:10]
    ax1.barh(range(10), importance_1[indices_1])
    ax1.set_yticks(range(10))
    ax1.set_yticklabels(X.columns[indices_1])
    ax1.set_xlabel('平均|SHAP值|')
    ax1.set_title(f'{labels[0]} - Top 10 特征重要性')
    ax1.invert_yaxis()

    # 模型2
    indices_2 = np.argsort(importance_2)[::-1][:10]
    ax2.barh(range(10), importance_2[indices_2])
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(X.columns[indices_2])
    ax2.set_xlabel('平均|SHAP值|')
    ax2.set_title(f'{labels[1]} - Top 10 特征重要性')
    ax2.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型对比图已保存至: {save_path}")

    plt.show()