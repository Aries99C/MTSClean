import numpy as np
import pandas as pd
import hypernetx as hnx
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from constraints import RowConstraintMiner, ColConstraintMiner
from data_manager import DataManager


def check_constraints_violations(data, constraints):
    violations_count = 0

    for index, row in data.iterrows():
        for _, coefs, rho_min, rho_max in constraints:
            value = np.dot(coefs, row)
            if not (rho_min <= value <= rho_max):
                violations_count += 1
                break  # 如果这一行已经违反了某个约束，就不再检查其他约束

    return violations_count


def calculate_average_absolute_difference(cleaned_data, observed_data):
    total_diff = np.abs(cleaned_data.values - observed_data.values).sum()
    n_elements = np.prod(observed_data.shape)  # 数据集中的元素总数
    average_diff = total_diff / n_elements
    return average_diff


def create_hypergraph(data, row_constraints, speed_constraints, start_index, window):
    # 准备一个字典来构建超图
    scenes = {}

    # 考虑窗口内的数据
    window_data = data.iloc[start_index:start_index + window]

    # # 添加行约束为超边
    # for i, row_constraint in enumerate(row_constraints):
    #     # 仅考虑窗口内的行
    #     for j in range(start_index, start_index + window - 1):
    #         hyperedge = [(j, col) for coef_val, col in zip(row_constraint[1], window_data.columns) if coef_val != 0]
    #         print(hyperedge)
    #         scenes[f'Row_{i}_{j}'] = set(hyperedge)
    #
    # # 添加速度约束为超边
    # for col in window_data.columns:
    #     hyperedge = [(i, col) for i in range(start_index, start_index + window - 1)]
    #     scenes[f'Speed_{col}'] = set(hyperedge)

    # 添加行约束为超边
    for i, row_constraint in enumerate(row_constraints):
        # 仅考虑窗口内的行
        for j in range(start_index, start_index + 1):
            hyperedge = [(j, col) for coef_val, col in zip(row_constraint[1], window_data.columns) if coef_val != 0]
            scenes[f'Row_{i}_{j}'] = set(hyperedge)

    # 添加速度约束为超边
    # for col in window_data.columns:
    #     hyperedge = [(i, col) for i in range(start_index, start_index + window - 1)]
    #     scenes[f'Speed_{col}'] = set(hyperedge)

    # 创建超图
    H = hnx.Hypergraph(scenes)
    return H


def compute_row_violation(window_data, row_constraint):
    _, coefs, rho_min, rho_max = row_constraint
    violation = 0
    for _, row in window_data.iterrows():
        value = np.dot(coefs, row)
        violation = max(violation, max(0, value - rho_max, rho_min - value))
    return violation


def compute_speed_violation(window_data, speed_constraint, col):
    speed_lb, speed_ub = speed_constraint
    violation = 0
    for i in range(len(window_data) - 1):
        speed = window_data[col].iloc[i + 1] - window_data[col].iloc[i]
        violation = max(violation, max(0, speed - speed_ub, speed_lb - speed))
    return violation


def draw_hypergraph(hypergraph, window_data, row_constraints, speed_constraints):
    violation_degrees = {}
    num_colors = len(row_constraints)
    cmap = cm.get_cmap('tab10', num_colors)  # 获取调色板

    # 计算行约束和速度约束的违反程度
    for i, row_constraint in enumerate(row_constraints):
        violation = compute_row_violation(window_data, row_constraint)
        violation_degrees[f'Row_{i}'] = violation

    for col in window_data.columns:
        violation = compute_speed_violation(window_data, speed_constraints[col], col)
        violation_degrees[f'Speed_{col}'] = violation

    # 设置边的粗细和颜色
    edge_colors = {}
    edge_widths = {}
    for label in hypergraph.edges:
        linewidth = 1 + 5 * violation_degrees.get(label, 0)
        edge_widths[label] = linewidth
        if label.startswith('Row'):
            color_index = int(label.split('_')[1]) % num_colors
            color = cmap.colors[color_index]
        else:
            color = 'black'
        edge_colors[label] = color

    # 绘制超图，不显示边和顶点的标签
    hnx.draw(hypergraph, label_alpha=0, edges_kwargs={
        'facecolors': 'none',
        'edgecolors': [edge_colors[label] for label in hypergraph.edges],
        'linewidths': [edge_widths[label] for label in hypergraph.edges]
    }, with_edge_labels=False, with_node_labels=False)

    plt.show()  # 展示图像


def test_hypergraph_on_idf_dataset():
    # 初始化DataManager并加载数据
    data_manager = DataManager('idf', 'datasets/idf.csv')

    # 使用RowConstraintMiner和ColConstraintMiner挖掘约束
    row_miner = RowConstraintMiner(data_manager.clean_data)
    row_constraints, covered_attrs = row_miner.mine_row_constraints(attr_num=3)

    col_miner = ColConstraintMiner(data_manager.clean_data)
    speed_constraints, _ = col_miner.mine_col_constraints()

    # 在DataManager中注入错误
    data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], row_constraints=row_constraints)

    # 从DataManager获取观测数据
    observed_data = data_manager.observed_data

    # 定义窗口的开始位置和大小
    start_index = 1000  # 例如从第1000行开始
    window_size = 3   # 例如窗口大小为5行

    # 创建超图
    H = create_hypergraph(observed_data, row_constraints, speed_constraints, start_index, window_size)

    # 获取窗口内的数据
    window_data = observed_data.iloc[start_index:start_index + window_size]

    # 绘制超图
    draw_hypergraph(H, window_data, row_constraints, speed_constraints)


if __name__ == '__main__':
    test_hypergraph_on_idf_dataset()

