import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import warnings
import tkinter as tk

from sklearn.metrics import f1_score

from data_manager import DataManager
from data_utils import calculate_average_absolute_difference
from constraints import RowConstraintMiner, ColConstraintMiner
from cleaning_algorithms.MTSClean import MTSCleanRow, MTSClean, MTSCleanSoft
from cleaning_algorithms.SCREEN import LocalSpeedClean, GlobalSpeedClean, LocalSpeedAccelClean, GlobalSpeedAccelClean
from cleaning_algorithms.Smooth import EWMAClean, MedianFilterClean, KalmanFilterClean
from cleaning_algorithms.IMR import IMRClean
from tkinter import messagebox


warnings.filterwarnings("ignore")


def plot_segment(dm, col, start, end, buffer, cleaned_results, row_constraints):
    plot_start = max(0, start - buffer)
    plot_end = min(len(dm.observed_data), end + buffer)
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制观测值和真实值
    ax.plot(dm.observed_data[col].iloc[plot_start:plot_end], label='Observed', color='gray', marker='o', linestyle='--')
    ax.plot(dm.clean_data[col].iloc[plot_start:plot_end], label='True', color='green', marker='x', linestyle='-.')

    # 绘制清洗算法的结果
    markers = {'MTSClean': '^', 'MTSCleanSoft': '*', 'GlobalSpeedAccelClean': 's', 'MedianFilterClean': 'd',
               'IMRClean': '+'}
    colors = {'MTSClean': 'red', 'MTSCleanSoft': 'pink', 'GlobalSpeedAccelClean': 'purple',
              'MedianFilterClean': 'orange', 'IMRClean': 'blue'}
    for name, cleaned_data in cleaned_results.items():
        ax.plot(cleaned_data[col].iloc[plot_start:plot_end], label=name, marker=markers[name], color=colors[name])

    # 获取当前列的索引
    col_index = dm.observed_data.columns.get_loc(col)

    # 计算并绘制正确值的取值范围
    for _, coefs, rho_min, rho_max in row_constraints:
        if coefs[col_index] != 0:  # 判断当前行约束是否与col有关
            lower_bound, upper_bound = calculate_correct_value_range(dm, coefs, rho_min, rho_max, col_index, plot_start, plot_end)
            ax.fill_between(range(plot_start, plot_end), lower_bound, upper_bound, color='yellow', alpha=0.3, label='Correct Value Range')

    ax.set_title(f'Comparison of Cleaning Results for {col} [{start}:{end}]')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()

    return fig, ax


def calculate_correct_value_range(dm, coefs, rho_min, rho_max, col_index, start, end):
    lower_bound = []
    upper_bound = []

    for index in range(start, end):
        # 计算除目标列外的其他列的加权和
        other_values_sum = sum(dm.observed_data.iloc[index, i] * coefs[i] for i in range(len(coefs)) if i != col_index)

        # 计算当前行的取值范围
        target_coef = coefs[col_index]
        if target_coef != 0:  # 避免除以零
            min_value = (rho_min - other_values_sum) / target_coef
            max_value = (rho_max - other_values_sum) / target_coef
            lower_bound.append(min_value)
            upper_bound.append(max_value)
        else:
            lower_bound.append(None)
            upper_bound.append(None)

    return lower_bound, upper_bound


def should_visualize_segment(dm, cleaned_results, col, start, end):
    mtsclean_error = calculate_segment_error(cleaned_results['MTSClean'][col], dm.clean_data[col], start, end)

    for name, cleaned_data in cleaned_results.items():
        if name != 'MTSClean':
            other_error = calculate_segment_error(cleaned_data[col], dm.clean_data[col], start, end)
            if other_error < mtsclean_error:
                return False

    return True


def calculate_segment_error(cleaned_data, true_data, start, end):
    return np.mean(np.abs(cleaned_data[start:end] - true_data[start:end]))


def save_segment_to_csv(dm, cleaned_results, col, start, end, buffer, dir_path, row_constraints):
    plot_start = max(0, start - buffer)
    plot_end = min(len(dm.observed_data), end + buffer)
    segment_df = pd.DataFrame({
        'Observed': dm.observed_data[col].iloc[plot_start:plot_end],
        'True': dm.clean_data[col].iloc[plot_start:plot_end]
    })

    for name, cleaned_data in cleaned_results.items():
        segment_df[name] = cleaned_data[col].iloc[plot_start:plot_end]

    segment_csv_file = f'{dir_path}/{col}_segment_{start}_{end}.csv'
    segment_df.to_csv(segment_csv_file)

    # 保存图像
    fig, ax = plot_segment(dm, col, start, end, buffer, cleaned_results, row_constraints)
    segment_image_file = f'{dir_path}/{col}_segment_{start}_{end}.png'
    fig.savefig(segment_image_file)
    plt.close(fig)  # 关闭图表对象


def plot_error_segments(dm, buffer, cleaned_results, row_constraints):
    examples_dir = 'examples'
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)

    for col in dm.error_mask.columns:
        error_locations = dm.error_mask[col]
        start = None
        for i in range(len(error_locations)):
            if error_locations[i] and start is None:
                start = i
            elif not error_locations[i] and start is not None:
                end = i
                # if should_visualize_segment(dm, cleaned_results, col, start, end):
                if True:
                    fig, ax = plot_segment(dm, col, start, end, buffer, cleaned_results, row_constraints)
                    plt.show()  # 显示图表

                    root = tk.Tk()
                    root.withdraw()  # 不显示主窗口

                    # 弹出对话框询问是否保存
                    save = messagebox.askyesno("保存", "是否需要保存这段错误的可视化结果?")
                    if save:
                        save_segment_to_csv(dm, cleaned_results, col, start, end, buffer, examples_dir, row_constraints)

                    # 弹出对话框询问是否继续
                    continue_plot = messagebox.askyesno("继续", "是否继续查看下一个错误段的可视化结果?")
                    if not continue_plot:
                        return  # 提前结束函数，不再处理后续错误段

                start = None

        if start is not None:
            end = len(error_locations)
            # if should_visualize_segment(dm, cleaned_results, col, start, end):
            if True:
                fig, ax = plot_segment(dm, col, start, end, buffer, cleaned_results, row_constraints)
                plt.show()  # 显示图表

                root = tk.Tk()
                root.withdraw()  # 不显示主窗口

                save = messagebox.askyesno("保存", "是否需要保存这段错误的可视化结果?")
                if save:
                    save_segment_to_csv(dm, cleaned_results, col, start, end, buffer, examples_dir, row_constraints)

                continue_plot = messagebox.askyesno("继续", "是否继续查看下一个错误段的可视化结果?")
                if not continue_plot:
                    return  # 提前结束函数，不再处理后续错误段


def calculate_error_segment_difference(cleaned_data, correct_data, error_mask):
    total_diff = 0
    # 遍历每个元素
    for i in range(error_mask.shape[0]):
        for j in range(error_mask.shape[1]):
            if error_mask.iat[i, j]:  # 检查是否是错误位置
                diff = abs(cleaned_data.iat[i, j] - correct_data.iat[i, j])
                total_diff += diff
    return total_diff / np.prod(cleaned_data.shape)


def visualize_cleaning_results(data_manager):
    # 挖掘行约束和速度/加速度约束
    row_miner = RowConstraintMiner(data_manager.clean_data)
    row_constraints, covered_attrs = row_miner.mine_row_constraints()

    for row_constraint in row_constraints:
        print(row_constraint[0])

    col_miner = ColConstraintMiner(data_manager.clean_data)
    speed_constraints, accel_constraints = col_miner.mine_col_constraints()

    # 注入错误到观测数据中
    # data_manager.inject_errors(error_ratio=0.25, error_types=['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=covered_attrs)
    data_manager.inject_errors(error_ratio=0.25, error_types=['drift'], covered_attrs=covered_attrs)

    # 为 KalmanFilterClean 估计参数
    kalman_params = data_manager.estimate_kalman_parameters()

    # 定义并执行所有清洗算法
    algorithms = {
        # 'MTSCleanRow': MTSCleanRow(),
        'MTSClean': MTSClean(),
        'MTSCleanSoft': MTSCleanSoft(),
        # 'LocalSpeedClean': LocalSpeedClean(),
        # 'GlobalSpeedClean': GlobalSpeedClean(),
        # 'LocalSpeedAccelClean': LocalSpeedAccelClean(),
        'GlobalSpeedAccelClean': GlobalSpeedAccelClean(),
        # 'EWMAClean': EWMAClean(),
        # 'MedianFilterClean': MedianFilterClean(),
        # 'KalmanFilterClean': KalmanFilterClean(*kalman_params),
        'IMRClean': IMRClean()
        # ... 添加其他清洗算法 ...
    }

    names = []
    errors = []
    cleaned_results = {}

    for name, algo in algorithms.items():
        if name in ['LocalSpeedClean', 'GlobalSpeedClean', 'LocalSpeedAccelClean', 'GlobalSpeedAccelClean']:
            cleaned_data = algo.clean(data_manager, speed_constraints=speed_constraints,
                                      accel_constraints=accel_constraints)
        elif name == 'MTSCleanRow':
            cleaned_data = algo.clean(data_manager, row_constraints=row_constraints)
        elif name in ['MTSClean', 'MTSCleanSoft']:
            cleaned_data = algo.clean(data_manager, row_constraints=row_constraints, speed_constraints=speed_constraints)
        else:
            cleaned_data = algo.clean(data_manager)

        cleaned_results[name] = cleaned_data
        error = calculate_error_segment_difference(cleaned_data, data_manager.clean_data, data_manager.error_mask)
        names.append(name)
        errors.append(error)

    # 可视化平均绝对误差
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))
    plt.bar(names, errors, color=colors)
    plt.xlabel('Cleaning Algorithms')
    plt.ylabel('Average Absolute Error')
    plt.title('Comparison of Cleaning Algorithms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 只选择特定的算法进行可视化
    selected_algorithms = ['MTSClean', 'MTSCleanSoft', 'GlobalSpeedAccelClean', 'MedianFilterClean', 'IMRClean']
    selected_cleaned_results = {name: result for name, result in cleaned_results.items() if name in selected_algorithms}

    # 调用 plot_error_segments 来可视化每段错误数据
    plot_error_segments(data_manager, buffer=10, cleaned_results=selected_cleaned_results, row_constraints=row_constraints)


# 计算 RRA
def calculate_rra(cleaned_data, observed_data, correct_data, error_mask):
    """
    计算相对修复精度（RRA）。

    :param cleaned_data: DataFrame, 清洗后的数据
    :param observed_data: DataFrame, 观测数据
    :param correct_data: DataFrame, 正确数据
    :param error_mask: DataFrame, 错误掩码，标记数据中的错误位置
    :return: RRA值
    """
    # 计算清洗后数据与正确数据之间的差的绝对值之和
    error_cleaned = calculate_error_segment_difference(cleaned_data, correct_data, error_mask)
    # 计算观测数据与正确数据之间的差的绝对值之和
    error_observed = calculate_error_segment_difference(observed_data, correct_data, error_mask)
    # 计算清洗后数据与观测数据之间的差的绝对值之和
    error_cleaned_observed = calculate_error_segment_difference(cleaned_data, observed_data, error_mask)
    # 计算RRA
    return 1 - error_cleaned / (error_observed + error_cleaned_observed)


# 计算 F1 分数
def calculate_f1_score(cleaned_data, observed_data, error_mask):
    detected_errors = (cleaned_data != observed_data).any(axis=1)
    return f1_score(error_mask.any(axis=1), detected_errors)


def check_row_violations(data, row_constraints):
    """检查每行数据是否违反行约束。

    :param data: DataFrame，数据集
    :param row_constraints: list，行约束列表，每个约束包含相关列的系数、最小值和最大值
    :return: 违反约束的行数
    """
    violation_count = 0
    for index, row in data.iterrows():
        for _, coefs, rho_min, rho_max in row_constraints:
            value = np.dot(coefs, row)
            if value < rho_min or value > rho_max:
                violation_count += 1
                break  # 只要违反任何一个约束即停止检查当前行
    return violation_count


def calculate_constraint_violation_ratio(cleaned_data, observed_data, row_constraints):
    """计算违反约束的行数比例。

    :param cleaned_data: DataFrame，清洗后的数据
    :param observed_data: DataFrame，观测数据
    :param row_constraints: list，行约束列表
    :return: 违反约束的行数比例
    """
    violation_count_cleaned = check_row_violations(cleaned_data, row_constraints)
    violation_count_observed = check_row_violations(observed_data, row_constraints)
    if violation_count_observed == 0:  # 防止除零错误
        return 0
    return violation_count_cleaned / violation_count_observed


def evaluate_cleaning_algorithms_by_segment_length(data_manager):
    # 挖掘行约束和速度/加速度约束
    row_miner = RowConstraintMiner(data_manager.clean_data)
    row_constraints, covered_attrs = row_miner.mine_row_constraints()

    col_miner = ColConstraintMiner(data_manager.clean_data)
    speed_constraints, accel_constraints = col_miner.mine_col_constraints()

    # 为 KalmanFilterClean 估计参数
    kalman_params = data_manager.estimate_kalman_parameters()

    # 定义清洗算法
    algorithms = {
        'MTSClean': MTSClean(),
        'MTSCleanSoft': MTSCleanSoft(),
        # 'LocalSpeedClean': LocalSpeedClean(),
        # 'GlobalSpeedClean': GlobalSpeedClean(),
        # 'LocalSpeedAccelClean': LocalSpeedAccelClean(),
        # 'GlobalSpeedAccelClean': GlobalSpeedAccelClean(),
        # 'EWMAClean': EWMAClean(),
        # 'MedianFilterClean': MedianFilterClean(),
        # 'KalmanFilterClean': KalmanFilterClean(*kalman_params),
        # 'IMRClean': IMRClean()
    }

    # 分段长度比例
    # segment_ratios = [1 / 5, 2 / 5, 3 / 5, 4 / 5, 1]
    segment_ratios = [2 / 5]

    # 为每个算法和每个指标初始化字典
    error_by_algorithm = {name: [] for name in algorithms.keys()}
    rra_by_algorithm = {name: [] for name in algorithms.keys()}
    f1_by_algorithm = {name: [] for name in algorithms.keys()}
    runtime_by_algorithm = {name: [] for name in algorithms.keys()}
    violation_ratio_by_algorithm = {name: [] for name in algorithms.keys()}

    for ratio in segment_ratios:
        dm_copy = copy.deepcopy(data_manager)
        segment_length = int(len(dm_copy.observed_data) * ratio)
        dm_copy.observed_data = dm_copy.observed_data.iloc[:segment_length]
        dm_copy.clean_data = dm_copy.clean_data.iloc[:segment_length]
        dm_copy.error_mask = dm_copy.error_mask.iloc[:segment_length]

        # 在每次迭代中注入错误
        dm_copy.inject_errors(error_ratio=0.25, error_types=['drift'], covered_attrs=covered_attrs)

        for name, algo in algorithms.items():
            start_time = time.time()
            cleaned_data = algo.clean(dm_copy, row_constraints=row_constraints, speed_constraints=speed_constraints, accel_constraints=accel_constraints)
            runtime = time.time() - start_time

            error = calculate_error_segment_difference(cleaned_data, dm_copy.clean_data, dm_copy.error_mask)
            rra = calculate_rra(cleaned_data, dm_copy.observed_data, dm_copy.clean_data, dm_copy.error_mask)
            f1 = calculate_f1_score(cleaned_data, dm_copy.observed_data, dm_copy.error_mask)
            violation_ratio = calculate_constraint_violation_ratio(cleaned_data, dm_copy.observed_data, row_constraints)

            error_by_algorithm[name].append(error)
            rra_by_algorithm[name].append(rra)
            f1_by_algorithm[name].append(f1)
            runtime_by_algorithm[name].append(runtime)
            violation_ratio_by_algorithm[name].append(violation_ratio)

    # 为每个指标绘制直方图
    plot_histograms(error_by_algorithm, segment_ratios, 'L1 Error', data_manager.dataset)
    plot_histograms(rra_by_algorithm, segment_ratios, 'Relative Repair Accuracy', data_manager.dataset)
    plot_histograms(f1_by_algorithm, segment_ratios, 'F1 Score', data_manager.dataset)
    plot_histograms(runtime_by_algorithm, segment_ratios, 'Runtime', data_manager.dataset)
    plot_histograms(violation_ratio_by_algorithm, segment_ratios, 'Constraint Violation Ratio', data_manager.dataset)

    # 保存结果到CSV文件
    save_results_to_csv(error_by_algorithm, segment_ratios, f'{data_manager.dataset}_l1_by_segment_length.csv')
    save_results_to_csv(rra_by_algorithm, segment_ratios, f'{data_manager.dataset}_rra_by_segment_length.csv')
    save_results_to_csv(f1_by_algorithm, segment_ratios, f'{data_manager.dataset}_f1_by_segment_length.csv')
    save_results_to_csv(runtime_by_algorithm, segment_ratios, f'{data_manager.dataset}_runtime_by_segment_length.csv')
    save_results_to_csv(violation_ratio_by_algorithm, segment_ratios, f'{data_manager.dataset}_violation_ratio_by_segment_length.csv')


def evaluate_cleaning_algorithms_by_error_ratio(data_manager):
    # 挖掘行约束和速度/加速度约束
    row_miner = RowConstraintMiner(data_manager.clean_data)
    row_constraints, covered_attrs = row_miner.mine_row_constraints()

    col_miner = ColConstraintMiner(data_manager.clean_data)
    speed_constraints, accel_constraints = col_miner.mine_col_constraints()

    # 为 KalmanFilterClean 估计参数
    kalman_params = data_manager.estimate_kalman_parameters()

    # 定义清洗算法
    algorithms = {
        'MTSClean': MTSClean(),
        'MTSCleanSoft': MTSCleanSoft(),
        'LocalSpeedClean': LocalSpeedClean(),
        'GlobalSpeedClean': GlobalSpeedClean(),
        'LocalSpeedAccelClean': LocalSpeedAccelClean(),
        'GlobalSpeedAccelClean': GlobalSpeedAccelClean(),
        'EWMAClean': EWMAClean(),
        'MedianFilterClean': MedianFilterClean(),
        'KalmanFilterClean': KalmanFilterClean(*kalman_params),
        'IMRClean': IMRClean()
    }

    # 错误注入比例
    error_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

    # 为每个算法和每个指标初始化字典
    error_by_algorithm = {name: [] for name in algorithms.keys()}
    rra_by_algorithm = {name: [] for name in algorithms.keys()}
    f1_by_algorithm = {name: [] for name in algorithms.keys()}
    runtime_by_algorithm = {name: [] for name in algorithms.keys()}
    violation_ratio_by_algorithm = {name: [] for name in algorithms.keys()}

    for error_ratio in error_ratios:
        dm_copy = copy.deepcopy(data_manager)

        dm_copy.observed_data = dm_copy.observed_data.iloc[:4000]
        dm_copy.clean_data = dm_copy.clean_data.iloc[:4000]
        dm_copy.error_mask = dm_copy.error_mask.iloc[:4000]

        # 在每次迭代中注入错误
        dm_copy.inject_errors(error_ratio=error_ratio, error_types=['drift'], covered_attrs=covered_attrs)

        for name, algo in algorithms.items():
            start_time = time.time()
            cleaned_data = algo.clean(dm_copy, row_constraints=row_constraints, speed_constraints=speed_constraints, accel_constraints=accel_constraints)
            runtime = time.time() - start_time

            error = calculate_error_segment_difference(cleaned_data, dm_copy.clean_data, dm_copy.error_mask)
            rra = calculate_rra(cleaned_data, dm_copy.observed_data, dm_copy.clean_data, dm_copy.error_mask)
            f1 = calculate_f1_score(cleaned_data, dm_copy.observed_data, dm_copy.error_mask)
            violation_ratio = calculate_constraint_violation_ratio(cleaned_data, dm_copy.observed_data, row_constraints)

            error_by_algorithm[name].append(error)
            rra_by_algorithm[name].append(rra)
            f1_by_algorithm[name].append(f1)
            runtime_by_algorithm[name].append(runtime)
            violation_ratio_by_algorithm[name].append(violation_ratio)

    # 为每个指标绘制直方图
    plot_histograms(error_by_algorithm, error_ratios, 'L1 Error', data_manager.dataset)
    plot_histograms(rra_by_algorithm, error_ratios, 'Relative Repair Accuracy', data_manager.dataset)
    plot_histograms(f1_by_algorithm, error_ratios, 'F1 Score', data_manager.dataset)
    plot_histograms(runtime_by_algorithm, error_ratios, 'Runtime', data_manager.dataset)
    plot_histograms(violation_ratio_by_algorithm, error_ratios, 'Constraint Violation Ratio', data_manager.dataset)

    # 保存结果到CSV文件
    save_results_to_csv(error_by_algorithm, error_ratios, f'{data_manager.dataset}_l1_by_error_ratio.csv')
    save_results_to_csv(rra_by_algorithm, error_ratios, f'{data_manager.dataset}_rra_by_error_ratio.csv')
    save_results_to_csv(f1_by_algorithm, error_ratios, f'{data_manager.dataset}_f1_by_error_ratio.csv')
    save_results_to_csv(runtime_by_algorithm, error_ratios, f'{data_manager.dataset}_runtime_by_error_ratio.csv')
    save_results_to_csv(violation_ratio_by_algorithm, error_ratios, f'{data_manager.dataset}_violation_ratio_by_error_ratio.csv')


def plot_histograms(data, ratios, title, dataset_name):
    """
    为不同算法的指标绘制直方图。

    :param data: dict, 各算法的指标数据
    :param ratios: list, 数据段长度比例列表
    :param title: str, 图表标题
    :param dataset_name: str, 数据集名称
    """
    n_ratios = len(ratios)
    n_algorithms = len(data)
    bar_width = 1 / (n_algorithms + 1)  # 计算每个条形的宽度

    plt.figure(figsize=(12, 6))

    # 为每个算法和每个比例绘制条形
    for i, (name, values) in enumerate(data.items()):
        # 计算每个条形的位置
        positions = np.arange(n_ratios) + bar_width * i

        plt.bar(positions, values, width=bar_width, label=name)

    # 设置图表的标题和标签
    plt.title(f'{title} for Different Algorithms on {dataset_name}')
    plt.xlabel('Segment Length Ratio')
    plt.ylabel(title)

    # 设置x轴刻度标签
    plt.xticks(np.arange(n_ratios) + bar_width * (n_algorithms / 2), ratios)

    plt.legend()
    plt.show()


def save_results_to_csv(data, segment_ratios, filename):
    """
    将结果保存到CSV文件。

    :param data: dict, 每个算法在不同段长比例下的指标数据
    :param segment_ratios: list, 数据段长度比例列表
    :param filename: str, 保存结果的文件名
    """
    df = pd.DataFrame(data, index=segment_ratios)
    df.to_csv(filename)


if __name__ == '__main__':
    # 指定数据集的路径
    data_path = '../datasets/SWaT.csv'

    # 创建 DataManager 实例
    data_manager = DataManager(dataset='test', dataset_path=data_path)

    # 随机标记一定比例的数据为需要清洗的数据
    data_manager.randomly_label_data(0.05)

    # 调用可视化清洗结果对比的函数
    # visualize_cleaning_results(data_manager)

    # 调用新的函数
    evaluate_cleaning_algorithms_by_segment_length(data_manager)
    # evaluate_cleaning_algorithms_by_error_ratio(data_manager)


