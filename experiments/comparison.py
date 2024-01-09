import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import messagebox
from data_manager import DataManager
from data_utils import calculate_average_absolute_difference
from constraints import RowConstraintMiner, ColConstraintMiner
from cleaning_algorithms.MTSClean import MTSCleanRow, MTSClean, MTSCleanSoft
from cleaning_algorithms.SCREEN import LocalSpeedClean, GlobalSpeedClean, LocalSpeedAccelClean, GlobalSpeedAccelClean
from cleaning_algorithms.Smooth import EWMAClean, MedianFilterClean, KalmanFilterClean
from cleaning_algorithms.IMR import IMRClean
# ... 导入其他所需模块 ...


def prompt_save_result(dm, cleaned_results, col, start, end, buffer, examples_dir, row_constraints):
    # 创建一个Tkinter窗口
    root = tk.Tk()
    root.withdraw()  # 不显示主窗口

    # 显示保存询问对话框
    save_response = messagebox.askyesno("保存可视化结果", "是否需要保存这段错误的可视化结果?")
    if save_response:
        save_segment_to_csv(dm, cleaned_results, col, start, end, buffer, examples_dir, row_constraints)

    # 显示继续查看询问对话框
    continue_response = messagebox.askyesno("继续查看", "是否继续查看下一个错误段的可视化结果?")
    if not continue_response:
        root.destroy()
        return False  # 停止继续查看

    root.destroy()
    return True  # 继续查看


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
    combined_lower_bound = []
    combined_upper_bound = []

    for index in range(plot_start, plot_end):
        min_bounds = []
        max_bounds = []
        for _, coefs, rho_min, rho_max in row_constraints:
            if coefs[col_index] != 0:  # 只处理涉及当前列的约束
                min_val, max_val = calculate_correct_value_range(dm, coefs, rho_min, rho_max, col_index, index)
                min_bounds.append(min_val)
                max_bounds.append(max_val)
        combined_lower_bound.append(max(min_bounds))  # 取最大的下界
        combined_upper_bound.append(min(max_bounds))  # 取最小的上界

    ax.fill_between(range(plot_start, plot_end), combined_lower_bound, combined_upper_bound, color='yellow', alpha=0.3,
                    label='Correct Value Range')

    ax.set_title(f'Comparison of Cleaning Results for {col} [{start}:{end}]')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()

    return fig, ax


def calculate_correct_value_range(dm, coefs, rho_min, rho_max, col_index, index):
    other_values_sum = sum(dm.observed_data.iloc[index, i] * coefs[i] for i in range(len(coefs)) if i != col_index)
    target_coef = coefs[col_index]
    if target_coef != 0:
        min_value = (rho_min - other_values_sum) / target_coef
        max_value = (rho_max - other_values_sum) / target_coef
        return (min_value, max_value) if min_value <= max_value else (max_value, min_value)
    return (0, 30)


def calculate_error_on_segments(cleaned_data, dm):
    total_error = 0
    total_elements = 0

    for col in dm.error_mask.columns:
        error_indices = dm.error_mask[col]
        start = None
        for i in range(len(error_indices)):
            if error_indices[i] and start is None:
                start = i
            elif not error_indices[i] and start is not None:
                end = i
                total_error += np.sum(np.abs(cleaned_data[col][start:end] - dm.clean_data[col][start:end]))
                total_elements += end - start
                start = None
        if start is not None:  # 处理最后一个错误段
            end = len(error_indices)
            total_error += np.sum(np.abs(cleaned_data[col][start:end] - dm.clean_data[col][start:end]))
            total_elements += end - start

    return total_error / total_elements if total_elements > 0 else 0


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
                if should_visualize_segment(dm, cleaned_results, col, start, end):
                    fig, ax = plot_segment(dm, col, start, end, buffer, cleaned_results, row_constraints)
                    plt.show()  # 显示图表

                    if not prompt_save_result(dm, cleaned_results, col, start, end, buffer, examples_dir, row_constraints):
                        return  # 用户选择停止继续查看

                start = None

        if start is not None:
            end = len(error_locations)
            if should_visualize_segment(dm, cleaned_results, col, start, end):
                fig, ax = plot_segment(dm, col, start, end, buffer, cleaned_results, row_constraints)
                plt.show()  # 显示图表

                if not prompt_save_result(dm, cleaned_results, col, start, end, buffer, examples_dir, row_constraints):
                    return  # 用户选择停止继续查看


def visualize_cleaning_results(data_manager):
    # 挖掘行约束和速度/加速度约束
    row_miner = RowConstraintMiner(data_manager.clean_data)
    row_constraints, covered_attrs = row_miner.mine_row_constraints()

    # for row_constraint in row_constraints:
    #     print(row_constraint[0])

    col_miner = ColConstraintMiner(data_manager.clean_data)
    speed_constraints, accel_constraints = col_miner.mine_col_constraints()

    # 注入错误到观测数据中
    data_manager.inject_errors(error_ratio=0.2, error_types=['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=covered_attrs, row_constraints=row_constraints)

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
        # 'GlobalSpeedAccelClean': GlobalSpeedAccelClean(),
        # 'EWMAClean': EWMAClean(),
        # 'MedianFilterClean': MedianFilterClean(),
        # 'KalmanFilterClean': KalmanFilterClean(*kalman_params),
        'IMRClean': IMRClean()
        # ... 添加其他清洗算法 ...
    }

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
        # 使用新函数计算每个清洗算法在错误段上的平均绝对误差
        error = calculate_error_on_segments(cleaned_data, data_manager)
        errors.append(error)

    # 可视化平均绝对误差
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))
    plt.bar(algorithms.keys(), errors, color=colors)
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


if __name__ == '__main__':
    # 指定数据集的路径
    data_path = '../datasets/idf.csv'

    # 创建 DataManager 实例
    data_manager = DataManager(dataset='idf', dataset_path=data_path)

    # 随机标记一定比例的数据为需要清洗的数据
    data_manager.randomly_label_data(0.05)

    # 调用可视化清洗结果对比的函数
    visualize_cleaning_results(data_manager)
