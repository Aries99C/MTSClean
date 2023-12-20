import matplotlib.pyplot as plt
import numpy as np
from data_manager import DataManager
from data_utils import calculate_average_absolute_difference
from constraints import RowConstraintMiner, ColConstraintMiner
from cleaning_algorithms.MTSClean import MTSCleanRow, MTSClean
from cleaning_algorithms.SCREEN import LocalSpeedClean, GlobalSpeedClean, LocalSpeedAccelClean, GlobalSpeedAccelClean
from cleaning_algorithms.Smooth import EWMAClean, MedianFilterClean, KalmanFilterClean
from cleaning_algorithms.IMR import IMRClean
# ... 导入其他所需模块 ...


def plot_segment(dm, col, start, end, buffer, cleaned_results):
    # 计算绘图范围
    plot_start = max(0, start - buffer)
    plot_end = min(len(dm.observed_data), end + buffer)

    # 绘制观测值、真实值
    plt.figure(figsize=(12, 6))
    plt.plot(dm.observed_data[col].iloc[plot_start:plot_end], label='Observed', color='gray', marker='o', linestyle='--')
    plt.plot(dm.clean_data[col].iloc[plot_start:plot_end], label='True', color='green', marker='x', linestyle='-.')

    # 为不同的清洗算法指定不同的标记
    markers = {'MTSClean': '^', 'GlobalSpeedAccelClean': 's', 'MedianFilterClean': 'd', 'IMRClean': '+'}
    colors = {'MTSClean': 'red', 'GlobalSpeedAccelClean': 'purple', 'MedianFilterClean': 'orange', 'IMRClean': 'blue'}

    for name, cleaned_data in cleaned_results.items():
        marker = markers.get(name, '^')
        color = colors.get(name, 'gray')
        plt.plot(cleaned_data[col].iloc[plot_start:plot_end], label=name, color=color, marker=marker)

    plt.title(f'Comparison of Cleaning Results for {col} [{start}:{end}]')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def plot_error_segments(dm, buffer, cleaned_results):
    for col in dm.error_mask.columns:
        error_locations = dm.error_mask[col]
        start = None
        for i in range(len(error_locations)):
            if error_locations[i] and start is None:
                start = i
            elif not error_locations[i] and start is not None:
                end = i
                plot_segment(dm, col, start, end, buffer, cleaned_results)
                start = None
        if start is not None:  # 处理最后一个错误段
            plot_segment(dm, col, start, len(error_locations), buffer, cleaned_results)


def visualize_cleaning_results(data_manager):
    # 挖掘行约束和速度/加速度约束
    row_miner = RowConstraintMiner(data_manager.clean_data)
    row_constraints, covered_attrs = row_miner.mine_row_constraints()

    col_miner = ColConstraintMiner(data_manager.clean_data)
    speed_constraints, accel_constraints = col_miner.mine_col_constraints()

    # 注入错误到观测数据中
    data_manager.inject_errors(error_ratio=0.25, error_types=['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=covered_attrs)

    # 为 KalmanFilterClean 估计参数
    kalman_params = data_manager.estimate_kalman_parameters()

    # 定义并执行所有清洗算法
    algorithms = {
        # 'MTSCleanRow': MTSCleanRow(),
        'MTSClean': MTSClean(),
        # 'LocalSpeedClean': LocalSpeedClean(),
        # 'GlobalSpeedClean': GlobalSpeedClean(),
        # 'LocalSpeedAccelClean': LocalSpeedAccelClean(),
        'GlobalSpeedAccelClean': GlobalSpeedAccelClean(),
        # 'EWMAClean': EWMAClean(),
        'MedianFilterClean': MedianFilterClean(),
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
        elif name == 'MTSClean':
            cleaned_data = algo.clean(data_manager, row_constraints=row_constraints, speed_constraints=speed_constraints)
        else:
            cleaned_data = algo.clean(data_manager)

        cleaned_results[name] = cleaned_data
        error = calculate_average_absolute_difference(cleaned_data, data_manager.clean_data)
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
    selected_algorithms = ['MTSClean', 'GlobalSpeedAccelClean', 'MedianFilterClean', 'IMRClean']
    selected_cleaned_results = {name: result for name, result in cleaned_results.items() if name in selected_algorithms}

    # 调用 plot_error_segments 来可视化每段错误数据
    plot_error_segments(data_manager, buffer=10, cleaned_results=selected_cleaned_results)


if __name__ == '__main__':
    # 指定数据集的路径
    data_path = '../datasets/idf.csv'

    # 创建 DataManager 实例
    data_manager = DataManager(dataset='idf', dataset_path=data_path)

    # 随机标记一定比例的数据为需要清洗的数据
    data_manager.randomly_label_data(0.05)

    # 调用可视化清洗结果对比的函数
    visualize_cleaning_results(data_manager)
