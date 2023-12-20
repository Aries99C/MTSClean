import matplotlib.pyplot as plt
import numpy as np
from data_manager import DataManager
from data_utils import calculate_average_absolute_difference
from constraints import RowConstraintMiner, ColConstraintMiner
from cleaning_algorithms.MTSClean import MTSCleanRow
from cleaning_algorithms.SCREEN import LocalSpeedClean, GlobalSpeedClean, LocalSpeedAccelClean, GlobalSpeedAccelClean
from cleaning_algorithms.Smooth import EWMAClean, MedianFilterClean, KalmanFilterClean
# ... 导入其他所需模块 ...


def visualize_cleaning_results(data_manager):
    # 挖掘行约束和速度/加速度约束
    row_miner = RowConstraintMiner(data_manager.clean_data)
    row_constraints = row_miner.mine_row_constraints()

    col_miner = ColConstraintMiner(data_manager.clean_data)
    speed_constraints, accel_constraints = col_miner.mine_col_constraints()

    # 为 KalmanFilterClean 估计参数
    kalman_params = data_manager.estimate_kalman_parameters()

    # 定义并执行所有清洗算法
    algorithms = {
        'MTSCleanRow': MTSCleanRow(),
        'LocalSpeedClean': LocalSpeedClean(),
        'GlobalSpeedClean': GlobalSpeedClean(),
        'LocalSpeedAccelClean': LocalSpeedAccelClean(),
        'GlobalSpeedAccelClean': GlobalSpeedAccelClean(),
        'EWMAClean': EWMAClean(),
        'MedianFilterClean': MedianFilterClean(),
        'KalmanFilterClean': KalmanFilterClean(*kalman_params)
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
        elif name == 'KalmanFilterClean':
            cleaned_data = algo.clean(data_manager)
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

    # 可视化每段错误的观测值、正确值和清洗值
    error_indices = np.where(data_manager.error_mask)
    for i in error_indices[0]:
        plt.figure(figsize=(12, 6))
        plt.plot(data_manager.observed_data.iloc[i], label='Observed', color='gray', marker='o')
        plt.plot(data_manager.clean_data.iloc[i], label='True', color='green', marker='x')

        for name, cleaned_data in cleaned_results.items():
            plt.plot(cleaned_data.iloc[i], label=name, marker='^')  # 不同算法使用不同颜色和marker

        plt.title(f'Comparison of Cleaning Results for Error Segment {i}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 指定数据集的路径
    data_path = '../datasets/idf.csv'

    # 创建 DataManager 实例
    data_manager = DataManager(dataset='idf', dataset_path=data_path)

    # 注入错误到观测数据中
    data_manager.inject_errors(error_ratio=0.2, error_types=['drift', 'gaussian', 'volatility', 'gradual', 'sudden'])

    # 随机标记一定比例的数据为需要清洗的数据
    data_manager.randomly_label_data(0.1)

    # 调用可视化清洗结果对比的函数
    visualize_cleaning_results(data_manager)
