import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from constraints import RowConstraintMiner


class DataManager:
    def __init__(self, dataset, dataset_path, label_rate=0.1):
        """
        初始化数据管理器。
        :param dataset: 数据库的名称。
        :param dataset_path: 数据集的文件路径。
        """
        self.dataset = dataset
        self.clean_data = pd.read_csv(dataset_path)
        self.scale_factors = {}  # 用于存储每列的缩放因子和最小值
        self._adjust_scale()
        self.observed_data = self.clean_data.copy()
        self.error_mask = pd.DataFrame(False, index=self.clean_data.index, columns=self.clean_data.columns)
        self.is_label = pd.DataFrame(False, index=self.observed_data.index, columns=self.observed_data.columns)  # 初始化is_label
        self.randomly_label_data(label_rate)  # 随机标记指定比例的数据

    def estimate_kalman_parameters(self):
        # 针对每列估计参数
        initial_state = {}
        observation_covariance = {}
        transition_covariance = {}
        transition_matrices = {}

        for col in self.clean_data.columns:
            initial_state[col] = self.clean_data[col].iloc[0]
            var_col = np.var(self.clean_data[col])
            observation_covariance[col] = var_col if var_col > 0 else 1
            transition_covariance[col] = var_col / 2 if var_col > 0 else 0.5
            transition_matrices[col] = 1  # 单变量情况下，转换矩阵是标量 1

        return initial_state, observation_covariance, transition_covariance, transition_matrices

    def randomly_label_data(self, error_rate):
        # 确保error_rate在合理范围
        if not 0 <= error_rate <= 1:
            raise ValueError("Error rate must be between 0 and 1.")

        n_rows, n_cols = self.observed_data.shape
        total_elements = n_rows * n_cols
        n_errors = int(total_elements * error_rate)

        # 首先标记前10行为True
        self.is_label.iloc[:10, :] = True
        marked_elements = 10 * n_cols

        # 随机选择剩余要标记的数据单元
        remaining_elements = total_elements - marked_elements
        additional_errors = n_errors - marked_elements
        if additional_errors > 0:
            error_indices = np.random.choice(remaining_elements, additional_errors, replace=False)
            for idx in error_indices:
                # 调整索引以跳过已标记的前10行
                adjusted_idx = idx + marked_elements
                row, col = divmod(adjusted_idx, n_cols)
                self.is_label.iloc[row, col] = True

    def _adjust_scale(self):
        """
        调整数据的量纲，使每列的均值约为10，且没有负数。
        """
        for col in self.clean_data.columns:
            min_val = self.clean_data[col].min()
            self.clean_data[col] -= min_val
            mean_val = self.clean_data[col].mean()
            scale_factor = 10 / mean_val
            self.clean_data[col] *= scale_factor
            self.scale_factors[col] = (min_val, scale_factor)

    def restore_original_scale(self, data):
        """
        将数据复原到原始的量纲。
        :param data: 要复原的数据。
        :return: 复原后的数据。
        """
        restored_data = data.copy()
        for col in restored_data.columns:
            min_val, scale_factor = self.scale_factors[col]
            restored_data[col] = (restored_data[col] / scale_factor) + min_val
        return restored_data

    def inject_errors(self, error_ratio, error_types, covered_attrs=None, row_constraints=None):
        n_rows, _ = self.clean_data.shape
        total_errors = int(n_rows * error_ratio)

        while total_errors > 0:
            error_length = random.randint(30, 50)
            if total_errors < error_length:
                error_length = total_errors

            start_row = random.randint(100, n_rows - error_length)

            if row_constraints:
                # 从行约束中随机选择一个，并筛选出系数绝对值大于等于0.6的属性
                _, coefs, _, _ = random.choice(row_constraints)
                involved_cols = [col for col, coef in zip(self.clean_data.columns, coefs) if abs(coef) >= 0.6]
                if not involved_cols:  # 如果没有符合条件的列，则选择下一个行约束
                    continue
            else:
                # 如果没有行约束，从 covered_attrs 中选择列
                involved_cols = covered_attrs

            selected_col = random.choice(involved_cols)
            if self.error_mask.iloc[start_row:start_row + error_length][selected_col].any():
                continue

            error_type = random.choice(error_types)
            self._inject_error(start_row, error_length, [selected_col], error_type)
            total_errors -= error_length

        # 更新复原后的数据
        self.restored_clean_data = self.restore_original_scale(self.clean_data)
        self.restored_observed_data = self.restore_original_scale(self.observed_data)

    def _inject_error(self, start_row, length, col, error_type):
        if error_type == 'drift':
            self._inject_drift_error(start_row, length, col)
        elif error_type == 'gaussian':
            self._inject_gaussian_error(start_row, length, col)
        elif error_type == 'volatility':
            self._inject_volatility_error(start_row, length, col)
        elif error_type == 'gradual':
            self._inject_gradual_error(start_row, length, col)
        elif error_type == 'sudden':
            self._inject_sudden_error(start_row, length, col)

    def _inject_drift_error(self, start_row, length, cols):
        for col in cols:
            # 随机选择漂移值的范围
            if random.choice([True, False]):
                drift_value = random.uniform(-4, -2)
            else:
                drift_value = random.uniform(2, 4)

            temp_values = self.observed_data.loc[start_row:start_row+length-1, col] + drift_value
            # 确保数据在0到30的范围内
            self.observed_data.loc[start_row:start_row+length-1, col] = np.clip(temp_values, 0, 30)
            self.error_mask.loc[start_row:start_row+length-1, col] = True

    def _inject_gaussian_error(self, start_row, length, cols):
        snr = 20  # 信噪比为20dB
        for col in cols:
            signal_power = np.mean(self.clean_data[col] ** 2)
            noise_power = signal_power / (10 ** (snr / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), length)
            temp_values = self.observed_data.loc[start_row:start_row+length-1, col] + noise
            # 确保数据在0到30的范围内
            self.observed_data.loc[start_row:start_row+length-1, col] = np.clip(temp_values, 0, 30)
            self.error_mask.loc[start_row:start_row+length-1, col] = True

    def _inject_volatility_error(self, start_row, length, cols):
        for col in cols:
            # 生成波动因子向量
            volatility_factors = np.random.uniform(0.7, 1.3, length)
            original_values = self.clean_data.loc[start_row:start_row+length-1, col]
            temp_values = original_values * volatility_factors
            # 确保数据在0到30的范围内
            self.observed_data.loc[start_row:start_row+length-1, col] = np.clip(temp_values, 0, 30)
            self.error_mask.loc[start_row:start_row+length-1, col] = True

    def _inject_gradual_error(self, start_row, length, cols):
        for col in cols:
            # 随机选择错误的增减方向和最终幅度
            direction = random.choice([-1, 1])
            magnitude = random.uniform(5, 10) * direction
            # 创建一个渐变的错误向量
            gradual_change = np.linspace(0, magnitude, length)
            # 在错误的最后一行进行迅速恢复
            gradual_change[-1] = 0

            temp_values = self.observed_data.loc[start_row:start_row + length - 1, col] + gradual_change
            self.observed_data.loc[start_row:start_row + length - 1, col] = np.clip(temp_values, 0, 30)
            self.error_mask.loc[start_row:start_row + length - 1, col] = True

    def _inject_sudden_error(self, start_row, length, cols):
        for col in cols:
            # 随机选择错误的增减方向和最终幅度
            direction = random.choice([-1, 1])
            magnitude = random.uniform(5, 10) * direction
            # 创建一个突变的错误向量
            sudden_change = np.full(length, magnitude)
            # 计算恢复阶段的长度
            recovery_length = length - (length // 2)
            # 在错误段进行逐渐恢复
            sudden_change[-recovery_length:] = np.linspace(magnitude, 0, recovery_length)

            temp_values = self.observed_data.loc[start_row:start_row + length - 1, col] + sudden_change
            self.observed_data.loc[start_row:start_row + length - 1, col] = np.clip(temp_values, 0, 30)
            self.error_mask.loc[start_row:start_row + length - 1, col] = True


def calculate_value_range_for_constraint(row_data, constraint, target_col_index):
    coefs, rho_min, rho_max = constraint
    sum_other = np.dot(np.concatenate((coefs[:target_col_index], coefs[target_col_index + 1:])),
                       np.concatenate((row_data[:target_col_index], row_data[target_col_index + 1:])))
    if coefs[target_col_index] != 0:
        min_val = (rho_min - sum_other) / coefs[target_col_index]
        max_val = (rho_max - sum_other) / coefs[target_col_index]

        min_val = max(min_val, 0)
        max_val = min(max_val, 30)
        if min_val < max_val:
            return max(min_val, 0), min(max_val, 30)  # 确保范围在合理区间
        else:
            return max(max_val, 0), min(min_val, 30)  # 确保范围在合理区间
    return 0, 30  # 如果当前列系数为0，则允许整个范围


def find_common_range(dm, col, row_constraints, start, end):
    col_index = dm.clean_data.columns.get_loc(col)
    common_min, common_max = 0, 30
    for constraint in row_constraints:
        if constraint[0][col_index] != 0:  # 检查是否涉及当前列
            for idx in range(start, end):
                row_data = dm.clean_data.iloc[idx].values
                min_val, max_val = calculate_value_range_for_constraint(row_data, constraint, col_index)
                common_min = max(common_min, min_val)
                common_max = min(common_max, max_val)
    return common_min, common_max


def plot_error_segments(dm, buffer=10, row_constraints=[]):
    """
    遍历并绘制每列上每段错误数据及其前后缓冲区的数据。
    :param dm: DataManager实例。
    :param buffer: 在错误数据前后额外包含的行数。
    """
    for col in dm.error_mask.columns:
        # 寻找错误段
        error_locations = dm.error_mask[col]
        start = None
        for i in range(len(error_locations)):
            if error_locations[i] and start is None:
                start = i
            elif not error_locations[i] and start is not None:
                end = i
                plot_segment(dm, col, start, end, buffer, row_constraints)  # 传递行约束参数
                start = None
        if start is not None:  # 处理最后一个错误段
            plot_segment(dm, col, start, end, buffer, row_constraints)  # 传递行约束参数


def plot_segment(dm, col, start, end, buffer, row_constraints):
    plot_start = max(0, start - buffer)
    plot_end = min(len(dm.clean_data), end + buffer)

    common_min, common_max = find_common_range(dm, col, row_constraints, plot_start, plot_end)

    plt.figure(figsize=(10, 4))
    plt.plot(dm.clean_data[col][plot_start:plot_end], label='Clean Data', color='blue')
    plt.plot(dm.observed_data[col][plot_start:plot_end], label='Observed Data', color='orange')
    plt.fill_between(range(plot_start, plot_end), common_min, common_max, color='yellow', alpha=0.3, label='Allowed Range')
    plt.axvline(x=start, color='green', linestyle='--', label='Error Start')
    plt.axvline(x=end - 1, color='red', linestyle='--', label='Error End')
    plt.title(f'Error Segment in Column {col}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dm = DataManager('idf', 'datasets/idf.csv')

    # 使用 RowConstraintMiner 挖掘行约束
    row_miner = RowConstraintMiner(dm.clean_data)
    row_constraints, _ = row_miner.mine_row_constraints(attr_num=3)

    # for constraint in row_constraints:
    #     print(constraint[0])

    dm.inject_errors(0.1, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], row_constraints=row_constraints)

    # 调整行约束格式（去除字符串描述，只保留系数和范围）
    formatted_row_constraints = [(constraint[1], constraint[2], constraint[3]) for constraint in row_constraints]

    # 绘制每个错误段数据
    plot_error_segments(dm, buffer=10, row_constraints=formatted_row_constraints)
