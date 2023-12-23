from tqdm import tqdm

from cleaning_algorithms.base_algorithm import BaseCleaningAlgorithm
from data_manager import DataManager
from constraints import ColConstraintMiner
import numpy as np
import pandas as pd
import data_utils
from scipy.optimize import linprog


class LocalSpeedClean(BaseCleaningAlgorithm):
    def clean(self, data_manager, **args):
        observed_data = data_manager.observed_data
        speed_constraints = args.get('speed_constraints')

        if speed_constraints is None:
            raise ValueError("Speed constraints are required for local speed cleaning.")

        w = args.get('window', 10)  # 获取窗口长度

        # 创建清洗后的数据副本
        cleaned_data = observed_data.copy()

        for col in tqdm(cleaned_data.columns, desc="Local Speed Cleaning"):
            # 获取当前列的速度约束上下界
            speed_lb = speed_constraints[col][0]
            speed_ub = speed_constraints[col][1]
            data = observed_data[col].values

            for i in range(1, len(data) - 1):
                x_i_min = speed_lb + data[i - 1]
                x_i_max = speed_ub + data[i - 1]
                candidate_i = [data[i]]
                for k in range(i + 1, len(data)):
                    if k > i + w:
                        break
                    candidate_i.append(speed_lb + data[k])
                    candidate_i.append(speed_ub + data[k])
                candidate_i = np.array(candidate_i)
                x_i_mid = np.median(candidate_i)
                if x_i_mid < x_i_min:
                    cleaned_data.at[i, col] = x_i_min
                elif x_i_mid > x_i_max:
                    cleaned_data.at[i, col] = x_i_max

        return cleaned_data

    @staticmethod
    def test_LocalSpeedClean():
        # 加载数据
        data_path = '../datasets/idf.csv'

        # 初始化DataManager
        data_manager = DataManager('idf', data_path)

        # 在DataManager中注入错误
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=data_manager.clean_data.columns)

        # 计算清洗前数据的平均绝对误差
        average_absolute_diff_before = data_utils.calculate_average_absolute_difference(data_manager.clean_data, data_manager.observed_data)
        # 输出结果
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        # 使用 ColConstraintMiner 从 clean_data 中挖掘列约束
        miner = ColConstraintMiner(data_manager.clean_data)
        speed_constraints = miner.mine_col_constraints()[0]  # 获取速度约束

        # 创建LocalSpeedClean实例并清洗数据，传递speed_constraints参数
        localspeedclean = LocalSpeedClean()
        cleaned_data = localspeedclean.clean(data_manager, speed_constraints=speed_constraints)

        # 计算清洗后数据的平均绝对误差
        average_absolute_diff_after = data_utils.calculate_average_absolute_difference(cleaned_data, data_manager.clean_data)
        # 输出结果
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)


class GlobalSpeedClean(BaseCleaningAlgorithm):
    def clean(self, data_manager, **args):
        observed_data = data_manager.observed_data
        speed_constraints = args.get('speed_constraints')
        chunk_length = args.get('chunk_length', 30)
        overlap = args.get('overlap', 10)  # 重叠部分长度

        if speed_constraints is None:
            raise ValueError("Speed constraints are required for global speed cleaning.")

        cleaned_data = observed_data.copy()

        for col in tqdm(observed_data.columns, desc="Global Speed Cleaning"):
            speed_lb, speed_ub = speed_constraints[col]
            x = observed_data[col].values
            n_rows = len(x)

            for start in range(0, n_rows, chunk_length - overlap):
                end = min(start + chunk_length, n_rows)
                chunk = x[start:end]

                # 构建线性规划的目标函数和约束
                c = np.ones(len(chunk) * 2)
                A_ub = []
                b_ub = []
                for i in range(len(chunk) - 1):
                    for j in range(i + 1, len(chunk)):
                        row_diff = j - i

                        # 速度上界约束
                        A_row = np.zeros(len(chunk) * 2)
                        A_row[i] = -1
                        A_row[j] = 1
                        A_row[len(chunk) + i] = 1
                        A_row[len(chunk) + j] = -1
                        A_ub.append(A_row)
                        b_ub.append(speed_ub * row_diff - (chunk[j] - chunk[i]))

                        # 速度下界约束
                        A_row = np.zeros(len(chunk) * 2)
                        A_row[i] = 1
                        A_row[j] = -1
                        A_row[len(chunk) + i] = -1
                        A_row[len(chunk) + j] = 1
                        A_ub.append(A_row)
                        b_ub.append(-speed_lb * row_diff + (chunk[j] - chunk[i]))

                # 使用线性规划求解
                bounds = [(0, None) for _ in range(len(chunk) * 2)]
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

                # 处理结果
                if result.success:
                    cleaned_chunk = result.x[:len(chunk)] - result.x[len(chunk):] + chunk
                    cleaned_data[col].iloc[start:end] = cleaned_chunk
                else:
                    print(f"线性规划求解失败于列 {col}, 数据块 {start}-{end}: {result.message}")

        return cleaned_data

    @staticmethod
    def test_GlobalSpeedClean():
        # 加载数据
        data_path = '../datasets/idf.csv'

        # 初始化DataManager
        data_manager = DataManager('idf', data_path)

        # 在DataManager中注入错误
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=data_manager.clean_data.columns)

        # 计算清洗前数据的平均绝对误差
        average_absolute_diff_before = data_utils.calculate_average_absolute_difference(data_manager.clean_data, data_manager.observed_data)
        # 输出结果
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        # 使用 ColConstraintMiner 从 clean_data 中挖掘列约束
        miner = ColConstraintMiner(data_manager.clean_data)
        speed_constraints = miner.mine_col_constraints()[0]  # 获取速度约束

        # 创建GlobalSpeedClean实例并清洗数据，传递speed_constraints参数
        globalspeedclean = GlobalSpeedClean()
        cleaned_data = globalspeedclean.clean(data_manager, speed_constraints=speed_constraints)

        # 计算清洗后数据的平均绝对误差
        average_absolute_diff_after = data_utils.calculate_average_absolute_difference(cleaned_data, data_manager.clean_data)
        # 输出结果
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)


class LocalSpeedAccelClean(BaseCleaningAlgorithm):
    def clean(self, data_manager, **args):
        observed_data = data_manager.observed_data
        speed_constraints = args.get('speed_constraints')
        accel_constraints = args.get('accel_constraints')
        w = args.get('window', 10)

        if speed_constraints is None or accel_constraints is None:
            raise ValueError("Both speed and acceleration constraints are required for local speed-accel cleaning.")

        cleaned_data = observed_data.copy()

        for col in tqdm(cleaned_data.columns, desc="Local Speed-Accel Cleaning"):
            speed_lb, speed_ub = speed_constraints[col]
            acc_lb, acc_ub = accel_constraints[col]
            data = observed_data[col].values

            for k in range(2, len(data) - 1):
                x_k_min = max(speed_lb + data[k - 1], acc_lb + data[k - 1] - data[k - 2] + data[k - 1])
                x_k_max = min(speed_ub + data[k - 1], acc_ub + data[k - 1] - data[k - 2] + data[k - 1])
                candidate_k = [data[k]]
                candidate_k_min = []
                candidate_k_max = []

                for i in range(k + 1, len(data)):
                    if i > k + w:
                        break
                    z_k_i_a_min = (data[k - 1] * (i - k) - (acc_ub * ((i - k) ** 2) - data[i])) / (i - k + 1)
                    z_k_i_a_max = (data[k - 1] * (i - k) - (acc_lb * ((i - k) ** 2) - data[i])) / (i - k + 1)
                    z_k_i_s_min = data[i] - speed_ub * (i - k)
                    z_k_i_s_max = data[i] - speed_lb * (i - k)
                    candidate_k_min.append(min(z_k_i_s_min, z_k_i_a_min))
                    candidate_k_max.append(max(z_k_i_s_max, z_k_i_a_max))

                candidate_k.extend(candidate_k_min)
                candidate_k.extend(candidate_k_max)
                candidate_k = np.array(candidate_k)
                x_k_mid = np.median(candidate_k)

                if x_k_mid < x_k_min:
                    cleaned_data.at[k, col] = x_k_min
                elif x_k_mid > x_k_max:
                    cleaned_data.at[k, col] = x_k_max

        return cleaned_data

    @staticmethod
    def test_LocalSpeedAccelClean():
        # 加载数据
        data_path = '../datasets/idf.csv'

        # 初始化DataManager
        data_manager = DataManager('idf', data_path)

        # 在DataManager中注入错误
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=data_manager.clean_data.columns)

        # 计算清洗前数据的平均绝对误差
        average_absolute_diff_before = data_utils.calculate_average_absolute_difference(data_manager.clean_data,
                                                                                        data_manager.observed_data)
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        # 使用 ColConstraintMiner 从 clean_data 中挖掘列约束
        miner = ColConstraintMiner(data_manager.clean_data)
        speed_constraints, accel_constraints = miner.mine_col_constraints()  # 获取速度和加速度约束

        # 创建LocalSpeedAccelClean实例并清洗数据，传递速度和加速度约束参数
        localspeedaccelclean = LocalSpeedAccelClean()
        cleaned_data = localspeedaccelclean.clean(data_manager, speed_constraints=speed_constraints,
                                                  accel_constraints=accel_constraints)

        # 计算清洗后数据的平均绝对误差
        average_absolute_diff_after = data_utils.calculate_average_absolute_difference(cleaned_data,
                                                                                       data_manager.clean_data)
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)


class GlobalSpeedAccelClean(BaseCleaningAlgorithm):
    def clean(self, data_manager, **args):
        observed_data = data_manager.observed_data
        speed_constraints = args.get('speed_constraints')
        accel_constraints = args.get('accel_constraints')
        chunk_length = args.get('chunk_length', 30)
        overlap = args.get('overlap', 10)

        if speed_constraints is None or accel_constraints is None:
            raise ValueError("Both speed and acceleration constraints are required for global speed-accel cleaning.")

        cleaned_data = observed_data.copy()

        for col in tqdm(observed_data.columns, desc="Global Speed-Accel Cleaning"):
            speed_lb, speed_ub = speed_constraints[col]
            acc_lb, acc_ub = accel_constraints[col]
            x = observed_data[col].values
            n_rows = len(x)

            for start in range(0, n_rows, chunk_length - overlap):
                end = min(start + chunk_length, n_rows)
                chunk = x[start:end]

                c = np.ones(2 * len(chunk))
                A_ub = []
                b_ub = []
                bounds = [(0, None) for _ in range(2 * len(chunk))]

                for i in range(len(chunk)):
                    for j in range(i + 1, len(chunk)):

                        bij_max = speed_ub * (j - i) - (chunk[j] - chunk[i])
                        bij_min = -speed_lb * (j - i) + (chunk[j] - chunk[i])

                        # 速度约束
                        A_row_max = np.zeros(2 * len(chunk))
                        A_row_min = np.zeros(2 * len(chunk))
                        A_row_max[j], A_row_max[i] = 1, -1
                        A_row_max[j + len(chunk)], A_row_max[i + len(chunk)] = -1, 1
                        A_row_min[j], A_row_min[i] = -1, 1
                        A_row_min[j + len(chunk)], A_row_min[i + len(chunk)] = 1, -1
                        A_ub.append(A_row_max)
                        b_ub.append(bij_max)
                        A_ub.append(A_row_min)
                        b_ub.append(bij_min)

                        # 加速度约束
                        if i >= 1:
                            bij_max = acc_ub * (j - i) - (chunk[j] - chunk[i]) / (j - i) + (chunk[i] - chunk[i - 1])
                            bij_min = -acc_lb * (j - i) + (chunk[j] - chunk[i]) / (j - i) - (chunk[i] - chunk[i - 1])
                            A_row_max = np.zeros(2 * len(chunk))
                            A_row_min = np.zeros(2 * len(chunk))
                            tmp1, tmp2 = 1 / (j - i), 1
                            A_row_max[j], A_row_max[j + len(chunk)] = tmp1, -tmp1
                            A_row_max[i], A_row_max[i + len(chunk)] = -tmp1 - tmp2, tmp1 + tmp2
                            A_row_max[i - 1], A_row_max[i - 1 + len(chunk)] = tmp2, -tmp2
                            A_row_min[j], A_row_min[j + len(chunk)] = -tmp1, tmp1
                            A_row_min[i], A_row_min[i + len(chunk)] = tmp1 + tmp2, -tmp1 - tmp2
                            A_row_min[i - 1], A_row_min[i - 1 + len(chunk)] = -tmp2, tmp2
                            A_ub.append(A_row_max)
                            b_ub.append(bij_max)
                            A_ub.append(A_row_min)
                            b_ub.append(bij_min)

                result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
                if result.success:
                    cleaned_chunk = result.x[:len(chunk)] - result.x[len(chunk):] + chunk
                    cleaned_data[col].iloc[start:end] = cleaned_chunk
                else:
                    print(f"线性规划求解失败于列 {col}, 数据块 {start}-{end}: {result.message}")

        return cleaned_data

    @staticmethod
    def test_GlobalSpeedAccelClean():
        # 加载数据
        data_path = '../datasets/idf.csv'

        # 初始化DataManager
        data_manager = DataManager('idf', data_path)

        # 在DataManager中注入错误
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=data_manager.clean_data.columns)

        # 计算清洗前数据的平均绝对误差
        average_absolute_diff_before = data_utils.calculate_average_absolute_difference(data_manager.clean_data,
                                                                                        data_manager.observed_data)
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        # 使用 ColConstraintMiner 从 clean_data 中挖掘列约束
        miner = ColConstraintMiner(data_manager.clean_data)
        speed_constraints, accel_constraints = miner.mine_col_constraints()

        # 创建GlobalSpeedAccelClean实例并清洗数据，传递速度和加速度约束参数
        globalspeedaccelclean = GlobalSpeedAccelClean()
        cleaned_data = globalspeedaccelclean.clean(data_manager, speed_constraints=speed_constraints,
                                                   accel_constraints=accel_constraints)

        # 计算清洗后数据的平均绝对误差
        average_absolute_diff_after = data_utils.calculate_average_absolute_difference(cleaned_data,
                                                                                       data_manager.clean_data)
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)


if __name__ == "__main__":
    LocalSpeedClean.test_LocalSpeedClean()
    GlobalSpeedClean.test_GlobalSpeedClean()
    LocalSpeedAccelClean.test_LocalSpeedAccelClean()
    GlobalSpeedAccelClean.test_GlobalSpeedAccelClean()
