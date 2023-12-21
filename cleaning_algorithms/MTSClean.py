import data_utils
from cleaning_algorithms.base_algorithm import BaseCleaningAlgorithm
from data_manager import DataManager  # 确保DataManager类能够被正确导入
from constraints import RowConstraintMiner, ColConstraintMiner
import numpy as np
import pandas as pd
from scipy.optimize import linprog


class MTSCleanRow(BaseCleaningAlgorithm):
    def __init__(self):
        # 初始化可以放置一些必要的设置
        pass

    def clean(self, data_manager, **args):
        # 从args中获取constraints，如果没有提供，则生成或处理
        constraints = args.get('constraints')
        if constraints is None:
            # 如果未提供constraints，可以在这里生成默认约束
            # 或者返回一个错误消息，取决于您的需求
            miner = RowConstraintMiner(data_manager.clean_data)
            constraints, _ = miner.mine_row_constraints(attr_num=3)

        # 为 observed_data 的每行构建并求解线性规划问题
        n_rows, n_cols = data_manager.observed_data.shape
        cleaned_data = np.zeros_like(data_manager.observed_data)

        for row_idx in range(n_rows):
            row = data_manager.observed_data.iloc[row_idx, :]

            # 目标函数系数（最小化u和v的和）
            c = np.hstack([np.ones(n_cols), np.ones(n_cols)])  # 对每个x有两个变量u和v

            # 构建不等式约束
            A_ub = []
            b_ub = []
            for _, coefs, rho_min, rho_max in constraints:
                # 扩展系数以适应u和v
                extended_coefs = np.hstack([coefs, -coefs])

                # 添加两个不等式约束
                A_ub.append(extended_coefs)
                b_ub.append(rho_max - np.dot(coefs, row))
                A_ub.append(-extended_coefs)
                b_ub.append(-rho_min + np.dot(coefs, row))

            # 使用线性规划求解
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n_cols * 2)

            # 处理结果
            if result.success:
                cleaned_row = result.x[:n_cols] - result.x[n_cols:] + row  # x' = u - v + x
                cleaned_data[row_idx, :] = cleaned_row
            else:
                print(f"线性规划求解失败于行 {row_idx}: {result.message}")
                cleaned_data[row_idx, :] = row

        return pd.DataFrame(cleaned_data, columns=data_manager.observed_data.columns)

    @staticmethod
    def test_MTSCleanRow():
        # 加载数据
        data_path = '../datasets/idf.csv'

        # 初始化DataManager
        data_manager = DataManager('idf', data_path)

        # 使用 RowConstraintMiner 从 clean_data 中挖掘行约束
        miner = RowConstraintMiner(data_manager.clean_data)
        constraints, covered_attrs = miner.mine_row_constraints(attr_num=3)

        # 在DataManager中注入错误
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs)

        # 计算清洗前数据的平均绝对误差
        average_absolute_diff = data_utils.calculate_average_absolute_difference(data_manager.clean_data, data_manager.observed_data)
        # 输出结果
        print("清洗前数据的平均绝对误差：", average_absolute_diff)

        # 检查数据是否违反了行约束
        violations_count = data_utils.check_constraints_violations(data_manager.observed_data, constraints)
        print(f"观测数据违反的行约束次数：{violations_count}")

        # 创建MTSClean实例并清洗数据，传递constraints参数
        mtsclean = MTSCleanRow()
        cleaned_data = mtsclean.clean(data_manager, constraints=constraints)

        # 计算清洗后数据的平均绝对误差
        average_absolute_diff = data_utils.calculate_average_absolute_difference(cleaned_data, data_manager.clean_data)
        # 输出结果
        print("清洗后数据的平均绝对误差：", average_absolute_diff)

        # 检查数据是否违反了行约束
        violations_count = data_utils.check_constraints_violations(cleaned_data, constraints)
        print(f"清洗数据违反的行约束次数：{violations_count}")


class MTSClean(BaseCleaningAlgorithm):
    def __init__(self):
        pass

    def clean(self, data_manager, **args):
        # 从args中获取行约束和速度约束
        row_constraints = args.get('row_constraints')
        if row_constraints is None:
            miner = RowConstraintMiner(data_manager.clean_data)
            row_constraints, _ = miner.mine_row_constraints(attr_num=3)

        for row_constraint in row_constraints:
            print(row_constraint[0])

        speed_constraints = args.get('speed_constraints')
        if speed_constraints is None:
            raise ValueError("Speed constraints are required for secondary cleaning.")

        # 首先利用行约束进行清洗
        cleaned_data = self._clean_with_row_constraints(data_manager.observed_data, row_constraints)

        # 然后利用速度约束进行二次清洗
        cleaned_data = self._clean_with_speed_constraints(cleaned_data, speed_constraints)

        return cleaned_data

    def _clean_with_row_constraints(self, data, constraints):
        n_rows, n_cols = data.shape
        cleaned_data = np.zeros_like(data)

        for row_idx in range(n_rows):
            row = data.iloc[row_idx, :]

            c = np.hstack([np.ones(n_cols), np.ones(n_cols)])
            A_ub = []
            b_ub = []
            for _, coefs, rho_min, rho_max in constraints:
                extended_coefs = np.hstack([coefs, -coefs])
                A_ub.append(extended_coefs)
                b_ub.append(rho_max - np.dot(coefs, row))
                A_ub.append(-extended_coefs)
                b_ub.append(-rho_min + np.dot(coefs, row))

            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * n_cols * 2)

            if result.success:
                cleaned_row = result.x[:n_cols] - result.x[n_cols:] + row
                cleaned_data[row_idx, :] = cleaned_row
            else:
                cleaned_data[row_idx, :] = row

        return pd.DataFrame(cleaned_data, columns=data.columns)

    def _clean_with_speed_constraints(self, data, speed_constraints):
        n_rows = data.shape[0]
        cleaned_data = data.copy()

        chunk_length = 50
        overlap = 10

        for col in data.columns:
            speed_lb, speed_ub = speed_constraints[col]
            x = data[col].values

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

                        A_row = np.zeros(len(chunk) * 2)
                        A_row[i] = -1
                        A_row[j] = 1
                        A_row[len(chunk) + i] = 1
                        A_row[len(chunk) + j] = -1
                        A_ub.append(A_row)
                        b_ub.append(speed_ub * row_diff - (chunk[j] - chunk[i]))

                        A_row = np.zeros(len(chunk) * 2)
                        A_row[i] = 1
                        A_row[j] = -1
                        A_row[len(chunk) + i] = -1
                        A_row[len(chunk) + j] = 1
                        A_ub.append(A_row)
                        b_ub.append(-speed_lb * row_diff + (chunk[j] - chunk[i]))

                bounds = [(0, None) for _ in range(len(chunk) * 2)]
                result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

                if result.success:
                    cleaned_chunk = result.x[:len(chunk)] - result.x[len(chunk):] + chunk
                    cleaned_data[col].iloc[start:end] = cleaned_chunk
                else:
                    print(f"线性规划求解失败于列 {col}, 数据块 {start}-{end}: {result.message}")

        return cleaned_data

    @staticmethod
    def test_MTSClean():
        # 加载数据
        data_path = '../datasets/idf.csv'

        # 初始化DataManager
        data_manager = DataManager('idf', data_path)

        # 使用 RowConstraintMiner 从 clean_data 中挖掘行约束
        row_miner = RowConstraintMiner(data_manager.clean_data)
        row_constraints, covered_attrs = row_miner.mine_row_constraints(attr_num=3)

        # 使用 ColConstraintMiner 从 clean_data 中挖掘速度约束
        col_miner = ColConstraintMiner(data_manager.clean_data)
        speed_constraints, _ = col_miner.mine_col_constraints()

        # 在DataManager中注入错误
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs)

        # 计算清洗前数据的平均绝对误差
        average_absolute_diff_before = data_utils.calculate_average_absolute_difference(data_manager.clean_data,
                                                                                        data_manager.observed_data)
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        # 检查数据是否违反了行约束
        violations_count = data_utils.check_constraints_violations(data_manager.observed_data, row_constraints)
        print(f"观测数据违反的行约束次数：{violations_count}")

        # 创建MTSClean实例并清洗数据，传递行约束和速度约束参数
        mtsclean = MTSClean()
        cleaned_data = mtsclean.clean(data_manager, row_constraints=row_constraints, speed_constraints=speed_constraints)

        # 计算清洗后数据的平均绝对误差
        average_absolute_diff_after = data_utils.calculate_average_absolute_difference(cleaned_data,
                                                                                       data_manager.clean_data)
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        # 检查数据是否违反了行约束
        violations_count = data_utils.check_constraints_violations(cleaned_data, row_constraints)
        print(f"清洗数据违反的行约束次数：{violations_count}")


# 在适当的时候调用测试函数
if __name__ == "__main__":
    # MTSCleanRow.test_MTSCleanRow()
    MTSClean.test_MTSClean()
