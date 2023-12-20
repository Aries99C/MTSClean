import data_utils
from cleaning_algorithms.base_algorithm import BaseCleaningAlgorithm
from data_manager import DataManager  # 确保DataManager类能够被正确导入
from constraints import RowConstraintMiner
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
            constraints = miner.mine_row_constraints(attr_num=3)

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

        # 在DataManager中注入错误
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'])

        # 计算清洗前数据的平均绝对误差
        average_absolute_diff = data_utils.calculate_average_absolute_difference(data_manager.clean_data, data_manager.observed_data)
        # 输出结果
        print("清洗前数据的平均绝对误差：", average_absolute_diff)

        # 使用 RowConstraintMiner 从 clean_data 中挖掘行约束
        miner = RowConstraintMiner(data_manager.clean_data)
        constraints = miner.mining_row_constraints(attr_num=3)

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


# 在适当的时候调用测试函数
if __name__ == "__main__":
    MTSCleanRow.test_MTSCleanRow()
