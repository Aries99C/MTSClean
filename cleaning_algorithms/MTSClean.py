import time

import data_utils
from cleaning_algorithms.base_algorithm import BaseCleaningAlgorithm
from data_manager import DataManager  # 确保DataManager类能够被正确导入
from constraints import RowConstraintMiner, ColConstraintMiner
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.optimize import minimize
import random
from deap import base, creator, tools, algorithms
# import geatpy as ea
from tqdm import tqdm
from multiprocessing import Pool

import cProfile
import pstats


profiler = cProfile.Profile()
profiler.enable()


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

    def _check_row_violations(self, row, constraints):
        """
        检查行是否违反约束。

        :param row: Series, 行数据
        :param constraints: list, 约束条件列表
        :return: bool, 是否违反约束
        """
        for _, coefs, rho_min, rho_max in constraints:
            value = np.dot(coefs, row)
            if value < rho_min or value > rho_max:
                return True  # 违反约束
        return False  # 未违反任何约束

    def clean(self, data_manager, **args):
        # 从args中获取行约束和速度约束
        row_constraints = args.get('row_constraints')
        if row_constraints is None:
            miner = RowConstraintMiner(data_manager.clean_data)
            row_constraints, _ = miner.mine_row_constraints(attr_num=3)

        speed_constraints = args.get('speed_constraints')
        if speed_constraints is None:
            raise ValueError("Speed constraints are required for secondary cleaning.")

        # 计算总的迭代次数
        total_steps = data_manager.observed_data.shape[0] * 5 + data_manager.observed_data.shape[0] * len(data_manager.observed_data.columns)
        # total_steps = data_manager.observed_data.shape[0]
        with tqdm(total=total_steps, desc="MTSClean Cleaning") as pbar:
            # 先进行行约束清洗
            for _ in range(4):
                cleaned_data = self._clean_with_row_constraints(data_manager.observed_data, row_constraints, pbar)
            # 再进行速度约束清洗
            cleaned_data = self._clean_with_speed_constraints(cleaned_data, speed_constraints, pbar)

        return cleaned_data

    def _clean_with_row_constraints(self, data, constraints, pbar):
        n_rows, n_cols = data.shape
        cleaned_data = np.zeros_like(data)

        for row_idx in range(n_rows):
            row = data.iloc[row_idx, :]

            # if not self._check_row_violations(row, constraints):
            #     pbar.update(1)  # 更新进度条
            #     continue

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

            pbar.update(1)  # 更新进度条

        return pd.DataFrame(cleaned_data, columns=data.columns)

    def _clean_with_speed_constraints(self, data, speed_constraints, pbar):
        """
        使用速度约束对数据进行清洗。

        :param data: DataFrame, 观测数据
        :param speed_constraints: dict, 速度约束
        :param pbar: tqdm, 进度条对象
        :return: DataFrame, 清洗后的数据
        """
        w = 10  # 窗口长度，您可以根据需要调整这个值
        cleaned_data = data.copy()

        for col in cleaned_data.columns:
            speed_lb, speed_ub = speed_constraints[col]
            data_col = data[col].values

            for i in range(1, len(data_col) - 1):
                x_i_min = speed_lb + data_col[i - 1]
                x_i_max = speed_ub + data_col[i - 1]
                candidate_i = [data_col[i]]
                for k in range(i + 1, min(i + w + 1, len(data_col))):
                    candidate_i.append(speed_lb + data_col[k - 1])
                    candidate_i.append(speed_ub + data_col[k - 1])
                candidate_i = np.array(candidate_i)
                x_i_mid = np.median(candidate_i)
                if x_i_mid < x_i_min:
                    cleaned_data.at[i, col] = x_i_min
                elif x_i_mid > x_i_max:
                    cleaned_data.at[i, col] = x_i_max

                pbar.update(1)  # 更新进度条

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

        for row_constraint in row_constraints:
            print(row_constraint[0])

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


class MTSCleanPareto(BaseCleaningAlgorithm):
    def __init__(self, num_generations=50, pop_size=100):
        self.num_generations = num_generations
        self.pop_size = pop_size
        self.row_constraints = None

    def clean(self, data_manager, **args):
        self.row_constraints = args.get('row_constraints')
        if self.row_constraints is None:
            raise ValueError("Row constraints are required for MTSCleanPareto.")

        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * (len(self.row_constraints) + 1))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        # 打开进程池
        pool = Pool()
        tasks = [(row,) for _, row in data_manager.observed_data.iterrows()]
        results = list(tqdm(pool.imap(self._optimize_row, tasks), total=len(tasks)))

        pool.close()
        pool.join()

        # 将并行结果合并到一个DataFrame中
        cleaned_data = pd.concat(results, axis=1).T
        cleaned_data.columns = data_manager.observed_data.columns

        return cleaned_data

    def _optimize_row(self, task):
        # 在每个子进程中重新创建所需的 DEAP 类
        if not hasattr(creator, "Individual"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * (len(self.row_constraints) + 1))
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        row, = task
        optimized_row = self._pareto_optimization(row, self.row_constraints)
        return pd.Series(optimized_row)

    def _pareto_optimization(self, row_data, constraints):
        toolbox = base.Toolbox()
        toolbox.register("individual", self._init_individual, row_data)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_individual, row_data=row_data, constraints=constraints)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", tools.selNSGA2)

        population = toolbox.population(n=self.pop_size)
        algorithms.eaMuPlusLambda(population, toolbox, mu=self.pop_size, lambda_=self.pop_size,
                                  cxpb=0.5, mutpb=0.2, ngen=self.num_generations, verbose=False)

        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        best_individual = pareto_front[0]

        return best_individual

    def _init_individual(self, row_data):
        individual = row_data.tolist()
        return creator.Individual(individual)

    def _evaluate_individual(self, individual, row_data, constraints):
        # 计算行约束违反程度
        constraint_violations = tuple(
            self._calculate_constraint_violation(individual, constraint) for constraint in constraints)

        # 计算与原始数据的绝对误差
        absolute_error = sum(abs(ind - obs) for ind, obs in zip(individual, row_data))

        # 将行约束违反程度和绝对误差组合为一个元组作为适应度
        fitness = constraint_violations + (absolute_error,)

        return fitness

    def _calculate_constraint_violation(self, individual, constraint):
        _, coefs, rho_min, rho_max = constraint
        value = sum(coef * ind for coef, ind in zip(coefs, individual))
        if value < rho_min:
            return abs(rho_min - value)
        elif value > rho_max:
            return abs(value - rho_max)
        return 0.0

    @staticmethod
    def test_MTSCleanPareto():
        # 加载数据
        data_path = '../datasets/idf.csv'

        # 初始化DataManager
        data_manager = DataManager('idf', data_path)

        # 使用 RowConstraintMiner 从 clean_data 中挖掘行约束
        row_miner = RowConstraintMiner(data_manager.clean_data)
        row_constraints, covered_attrs = row_miner.mine_row_constraints(attr_num=3)

        # 在DataManager中注入错误
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs)

        # 计算清洗前数据的平均绝对误差
        average_absolute_diff_before = data_utils.calculate_average_absolute_difference(data_manager.clean_data,
                                                                                        data_manager.observed_data)
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        # 检查数据是否违反了行约束
        violations_count = data_utils.check_constraints_violations(data_manager.observed_data, row_constraints)
        print(f"观测数据违反的行约束次数：{violations_count}")

        # 创建MTSCleanPareto实例并清洗数据，只传递行约束参数
        mtsclean_pareto = MTSCleanPareto()
        cleaned_data = mtsclean_pareto.clean(data_manager, row_constraints=row_constraints)

        # 计算清洗后数据的平均绝对误差
        average_absolute_diff_after = data_utils.calculate_average_absolute_difference(cleaned_data,
                                                                                       data_manager.clean_data)
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)

        # 检查数据是否违反了行约束
        violations_count = data_utils.check_constraints_violations(cleaned_data, row_constraints)
        print(f"清洗数据违反的行约束次数：{violations_count}")

        # 保存清洗后的数据到 CSV 文件
        cleaned_data.to_csv('Pareto_cleaned_data.csv', index=False)


class MTSCleanSoft(BaseCleaningAlgorithm):
    def __init__(self):
        pass

    def clean(self, data_manager, **args):
        row_constraints = args.get('row_constraints')
        speed_constraints = args.get('speed_constraints')

        # 检查约束
        if row_constraints is None or speed_constraints is None:
            raise ValueError("Row and speed constraints are required.")

        # 使用进程池进行并行计算
        with Pool() as pool:
            tasks = [(row, row_constraints, speed_constraints) for row in data_manager.observed_data.to_numpy()]
            results = list(tqdm(pool.imap(self._optimize_row, tasks), total=len(tasks), desc="MTSClean-Soft Cleaning"))

        # 将并行计算的结果合并到一个DataFrame中
        cleaned_data = pd.DataFrame(results, index=data_manager.observed_data.index,
                                    columns=data_manager.observed_data.columns)
        return cleaned_data

    def _optimize_row(self, task):
        row, row_constraints, speed_constraints = task

        # 注意：此处row是ndarray，直接使用即可
        def check_violations(x):
            violations = []
            for i, (_, coefs, rho_min, rho_max) in enumerate(row_constraints):
                value = np.dot(coefs, x)
                if value < rho_min:
                    violations.append((i, 'min'))
                elif value > rho_max:
                    violations.append((i, 'max'))
            return violations

        # 获取违反的约束
        violations = check_violations(row)  # 直接传递row（ndarray）

        # 如果没有违反的约束，直接返回原始行数据
        if not violations:
            return row

        def objective_function(x):
            score = 0
            sigmoid = lambda z: 1 / (1 + np.exp(-z))

            for i, (_, coefs, rho_min, rho_max) in enumerate(row_constraints):
                value = np.dot(coefs, x)
                violation_min = sigmoid(rho_min - value)
                violation_max = sigmoid(value - rho_max)

                if (i, 'min') in violations:
                    # 对于违反下界的约束，增加对应的权重
                    score += 1.35 * violation_min + 1.2 * violation_max
                elif (i, 'max') in violations:
                    # 对于违反上界的约束，增加对应的权重
                    score += 1.35 * violation_max + 1.2 * violation_min
                else:
                    # 对于未违反的约束，减少权重
                    score += 1.0 * violation_min + 1.0 * violation_max

            # 计算x与原始行数据row的距离
            distance = np.sum(0.001 * np.abs(x - row))
            score += sigmoid(distance)  # 使用sigmoid函数处理距离

            return score

        # 初始化搜索的起点为原始观测值
        initial_guess = row

        # 使用优化算法寻找最佳清洗值
        result = minimize(objective_function, initial_guess, method='L-BFGS-B', options={'maxiter': 100, 'tol': 1e-6})
        if result.success:
            return result.x
        else:
            return row  # 优化失败时返回原始观测值

    @staticmethod
    def test_MTSCleanSoft():
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

        # 创建MTSCleanSoft实例并清洗数据
        mtsclean_soft = MTSCleanSoft()
        cleaned_data = mtsclean_soft.clean(data_manager, row_constraints=row_constraints,
                                           speed_constraints=speed_constraints)

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
    # MTSCleanPareto.test_MTSCleanPareto()
    # MTSCleanSoft.test_MTSCleanSoft()
