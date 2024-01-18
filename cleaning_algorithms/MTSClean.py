import heapq

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
import geatpy as ea
from tqdm import tqdm
from multiprocessing import Pool


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
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], row_constraints=constraints)

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

        speed_constraints = args.get('speed_constraints')
        if speed_constraints is None:
            raise ValueError("Speed constraints are required for secondary cleaning.")

        # 计算总的迭代次数
        total_steps = data_manager.observed_data.shape[0] + data_manager.observed_data.shape[0] * data_manager.observed_data.shape[1]
        with tqdm(total=total_steps, desc="Total Cleaning Progress") as pbar:
            # 先进行行约束清洗
            cleaned_data = self._clean_with_row_constraints(data_manager.observed_data, row_constraints, pbar)
            # 再进行速度约束清洗
            cleaned_data = self._clean_with_speed_constraints(cleaned_data, speed_constraints, pbar)

        return cleaned_data

    def _clean_with_row_constraints(self, data, constraints, pbar):
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

            pbar.update(1)  # 更新进度条

        return pd.DataFrame(cleaned_data, columns=data.columns)

    def _clean_with_speed_constraints(self, lp_cleaned_data, speed_constraints, pbar):
        w = 10  # 窗口长度

        # 创建清洗后的数据副本
        cleaned_data = lp_cleaned_data.copy()

        for col in cleaned_data.columns:
            # 获取当前列的速度约束上下界
            speed_lb = speed_constraints[col][0]
            speed_ub = speed_constraints[col][1]
            data = lp_cleaned_data[col].values

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

            pbar.update(len(lp_cleaned_data))

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
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], row_constraints=row_constraints)

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
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], row_constraints=row_constraints)

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
    def clean(self, data_manager, **args):
        row_constraints = args.get('row_constraints')
        speed_constraints = args.get('speed_constraints')

        if row_constraints is None or speed_constraints is None:
            raise ValueError("Row and speed constraints are required.")

        cleaned_data = pd.DataFrame(columns=data_manager.observed_data.columns)
        previous_row = data_manager.observed_data.iloc[0].copy()
        for index, current_row in tqdm(data_manager.observed_data.iterrows(), total=data_manager.observed_data.shape[0], desc="Cleaning"):
            optimized_row = self._optimize_row(current_row, previous_row, row_constraints, speed_constraints)
            cleaned_data.loc[index] = optimized_row
            # 更新previous_row，确保数据类型保持一致
            previous_row = pd.Series(optimized_row, index=current_row.index)

        return cleaned_data

    def _optimize_row(self, current_row, previous_row, row_constraints, speed_constraints):
        violated_constraints = self._check_constraints_violation(current_row, row_constraints)

        if not violated_constraints:
            return current_row.values  # 没有违反的约束，无需优化

        target_col_indices = self._find_min_cover_columns(current_row, previous_row, violated_constraints,
                                                          speed_constraints)

        # 对每个待修复列计算允许的范围
        allowed_ranges = []
        for col_index in target_col_indices:
            # 找到与当前待修复列相关的行约束
            relevant_constraints = [constraint for constraint in violated_constraints if constraint[1][col_index] != 0]
            # 计算这些行约束的公共允许范围
            common_min, common_max = 0, 30
            for constraint in relevant_constraints:
                min_val, max_val = self._calculate_allowed_range(current_row, constraint, col_index)
                common_min = max(common_min, min_val)
                common_max = min(common_max, max_val)
            allowed_ranges.append((common_min, common_max))

        # 定义目标函数
        objective_function = lambda x: self._objective_function(x, current_row, allowed_ranges, target_col_indices)

        # 初始化搜索的起点为原始观测值中的目标列
        # initial_guess = current_row.iloc[list(target_col_indices)].values
        initial_guess = np.array([(common_min + common_max) / 2 for common_min, common_max in allowed_ranges])

        print(f'从点{[(current_row.iloc[col_index], (allowed_ranges[i][0] + allowed_ranges[i][1]) / 2) for i, col_index in enumerate(target_col_indices)]}出发开始搜索')

        # 使用优化算法寻找最佳清洗值
        result = minimize(objective_function, initial_guess, method='L-BFGS-B')
        if result.success:
            # current_row.iloc[list(target_col_indices)] = result.x
            current_row.iloc[list(target_col_indices)] = [(common_min + common_max) / 2 for common_min, common_max in allowed_ranges]
        else:
            # 将优化后的值更新到current_row中的目标列
            current_row.iloc[list(target_col_indices)] = [(common_min + common_max) / 2 for common_min, common_max in allowed_ranges]

        print(f'搜索结果{current_row.iloc[list(target_col_indices)]}')

        return current_row.values

    def _calculate_speed_violation(self, current_value, previous_value, speed_constraint):
        speed_lb, speed_ub = speed_constraint
        speed = current_value - previous_value
        if speed < speed_lb:
            return abs(speed_lb - speed)
        elif speed > speed_ub:
            return abs(speed - speed_ub)
        return 0

    def _find_min_cover_columns(self, current_row, previous_row, row_constraints, speed_constraints):
        # 初始化列的优先级队列
        priority_queue = []

        # 为每个列计算优先级
        for col_index, col_name in enumerate(current_row.index):
            # speed_violation = self._calculate_speed_violation(current_row[col_name], previous_row[col_name], speed_constraints[col_name])
            speed_violation = abs(current_row[col_name] - previous_row[col_name])
            priority = speed_violation
            priority_queue.append((-priority, col_index))  # 使用负值，因为队列是最大堆

        # 构建最小覆盖集合
        heapq.heapify(priority_queue)
        covered_constraints_count = 0
        min_cover_cols = set()

        while priority_queue and covered_constraints_count < len(row_constraints):
            _, col_index = heapq.heappop(priority_queue)
            min_cover_cols.add(col_index)
            for constraint in row_constraints:
                if constraint[1][col_index] != 0:
                    covered_constraints_count += 1

        return min_cover_cols

    def _calculate_allowed_range(self, row, constraint, target_col_index):
        _, coefs, rho_min, rho_max = constraint
        sum_other = np.dot(np.concatenate((coefs[:target_col_index], coefs[target_col_index + 1:])),
                           np.concatenate((row[:target_col_index], row[target_col_index + 1:])))
        if coefs[target_col_index] != 0:
            min_val = (rho_min - sum_other) / coefs[target_col_index]
            max_val = (rho_max - sum_other) / coefs[target_col_index]

            if min_val <= max_val:
                return max(min_val, 0), min(max_val, 30)
            else:
                return max(max_val, 0), min(min_val, 30)  # 交换值以确保合理的范围
        return 0, 30  # 如果当前列系数为0，则允许整个范围

    def _check_constraints_violation(self, row, row_constraints):
        violated_constraints = []
        for constraint in row_constraints:
            _, coefs, rho_min, rho_max = constraint
            value = np.dot(coefs, row)
            if value < rho_min or value > rho_max:
                violated_constraints.append(constraint)
        return violated_constraints

    def _objective_function(self, x, current_row, allowed_ranges, target_col_indices):
        score = 0
        sigmoid = lambda z: 1 / (1 + np.exp(-z))

        # 修复列的修复值与观测值的绝对差的权重
        weight_diff = 0.1

        # 行约束违反的权重
        weight_violation = 0.9

        # 创建从target_col_indices到x索引的映射
        col_index_to_x_index = {col_index: i for i, col_index in enumerate(target_col_indices)}

        # 修复列的修复值与观测值的绝对差
        for col_index in target_col_indices:
            x_index = col_index_to_x_index[col_index]
            score += weight_diff * abs(x[x_index] - current_row.iloc[col_index])

        # 行约束的允许范围内的违反代价
        for col_index, (lower_bound, upper_bound) in zip(target_col_indices, allowed_ranges):
            x_index = col_index_to_x_index[col_index]
            if x[x_index] < lower_bound or x[x_index] > upper_bound:
                # violation = min(abs(x[x_index] - lower_bound), abs(x[x_index] - upper_bound))
                score += weight_violation * abs(x[x_index] - (upper_bound + lower_bound) / 2)

        return score

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
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], row_constraints=row_constraints)

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
    # MTSClean.test_MTSClean()
    # MTSCleanPareto.test_MTSCleanPareto()
    MTSCleanSoft.test_MTSCleanSoft()
