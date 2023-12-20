import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import random


class RowConstraintMiner:
    def __init__(self, df):
        self.df = df

    def generate_binary_strings(self, m, k):
        base_array = np.zeros(m, dtype=int)
        all_combinations = []

        for positions in combinations(range(m), k):
            new_array = base_array.copy()
            new_array[list(positions)] = 1
            all_combinations.append(new_array)

        return all_combinations

    def train_models_and_evaluate(self, k):
        m = self.df.shape[1]
        binary_strings = self.generate_binary_strings(m, k)
        models_with_loss = []

        for binary_string in binary_strings:
            selected_columns = self.df.columns[binary_string == 1]

            # 从选中的列中随机选择一列作为y
            y_column = random.choice(selected_columns)
            y = self.df[y_column]
            X = self.df[selected_columns.drop(y_column)]

            # 训练 Ridge 模型
            model = Ridge()
            model.fit(X, y)
            y_pred = model.predict(X)
            loss = mean_squared_error(y, y_pred)

            models_with_loss.append((binary_string, y_column, model, loss))

        # 根据损失对模型排序
        models_with_loss.sort(key=lambda x: x[3])

        return models_with_loss

    def select_optimal_models(self, models_with_loss):
        # 对模型按照损失排序
        sorted_models = sorted(models_with_loss, key=lambda x: x[3])

        # 初始化一个全0的数组来跟踪已覆盖的顶点
        covered = np.zeros(self.df.shape[1], dtype=int)
        selected_models = []

        for model_info in sorted_models:
            binary_string, _, _, _ = model_info
            # 检查当前模型是否添加了新的覆盖
            if not np.any(np.bitwise_and(binary_string, np.bitwise_not(covered))):
                continue

            # 添加当前模型，并更新已覆盖的顶点
            selected_models.append(model_info)
            covered = np.bitwise_or(covered, binary_string)

            # 检查是否所有顶点都已被覆盖
            if np.all(covered == 1):
                break

        return selected_models

    def mine_row_constraints(self, attr_num=3):
        models_with_loss = self.train_models_and_evaluate(attr_num)
        optimal_models = self.select_optimal_models(models_with_loss)

        constraints = []
        for binary_string, y_column, model, _ in optimal_models:
            selected_columns = self.df.columns[binary_string == 1]
            X = self.df[selected_columns.drop(y_column)]
            y = self.df[y_column]

            # 重新训练 Ridge 模型
            new_model = Ridge()
            new_model.fit(X, y)
            y_pred = new_model.predict(X)

            # 计算每个数据点的损失
            losses = y_pred - y
            rho_min, rho_max = np.min(losses), np.max(losses)

            # 准备系数数组
            full_coef = np.zeros(self.df.shape[1])
            # 将模型系数放在正确的位置
            full_coef[[self.df.columns.get_loc(col) for col in selected_columns.drop(y_column)]] = new_model.coef_
            # y_column 的系数设置为 -1
            full_coef[self.df.columns.get_loc(y_column)] = -1

            # 构建约束字符串，只包括非零系数的项
            terms = [f"{coef_val:.3f} * {col}" for coef_val, col in zip(full_coef, self.df.columns) if coef_val != 0]
            constraint_str = f"{rho_min:.3f} <= {' + '.join(terms)} <= {rho_max:.3f}"

            constraints.append((constraint_str, full_coef, rho_min, rho_max))

        return constraints


class ColConstraintMiner:
    def __init__(self, df):
        self.df = df

    def _calculate_speeds(self):
        speed_constraints = {}
        for col in self.df.columns:
            speeds = self.df[col].diff()  # 一阶差分计算速度，保留符号
            speed_constraints[col] = (speeds.min(), speeds.max())
        return speed_constraints

    def _calculate_accelerations(self):
        acceleration_constraints = {}
        for col in self.df.columns:
            accelerations = self.df[col].diff().diff()  # 二阶差分计算加速度，保留符号
            acceleration_constraints[col] = (accelerations.min(), accelerations.max())
        return acceleration_constraints

    def mine_col_constraints(self):
        speed_constraints = self._calculate_speeds()
        acceleration_constraints = self._calculate_accelerations()
        return speed_constraints, acceleration_constraints
