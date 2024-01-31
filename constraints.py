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
        # 初始化一个记录属性被选中次数的字典
        attr_selected_count = {col: 0 for col in self.df.columns}

        selected_models = []
        for binary_string, y_column, model, loss in models_with_loss:
            # 检查模型的平均绝对误差是否在阈值以下
            if loss > 5:
                continue

            # 确定当前模型涉及的属性
            involved_attrs = self.df.columns[binary_string == 1]

            # 计算与已覆盖属性的重叠数
            overlap_count = sum([binary_string[self.df.columns.get_loc(col)] for col in attr_selected_count if
                                 attr_selected_count[col] > 0])

            # 获取唯一重叠的属性被选中的次数
            unique_overlap_selected_count = max(
                [attr_selected_count[col] for col in involved_attrs if attr_selected_count[col] > 0], default=0)

            # 根据规则判断是否添加当前模型
            if overlap_count <= 1 and unique_overlap_selected_count < 3:
                selected_models.append((binary_string, y_column, model, loss))
                # 更新属性被选中次数
                for attr in involved_attrs:
                    attr_selected_count[attr] += 1

        return selected_models

    def mine_row_constraints(self, attr_num=3):
        models_with_loss = self.train_models_and_evaluate(attr_num)
        optimal_models = self.select_optimal_models(models_with_loss)

        constraints = []
        covered_attrs = set()  # 用于记录所覆盖的属性集合
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
            rho_min, rho_max = np.min(losses) - new_model.intercept_, np.max(losses) - new_model.intercept_

            # 准备系数数组
            full_coef = np.zeros(self.df.shape[1])
            for col in selected_columns.drop(y_column):
                full_coef[self.df.columns.get_loc(col)] = new_model.coef_[X.columns.get_loc(col)]
            full_coef[self.df.columns.get_loc(y_column)] = -1

            # 更新covered_attrs，只包括系数为-1的列
            for col, coef in zip(self.df.columns, full_coef):
                if coef == -1:
                    covered_attrs.add(col)

            # 构建约束字符串，只包括非零系数的项
            terms = [f"{coef_val:.3f} * {col}" for coef_val, col in zip(full_coef, self.df.columns) if coef_val != 0]
            constraint_str = f"{rho_min:.3f} <= {' + '.join(terms)} <= {rho_max:.3f}"

            constraints.append((constraint_str, full_coef, rho_min, rho_max))

        return constraints, covered_attrs


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
