import numpy as np
import pandas as pd


def check_constraints_violations(data, constraints):
    violations_count = 0

    for index, row in data.iterrows():
        for _, coefs, rho_min, rho_max in constraints:
            value = np.dot(coefs, row)
            if not (rho_min <= value <= rho_max):
                violations_count += 1
                break  # 如果这一行已经违反了某个约束，就不再检查其他约束

    return violations_count


def calculate_average_absolute_difference(cleaned_data, observed_data):
    total_diff = np.abs(cleaned_data.values - observed_data.values).sum()
    n_elements = np.prod(observed_data.shape)  # 数据集中的元素总数
    average_diff = total_diff / n_elements
    return average_diff

