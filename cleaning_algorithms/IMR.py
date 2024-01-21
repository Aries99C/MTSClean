import numpy as np
from tqdm import tqdm

import data_utils
from cleaning_algorithms.base_algorithm import BaseCleaningAlgorithm
from data_manager import DataManager


class IMRClean(BaseCleaningAlgorithm):
    def __init__(self, p=3, threshold=0.1, max_iter=10000):
        # IMR模型参数
        self.p = p
        self.threshold = threshold
        self.max_iter = max_iter
        self.MIN_VAL = np.inf
        self.MAX_VAL = -np.inf

    def learnParamsOLS(self, x_matrix, y_matrix):
        middle_matrix = x_matrix.T.dot(x_matrix)
        phi = np.linalg.pinv(middle_matrix).dot(x_matrix.T).dot(y_matrix)
        return phi

    def combine(self, phi, x_matrix):
        yhat_matrix = x_matrix.dot(phi)
        return yhat_matrix

    def repairAMin(self, yhat_matrix, y_matrix, label_list):
        a_min = self.MIN_VAL
        target_index = -1

        for i in range(len(yhat_matrix)):
            if label_list[i + self.p]:
                continue
            if abs(yhat_matrix[i] - y_matrix[i]) < self.threshold:
                continue

            yhat_abs = abs(yhat_matrix[i])
            if yhat_abs < a_min:
                a_min = yhat_abs
                target_index = i

        return target_index

    def clean(self, data_manager, **args):
        cleaned_data = data_manager.observed_data.copy()

        for col in tqdm(cleaned_data.columns, desc="IMR Cleaning"):
            label_list = data_manager.is_label[col].values
            labels = data_manager.clean_data[col].values
            data = cleaned_data[col].values

            size = len(data)
            row_num = size - self.p

            zs = labels - data

            x = np.array([zs[self.p + i - j - 1] for i in range(row_num) for j in range(self.p)]).reshape(row_num, self.p)
            y = zs[self.p:]

            x_matrix = np.matrix(x)
            y_matrix = np.matrix(y).T

            iteration_num = 0

            while iteration_num < self.max_iter:
                iteration_num += 1

                phi = self.learnParamsOLS(x_matrix, y_matrix)
                yhat_matrix = self.combine(phi, x_matrix).A1

                index = self.repairAMin(yhat_matrix, y_matrix.A1, label_list)

                if index == -1:
                    break

                val = yhat_matrix[index]
                y_matrix[index, 0] = val
                for j in range(self.p):
                    i = index + 1 + j
                    if i >= row_num:
                        break
                    if i < 0:
                        continue
                    x_matrix[i, j] = val

            for i in range(size):
                if not label_list[i]:
                    data[i] = data[i] + y_matrix[i - self.p, 0]
                else:
                    data[i] = labels[i]

        return cleaned_data

    @staticmethod
    def test_IMRClean():
        # 加载数据
        data_path = '../datasets/idf.csv'

        # 初始化DataManager
        data_manager = DataManager('idf', data_path)

        # 在DataManager中注入错误
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=data_manager.clean_data.columns)

        # 随机标记一定比例的数据为需要清洗
        data_manager.randomly_label_data(0.1)

        # 计算清洗前数据的平均绝对误差
        average_absolute_diff_before = data_utils.calculate_average_absolute_difference(data_manager.clean_data,
                                                                                        data_manager.observed_data)
        print("清洗前数据的平均绝对误差：", average_absolute_diff_before)

        # 创建IMRClean实例并清洗数据
        imrclean = IMRClean()
        cleaned_data = imrclean.clean(data_manager)

        # 计算清洗后数据的平均绝对误差
        average_absolute_diff_after = data_utils.calculate_average_absolute_difference(cleaned_data,
                                                                                       data_manager.clean_data)
        print("清洗后数据的平均绝对误差：", average_absolute_diff_after)


if __name__ == "__main__":
    IMRClean.test_IMRClean()
