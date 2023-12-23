import pandas as pd
import numpy as np
from tqdm import tqdm

import data_utils
from cleaning_algorithms.base_algorithm import BaseCleaningAlgorithm
from scipy.signal import medfilt
from pykalman import KalmanFilter

from data_manager import DataManager


class EWMAClean(BaseCleaningAlgorithm):
    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def clean(self, data_manager, **args):
        cleaned_data = data_manager.observed_data.ewm(alpha=self.alpha, adjust=False).mean()
        return cleaned_data

    @staticmethod
    def test_EWMAClean():
        data_path = '../datasets/idf.csv'
        data_manager = DataManager('idf', data_path)
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=data_manager.clean_data.columns)

        average_absolute_diff_before = data_utils.calculate_average_absolute_difference(data_manager.clean_data,
                                                                                        data_manager.observed_data)
        print("EWMAClean - 清洗前数据的平均绝对误差：", average_absolute_diff_before)

        ewma_clean = EWMAClean(alpha=0.3)
        cleaned_data = ewma_clean.clean(data_manager)

        average_absolute_diff_after = data_utils.calculate_average_absolute_difference(cleaned_data,
                                                                                       data_manager.clean_data)
        print("EWMAClean - 清洗后数据的平均绝对误差：", average_absolute_diff_after)


class MedianFilterClean(BaseCleaningAlgorithm):
    def __init__(self, kernel_size=5):
        self.kernel_size = kernel_size

    def clean(self, data_manager, **args):
        # 使用 tqdm 包装列迭代
        cleaned_data = {col: medfilt(data_manager.observed_data[col].values, kernel_size=self.kernel_size)
                        for col in tqdm(data_manager.observed_data.columns, desc="Median Filtering")}

        return pd.DataFrame(cleaned_data, index=data_manager.observed_data.index)

    @staticmethod
    def test_MedianFilterClean():
        data_path = '../datasets/idf.csv'
        data_manager = DataManager('idf', data_path)
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=data_manager.clean_data.columns)

        average_absolute_diff_before = data_utils.calculate_average_absolute_difference(data_manager.clean_data,
                                                                                        data_manager.observed_data)
        print("MedianFilterClean - 清洗前数据的平均绝对误差：", average_absolute_diff_before)

        median_filter_clean = MedianFilterClean(kernel_size=5)
        cleaned_data = median_filter_clean.clean(data_manager)

        average_absolute_diff_after = data_utils.calculate_average_absolute_difference(cleaned_data,
                                                                                       data_manager.clean_data)
        print("MedianFilterClean - 清洗后数据的平均绝对误差：", average_absolute_diff_after)


class KalmanFilterClean(BaseCleaningAlgorithm):
    def __init__(self, initial_state=None, observation_covariance=None, transition_covariance=None, transition_matrices=None):
        self.initial_state = initial_state
        self.observation_covariance = observation_covariance
        self.transition_covariance = transition_covariance
        self.transition_matrices = transition_matrices

    def clean(self, data_manager, **args):
        cleaned_data = data_manager.observed_data.copy()

        # 使用 tqdm 包装列迭代
        for col in tqdm(cleaned_data.columns, desc="Kalman Filtering"):
            kf = KalmanFilter(initial_state_mean=self.initial_state[col],
                              observation_covariance=self.observation_covariance[col],
                              transition_covariance=self.transition_covariance[col],
                              transition_matrices=self.transition_matrices[col])

            cleaned_data[col], _ = kf.filter(cleaned_data[col].values)

        return cleaned_data

    @staticmethod
    def test_KalmanFilterClean():
        data_path = '../datasets/idf.csv'
        data_manager = DataManager('idf', data_path)
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'], covered_attrs=data_manager.clean_data.columns)

        initial_state, observation_covariance, transition_covariance, transition_matrices = data_manager.estimate_kalman_parameters()

        average_absolute_diff_before = data_utils.calculate_average_absolute_difference(data_manager.clean_data,
                                                                                        data_manager.observed_data)
        print("KalmanFilterClean - 清洗前数据的平均绝对误差：", average_absolute_diff_before)

        kalman_filter_clean = KalmanFilterClean(initial_state=initial_state,
                                                observation_covariance=observation_covariance,
                                                transition_covariance=transition_covariance,
                                                transition_matrices=transition_matrices)
        cleaned_data = kalman_filter_clean.clean(data_manager)

        average_absolute_diff_after = data_utils.calculate_average_absolute_difference(cleaned_data,
                                                                                       data_manager.clean_data)
        print("KalmanFilterClean - 清洗后数据的平均绝对误差：", average_absolute_diff_after)


if __name__ == '__main__':
    EWMAClean.test_EWMAClean()
    MedianFilterClean.test_MedianFilterClean()
    KalmanFilterClean.test_KalmanFilterClean()
