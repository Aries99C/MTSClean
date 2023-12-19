import pandas as pd
import numpy as np
import data_utils
from base_algorithm import BaseCleaningAlgorithm
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
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'])

        average_absolute_diff_before = data_utils.calculate_average_absolute_difference(data_manager.clean_data,
                                                                                        data_manager.observed_data)
        print("EWMAClean - 清洗前数据的平均绝对误差：", average_absolute_diff_before)

        ewma_clean = EWMAClean(alpha=0.3)
        cleaned_data = ewma_clean.clean(data_manager)

        average_absolute_diff_after = data_utils.calculate_average_absolute_difference(cleaned_data,
                                                                                       data_manager.clean_data)
        print("EWMAClean - 清洗后数据的平均绝对误差：", average_absolute_diff_after)


class MedianFilterClean(BaseCleaningAlgorithm):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def clean(self, data_manager, **args):
        cleaned_data = data_manager.observed_data.apply(lambda x: medfilt(x, kernel_size=self.kernel_size))
        return pd.DataFrame(cleaned_data, index=data_manager.observed_data.index, columns=data_manager.observed_data.columns)

    @staticmethod
    def test_MedianFilterClean():
        data_path = '../datasets/idf.csv'
        data_manager = DataManager('idf', data_path)
        data_manager.inject_errors(0.2, ['drift', 'gaussian', 'volatility', 'gradual', 'sudden'])

        average_absolute_diff_before = data_utils.calculate_average_absolute_difference(data_manager.clean_data,
                                                                                        data_manager.observed_data)
        print("MedianFilterClean - 清洗前数据的平均绝对误差：", average_absolute_diff_before)

        median_filter_clean = MedianFilterClean(kernel_size=3)
        cleaned_data = median_filter_clean.clean(data_manager)

        average_absolute_diff_after = data_utils.calculate_average_absolute_difference(cleaned_data,
                                                                                       data_manager.clean_data)
        print("MedianFilterClean - 清洗后数据的平均绝对误差：", average_absolute_diff_after)


if __name__ == '__main__':
    EWMAClean.test_EWMAClean()
    MedianFilterClean.test_MedianFilterClean()
