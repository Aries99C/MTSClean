# cleaning_algorithms/base_algorithm.py

from abc import ABC, abstractmethod


class BaseCleaningAlgorithm(ABC):
    @abstractmethod
    def clean(self, data_manager, **args):
        """
        清洗数据的抽象方法。
        :param data_manager: DataManager对象，包含clean_data和observed_data等信息。
        :param args: 算法需要的其他参数。
        :return: 清洗后的数据。
        """
        pass

