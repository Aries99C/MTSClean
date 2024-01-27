import pandas as pd

from constraints import RowConstraintMiner, ColConstraintMiner
from data_manager import DataManager

if __name__ == '__main__':
    # 指定数据集的路径
    data_path = './datasets/idf.csv'

    # 创建 DataManager 实例
    data_manager = DataManager(dataset='test', dataset_path=data_path)

    row_miner = RowConstraintMiner(data_manager.clean_data)
    row_constraints, covered_attrs = row_miner.mine_row_constraints()

    col_miner = ColConstraintMiner(data_manager.clean_data)
    speed_constraints, accel_constraints = col_miner.mine_col_constraints()


