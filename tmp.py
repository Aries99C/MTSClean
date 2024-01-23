import pandas as pd


if __name__ == '__main__':
    idf = pd.read_csv('datasets/SWaT.csv')

    idf.drop(columns=['Timestamp'], inplace=True)

    idf.to_csv('datasets/SWaT.csv', index=False, index_label=False)
