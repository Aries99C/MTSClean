import pandas as pd


if __name__ == '__main__':
    idf = pd.read_csv('datasets/WADI.csv')

    idf.drop(columns=['Row'], inplace=True)

    idf.to_csv('datasets/WADI.csv', index=False, index_label=False)
