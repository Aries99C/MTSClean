import pandas as pd


if __name__ == '__main__':
    idf = pd.read_csv('datasets/idf.csv')

    idf.drop(columns=['Unnamed: 0'], inplace=True)

    idf.to_csv('datasets/idf.csv', index=False, index_label=False)
