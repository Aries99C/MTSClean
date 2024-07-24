import numpy as np
import os
import pandas as pd


def read_data(file_name):
    path = os.getcwd() + file_name
    f = open(path, encoding='utf-8')
    data = pd.read_csv(f)
    name_list = ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure"
        , "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking", "time", "DEATH_EVENT"]
    data_x = data[name_list]
    x = data_x.values
    return x, name_list


def getDistanceRelation(r):
    res = []
    for i in range(len(r) - 1):
        j = i + 1
        kk = []
        for k in range(len(r[i])):
            kk.append(abs(r[i][k] - r[j][k]))
        res.append(kk)
    res = np.array(res)
    return res


def pre_glass(n, m):
    data, label = read_data('\\heart_failure_clinical_records_dataset.csv')

    data = np.array(data)
    data = data[:n, :m]
    Distance = getDistanceRelation(data)

    return Distance, label
