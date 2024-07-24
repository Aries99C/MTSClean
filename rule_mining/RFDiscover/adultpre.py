import numpy as np
import Levenshtein


def getDistanceRelation(r):
    res = []
    for i in range(len(r) - 1):
        j = i + 1
        kk = []
        for k in range(len(r[i])):
            kk.append(Levenshtein.distance(r[i][k], r[j][k]))
        res.append(kk)
    res = np.array(res)
    return res


def pre_glass(n,m):
    fileHandler = open("data/adult.data", "r")
    data = []
    while True:
        line = fileHandler.readline()
        if not line:
            break
        x = line.split(',')
        data.append(x)

    data = np.array(data)
    data = data[:n, :m]
    Distance = getDistanceRelation(data)

    label = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
             "race", "sex",
             "capital-gain", "capital-gain", "hours-per-week", "native-country", "income"]
    return Distance, label
