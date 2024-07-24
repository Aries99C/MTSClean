import numpy as np



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

def pre_glass(n,m):
    fileHandler = open("Rice_Cammeo_Osmancik.arff", "r")
    data = []
    while True:
        line = fileHandler.readline()
        if not line:
            break
        x = line.split(',')
        int_x=x[:len(x)-2]
        type_x=x[len(x)-1]
        if type_x=="Osmancik":
            continue
        int_x = [float(i) for i in int_x]
        data.append(int_x)


    data = np.array(data)
    data = data[:n, :m]
    Distance = getDistanceRelation(data)

    label = ["Area", "Perimeter", "Major Axis Length", "Minor Axis Length", "Eccentricity", "Eccentricity", "Extent"]
    return Distance,label

