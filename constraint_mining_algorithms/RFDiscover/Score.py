def ok(data, rfd):
    L = rfd[0]
    R = rfd[1]
    for i in L:
        if data[i[0]] > i[1]:
            return 0
    if data[R[0]] > R[1]:
        return 1
    return 2


def get_Score1(r, rfd):
    down = 0
    up = 0
    for i in r:
        flag = ok(i, rfd)
        if flag == 1:
            down += 1
        elif flag == 2:
            up += 1
            down += 1
    if up == 0:
        return 0
    return up / down


import numpy as np


def get_Score2(r, rfd):
    max_dis = np.zeros(len(r[0]))
    for i in r:
        for k in range(len(i)):
            max_dis[k] = max(max_dis[k], i[k])
    resl = 0
    L = rfd[0]
    R = rfd[1]
    cnt = 0
    for i in L:
        resl += i[1] / max_dis[i[0]]
        cnt += 0
    if cnt == 0:
        resl = 0
    resr = R[1] / max_dis[R[0]]
    return (resl + resr) / 2


def get_Score(r, RFDs):
    res = 0
    cnt = 0
    for k in RFDs:
        res += (get_Score1(r, k) + get_Score2(r, k)) / 2
        cnt += 1
        if cnt % 50000 == 0:
            break
    return res / cnt
