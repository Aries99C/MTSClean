import difflib
import itertools
import math
import random
import numpy as np

from sklearn import linear_model


class Linear:
    """
    线性时窗约束的挖掘模型，提供线性时窗约束的规则挖掘与可视化功能
    """

    def __init__(self, mts, win_size=2, confidence=0.95, n_components=1, max_len=3,
                 sample_train=1.0, pc=1.0, ridge_all=False, implication_check=False):
        # 配置数据集
        self.mts = mts

        # 配置模型参数
        self.win_size = win_size
        self.confidence = confidence
        self.n_components = n_components
        self.max_len = max_len
        self.sample_train = sample_train
        self.pc = pc
        self.ridge_all = ridge_all
        self.implication_check = implication_check
        self.variables = []                         # 规则中的变量
        self.rules = []                             # 规则集合

        self.success_imp = 0
        self.all_imp = 0
        self.imp_time = 0

        # 配置规则中的变量名
        self.generate_variable_name()

    def generate_variable_name(self):
        self.variables = []
        for t in range(self.win_size):
            for i, col in enumerate(self.mts.cols):
                self.variables.append((t, col, i))   # 每个变量的形式为(时间戳，列名称，列索引)

    def mini_mine_sh(self, x, y, x_vars, y_var, max_x=1, verbose=0):
        # 在所有可能的掩码中学习规则
        x_possible = all_masks(len(x_vars), max_x)
        # 过滤相似度低的属性，也过滤速度约束
        remove_mask = []
        for mask_str in x_possible:
            mask = str2mask(mask_str)
            for i in range(len(mask)):
                # if mask[i] == 1 and x_vars[i][2] == y_var[2]:     # 忽略列名近似性
                if mask[i] == 1 and (str_similar(x_vars[i][1], y_var[1]) < 0.8 or (x_vars[i][2] == y_var[2])):
                    remove_mask.append(mask_str)
        for mask in remove_mask:
            x_possible.remove(mask)

        # # 生成随机掩码集合来学习规则
        # x_possible = set()                      # 可能存在关联的变量X集合
        # while len(x_possible) < 1024:
        #     mask = random_mask(len(x_vars), max_x=max_x)    # 获得随机掩码
        #     if '1' not in mask:                 # 随机选择没有选到任何变量X
        #         continue
        #     else:
        #         x_possible.add(mask)

        # 使用sklearn学习线性模型
        while len(x_possible) > self.n_components:  # 一直筛选直到得到参数限定的掩码个数
            n_possible = len(x_possible)
            mask_loss = {}                  # 记录随机掩码对应的拟合误差
            budget = int(x.shape[0] / (n_possible * np.log2(n_possible)) + 10)
            x_train = x[:budget, :]         # 训练数据x
            y_train = y[:budget]            # 训练数据y
            for i, mask_str in enumerate(x_possible):     # 对每个掩码都学习一次线性函数
                mask = str2mask(mask_str)   # 将字符串转换为list
                if mask is None:            # 有bug莫名其妙多一个解析不了的字符串
                    continue
                selected_x_train = None
                # 由掩码生成训练数据
                for idx in range(len(mask)):
                    if mask[idx] == 1:      # 选中掩码对应的列加入训练集
                        if selected_x_train is None:    # 第一次加入训练数据
                            selected_x_train = x_train[:, idx]
                        else:                           # 合并后加入的训练数据
                            selected_x_train = np.c_[selected_x_train, x_train[:, idx]]
                if len(selected_x_train.shape) == 1:    # 如果只选中一列X，需要reshape成2d数据
                    selected_x_train = selected_x_train.reshape(-1, 1)
                # 训练线性模型
                model = linear_model.Ridge(self.pc)
                model.fit(selected_x_train, y_train)
                # 记录线性模型的拟合误差
                loss = np.sum((y_train - model.predict(selected_x_train)) ** 2)
                mask_loss[mask_str] = loss
            # 根据误差重新筛选随机掩码
            sorted_mask_str = sorted(mask_loss.items(), key=lambda kv: kv[1], reverse=False)  # 根据误差升序排列掩码
            x_possible = set()
            while len(x_possible) < n_possible / 2:     # 删除一半的掩码
                x_possible.add(sorted_mask_str[len(x_possible)][0])
        if verbose > 0:     # 日志显示
            idx = 1
            for mask_str in x_possible:
                print('变量X集合掩码{}: {}'.format(idx, np.array(mask_str)))
                idx = idx + 1
        # 使用筛选后的掩码生成规则
        for mask_str in x_possible:
            x_idx = []
            x_names = []
            mask = str2mask(mask_str)       # 将字符串转换为list
            for idx in range(len(mask)):    # 得到掩码对应的变量X集合
                if mask[idx] == 1:
                    x_idx.append(idx)
                    x_names.append(x_vars[idx])
            # 由掩码和模型参数计算生成规则所需的数据
            m, n = x.shape
            x_mine = x[:int(m * self.sample_train), x_idx]
            y_mine = y[:int(m * self.sample_train)]
            # 挖掘规则
            model = linear_model.Ridge(self.pc)
            model.fit(x_mine, y_mine)
            # 将规则转换成数据结构
            func = {    # 记录挖掘的函数信息
                'type': 'linear',                                       # 记录挖掘的函数类型
                'coef': [model_coef for model_coef in model.coef_],     # 记录变量X的系数
                'intercept': model.intercept_,                          # 函数的截距
            }
            # 根据置信度计算函数上下界
            losses = abs(model.predict(x_mine) - y_mine)    # 模型误差
            mean_loss = np.mean(losses)                     # 平均误差
            b = max(losses)                                 # 最大误差
            gamma = mean_loss
            +b * math.sqrt(math.log(1/(1-self.confidence))/(2*m))
            +b * math.sqrt(2*(len(x_vars)+1)*((math.log((math.e*m)/(2*(len(x_vars)+1))))/m))
            r_mine = Rule(x_names, y_var, func, -gamma, gamma, model, self.mts.dim, self.win_size)   # 解析规则
            if verbose > 0:     # 展示规则
                print(r_mine)
            self.rules.append(r_mine)                                   # 添加规则

    def mine(self, max_x=1, verbose=0):
        if verbose > 0:         # 日志显示
            print('{:=^80}'.format(' 在数据集{}上挖掘线性时窗约束 '.format(self.mts.dataset.upper())))
        d = array2window(self.mts.clean2array(), self.win_size)     # 对多元时序数据切片
        if verbose > 0:         # 日志显示
            print('数据切片shape: {}'.format(d.shape))

        for i in range(d.shape[1]):     # 扫描每个切片的所有属性在时窗内多个时间戳上的值
            if verbose > 0:     # 日志显示
                print('{:=^40}'.format(' 挖掘Y=t{}[{}]上的f(X) '.format(self.variables[i][0], self.variables[i][1])))
            y_var = self.variables[i]   # 待挖掘函数Y=f(X)的变量Y
            x_vars = [var for var in self.variables if not (var[1] == y_var[1] and var[0] == y_var[0])]     # 变量X集合
            y = d[:, i]     # 变量Y的数据
            x = np.c_[d[:, 0: i], d[:, i+1:]] if i > 0 else d[:, i+1:]  # 变量X集合的数据
            # 挖掘时窗函数规则
            self.mini_mine_sh(x, y, x_vars, y_var, max_x=max_x, verbose=verbose)


def array2window(x, win_size=2):
    """
    将数据根据时窗长度切片。原本数据的shape为(n,m)，其中n为长度，m为维数，
    切片后得到的切片组shape为(n-win_size+1, win_size*m))。
    :param x: 原始数据，类型为ndarray，shape=(n,m)
    :param win_size: 时窗长度
    :return: 切片后的数据，类型为ndarray，shape=(n-win_size+1, win_size*m)
    """
    n, m = x.shape              # 原始数据shape
    slices = []                 # 返回的切片组
    for i in range(n - win_size + 1):                       # 共计n-win_size+1个窗口
        slices.append(x[i: i + win_size, :].reshape(-1))    # 切片
    slices = np.array(slices)   # 将list转换为ndarray
    return slices


def random_mask(vec_len, max_x):
    """
    随机生成长度为vec_len的01掩码串
    :param vec_len: 生成的掩码长度
    :param max_x: 掩码中为1的最大位数，限制约束的复杂程度
    :return: 长度为vec_len的随机01掩码串
    """
    one_bits = random.randint(1, max_x)
    mask_list = ['0' for _ in range(vec_len)]
    while mask_list.count('1') < one_bits:
        random_bit = random.randint(0, vec_len-1)
        mask_list[random_bit] = '1'
    return ''.join(mask_list)


def all_masks(vec_len, max_x):
    """
    生成长度为vec_len的所有可能的01掩码串
    :param vec_len: 生成的掩码长度
    :param max_x: 掩码中为1的最大位数，限制约束的复杂程度
    :return: 长度为vec_len的所有01掩码串
    """
    masks = set()
    masks_with_one = [''.join(item) for item in itertools.product("01", repeat=max_x)]   # 生成长度为max_x的01全排列
    masks_with_one.pop(0)    # 删除全0的排列项
    for i in range(vec_len - max_x):
        for with_one in masks_with_one:
            mask = '0' * i + with_one + '0' * (vec_len - max_x - i)
            masks.add(mask)
    return masks


def str2mask(mask_str):
    try:
        mask = [int(char) for char in mask_str]
        return mask
    except ValueError:
        return None


def str_similar(a, b):
    return difflib.SequenceMatcher(None, a, b).quick_ratio()


class Rule:
    def __init__(self, x_names, y_name, func, lb, ub, model, m, w):
        self.x_names = x_names
        self.y_name = y_name
        self.func = func
        self.lb = lb
        self.ub = ub
        self.model = model
        self.m = m
        self.w = w
        self.alpha = np.zeros(self.m * self.w)
        self.mask = self.get_mask()

    def get_mask(self):
        mask = np.zeros(self.m * self.w)      # 初始化掩码为全0
        for i, x_name in enumerate(self.x_names):
            x_pos = x_name[0] * self.m + x_name[2]
            mask[x_pos] = 1
            self.alpha[x_pos] = self.func['coef'][i]
        y_pos = self.y_name[0] * self.m + self.y_name[2]
        mask[y_pos] = 1
        self.alpha[y_pos] = -1

        return mask

    def violation_degree(self, t):
        """
        计算约束违反程度
        :param t: 数据切片
        :return:
        """
        f = np.dot(t, self.alpha) + self.func['intercept']
        if f < self.lb:
            return self.lb - f
        if f > self.ub:
            return f - self.ub
        return 0.

    def __str__(self):
        f = '{:.3f} <= '.format(self.lb, 2)
        for i in range(len(self.x_names)):
            f = f + '{:.3f}*t{}[{}]'.format(self.func['coef'][i], self.x_names[i][0], self.x_names[i][1])
            f = f + ' + '
        f = f + ' ({:.3f}) '.format(self.func['intercept'])
        f = f + ' - (t{}[{}]) <= {:.3f}'.format(self.y_name[0], self.y_name[1], self.ub, 2)
        return f

    def __hash__(self):
        return hash(self.y_name[1])

    def __eq__(self, other):
        return self.y_name[1] == other.y_name[1]


if __name__ == '__main__':
    print(str_similar('U3_HNC10CT121', 'U3_HNC10CT121'))
