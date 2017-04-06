from data_util import DataUtils
import numpy as np
import math
import os


def load_data():
    train_file_x = 'dataset/train-images.idx3-ubyte'
    train_file_y = 'dataset/train-labels.idx1-ubyte'
    test_file_x = 'dataset/t10k-images.idx3-ubyte'
    test_file_y = 'dataset/t10k-labels.idx1-ubyte'

    train_x = DataUtils(filename=train_file_x).get_image()  # 60000*784
    train_y = DataUtils(filename=train_file_y).get_label()  # 60000*1
    test_x = DataUtils(test_file_x).get_image()  # 10000*784
    test_y = DataUtils(test_file_y).get_label()  # 10000*1

    # convert from float to int
    train_x.astype(np.int)
    train_y.astype(np.int)
    test_x.astype(np.int)
    test_y.astype(np.int)

    # save image in local, not necessary
    # path_train_set = "images/train"
    # path_test_set = "images/test"
    # if not os.path.exists(path_train_set):
    #     os.mkdir(path_train_set)
    # if not os.path.exists(path_test_set):
    #     os.mkdir(path_test_set)
    # DataUtils(outpath=path_train_set).out_image(train_x, train_y)
    # DataUtils(outpath=path_test_set).out_image(test_x, test_y)

    # save binary dataset to matrix, not necessary
    # np.savetxt('data/train_image_matrix.txt', train_x, fmt='%d', delimiter='\t')
    # np.savetxt('data/train_label_matrix.txt', train_y, fmt='%d', delimiter='\t')
    # np.savetxt('data/test_image_matrix.txt', test_x, fmt='%d', delimiter='\t')
    # np.savetxt('data/test_label_matrix.txt', test_y, fmt='%d', delimiter='\t')

    print('load data success...')
    return train_x, train_y, test_x, test_y


def get_activate(x):
    activate_vector = []
    for i in x:
        # 用的是单极性S型函数 值域从0到1
        activate_vector.append(1 / (1 + math.exp(-i)))
    activate_vector = np.array(activate_vector)
    return activate_vector


def train(w1, w2, hide_offset, output_offset):
    for i in range(train_sample_num):
        # get label for each sample
        t_label = np.zeros(output_num)
        t_label[train_y[i]] = 1

        # 前向过程
        # train_x[i]:1*784 w1:784*28 hide_offset:1*28
        hide_value = np.dot(train_x[i], w1) + hide_offset  # 隐藏层值
        hide_activate = get_activate(hide_value)  # 隐藏层激活值
        # hide_activate:1*28 w2:28*10 output_offset:1*10
        output_value = np.dot(hide_activate, w2) + output_offset  # 输出层值
        output_activate = get_activate(output_value)  # 输出层激活值

        # 后向过程
        # http://www.cnblogs.com/fengfenggirl/p/bp_network.html
        error = t_label - output_activate  # 输出值和真实值之间的误差
        # output_delta 10*1
        output_delta = error * output_activate * (1 - output_activate)
        # hide_delta 28*1
        hide_delta = hide_activate * (1 - hide_activate) * np.dot(w2, output_delta)

        for j in range(output_num):
            w2[:, j] += hide_learn_rate * output_delta[j] * hide_activate  # 更新隐藏层权矩阵
        for j in range(hide_num):
            w1[:, j] += input_learn_rate * hide_delta[j] * train_x[i]  # 更新输入层权矩阵

        output_offset += hide_learn_rate * output_delta  # 更新offset
        hide_offset += input_learn_rate * hide_delta
    return w1, w2, hide_offset, output_offset


def test(w1, w2, hide_offset, output_offset):
    test_sample_num = np.shape(test_y)[0]
    right = np.zeros(10)
    numbers = np.zeros(10)

    for i in range(test_sample_num):
        hide_value = np.dot(test_x[i], w1) + hide_offset
        hide_activate = get_activate(hide_value)
        output_value = np.dot(hide_activate, w2) + output_offset
        output_activate = get_activate(output_value)
        if np.argmax(output_activate) == test_y[i]:
            right[test_y[i]] += 1
    right_sum = right.sum()
    result = right_sum / test_sample_num
    print(result)


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data()
    train_sample_num, input_num = np.shape(train_x)  # 样本总数 输入层节点数
    output_num = 10  # 输出层节点数 0-9共10个数
    hide_num = 100   # 隐藏层节点数 28时正确率0.95- 100时正确率0.973- 400时正确率0.976-
    w1 = 0.2 * np.random.random((input_num, hide_num)) - 0.1  # 初始化输入层权矩阵
    w2 = 0.2 * np.random.random((hide_num, output_num)) - 0.1  # 初始化隐藏层权矩阵
    hide_offset = np.zeros(hide_num)  # 隐藏层偏置向量
    output_offset = np.zeros(output_num)  # 输出层偏置向量
    input_learn_rate = 0.2  # 输入层权值学习率
    hide_learn_rate = 0.2  # 隐藏层权值学习率

    for i in range(100):
        w1, w2, hide_offset, output_offset = train(w1, w2, hide_offset, output_offset)
        test(w1, w2, hide_offset, output_offset)
