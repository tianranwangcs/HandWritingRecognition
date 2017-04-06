from data_util import DataUtils
import numpy as np
import tensorflow as tf


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

    print('load data success...')
    return train_x, train_y, test_x, test_y


def tensor_flow(train_x, train_y, test_x, test_y):
    # place holder for input training data
    x = tf.placeholder(tf.float32, [None, 784])
    # variable is another place holder but its value is variable, #_#
    # weight
    w = tf.Variable(tf.zeros([784, 10]))
    # bias
    b = tf.Variable(tf.zeros([10]))
    # softmax func, yet can say activate func
    y = tf.nn.softmax(tf.matmul(x, w) + b)
    # place holder for input training label
    # y_ is actual while y is our output
    y_ = tf.placeholder(tf.float32, [None,10])
    # calc error between output and actual
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    # learning rate is 0.01
    # use gradient descent algorithm to minimize error
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(100):
        batch_xs= train_x.next_batch(100)
        batch_ys=train_y.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # test
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_data()
    train_y_=[]
    test_y_=[]
    for i in range(60000):
        label=np.zeros(10)
        label[train_y[i]]=1
        train_y_.append(label)
    for i in range(10000):
        label=np.zeros(10)
        label[test_y[i]]=1
        test_y_.append(label)
    tensor_flow(train_x,train_y_,test_x,test_y_)