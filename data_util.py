import numpy as np
import struct
# import matplotlib.pyplot as plt
# import os


# tutorial: http://www.csuldw.com/2016/02/25/2016-02-25-machine-learning-MNIST-dataset/
class DataUtils(object):
    """MNIST数据集加载
    输出格式为：numpy.array()

    使用方法如下
    from data_util import DataUtils
    def main():
        trainfile_X = '../dataset/MNIST/train-images.idx3-ubyte'
        trainfile_y = '../dataset/MNIST/train-labels.idx1-ubyte'
        testfile_X = '../dataset/MNIST/t10k-images.idx3-ubyte'
        testfile_y = '../dataset/MNIST/t10k-labels.idx1-ubyte'

        train_X = DataUtils(filename=trainfile_X).get_image()
        train_y = DataUtils(filename=trainfile_y).get_label()
        test_X = DataUtils(testfile_X).get_image()
        test_y = DataUtils(testfile_y).get_label()

        #以下内容是将图像保存到本地文件中
        #path_trainset = "../dataset/MNIST/imgs_train"
        #path_testset = "../dataset/MNIST/imgs_test"
        #if not os.path.exists(path_trainset):
        #    os.mkdir(path_trainset)
        #if not os.path.exists(path_testset):
        #    os.mkdir(path_testset)
        #DataUtils(outpath=path_trainset).out_image(train_X, train_y)
        #DataUtils(outpath=path_testset).out_image(test_X, test_y)

        return train_X, train_y, test_X, test_y
    """

    def __init__(self, filename=None, outpath=None):
        self._filename = filename
        self._outpath = outpath

        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'
        self._pictureBytes = '784B'  # 28*28Bytes
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes  # ‘>IIII’指的是使用大端法读取4个unsinged int 32 bit integer
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte

    def get_image(self):
        """
        将MNIST的二进制文件转换成像素特征数据
        """
        binfile = open(self._filename, 'rb')  # 以二进制方式打开文件
        buf = binfile.read()
        binfile.close()
        index = 0
        # introduction about pack: http://blog.csdn.net/ithomer/article/details/5974029
        numMagic, numImgs, numRows, numCols = struct.unpack_from(self._fourBytes2, buf, index)
        index += struct.calcsize(self._fourBytes)
        images = []
        for i in range(numImgs):
            imgVal = struct.unpack_from(self._pictureBytes2, buf, index)
            index += struct.calcsize(self._pictureBytes2)
            imgVal = list(imgVal)
            for j in range(len(imgVal)):
                if imgVal[j] > 1:
                    imgVal[j] = 1
            images.append(imgVal)
        return np.array(images)

    def get_label(self):
        """
        将MNIST中label二进制文件转换成对应的label数字特征
        """
        binfile = open(self._filename, 'rb')
        buf = binfile.read()
        binfile.close()
        index = 0
        magic, numItems = struct.unpack_from(self._twoBytes2, buf, index)
        index += struct.calcsize(self._twoBytes2)
        labels = [];
        for x in range(numItems):
            im = struct.unpack_from(self._labelByte2, buf, index)
            index += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)

    def out_image(self, arrX, arrY):
        """
        根据生成的特征和数字标号，输出png的图像
        """
        m, n = np.shape(arrX)
        # 每张图是28*28=784Byte
        for i in range(10):
            img = np.array(arrX[i])
            img = img.reshape(28, 28)
            outfile = str(i) + "_" + str(arrY[i]) + ".png"
            # plt.figure()
            plt.imshow(img, cmap='binary')  # 将图像黑白显示
            plt.savefig(self._outpath + "/" + outfile)
