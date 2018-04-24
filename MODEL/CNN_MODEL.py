import config as cf
import tensorflow as tf
import numpy as np
from PIL import Image
import os

class DATA():

    def get_branch_data(self, st, en):
        LEN = 0
        X = [None] * ((en - st + 10) * 5)
        Y = [None] * ((en - st + 10) * 5)

        for i in range(st, en):
            for k in range(5):

                FILE_DIR = cf.DATA_PATH + '训练集/' + chr(ord('A') + k) + '/'
                FILE_PATH = FILE_DIR + str(i) + '.jpg'

                img = Image.open(FILE_PATH)
                img_arr = np.array(img)

                if (img_arr.shape == (32, 32)):
                    X[LEN] = img_arr
                    Y[LEN] = [0] * 5

                    Y[LEN][k] = 1

                    LEN = LEN + 1

        X = X[: LEN]
        Y = Y[: LEN]

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    
    def get_train_data(self, size=19500):

        BRANCH_NUM = int(size/(cf.BRANCH_SIZE * 5))

        X = [None] * BRANCH_NUM
        Y = [None] * BRANCH_NUM

        for k in range(BRANCH_NUM):
            print(k)
            X[k], Y[k] = self.get_branch_data(k * cf.BRANCH_SIZE, (k + 1) * cf.BRANCH_SIZE)


        return X, Y


    def get_test_data(self,size=450):
        LEN = 0
        X = [None] * size
        Y = [None] * size

        for i in range(int(size / 5)):

            for k in range(5):
                FILE_DIR = cf.DATA_PATH + '验证集/' + chr(ord('A') + k) + '/'
                FILE_PATH = FILE_DIR + str(i) + '.jpg'

                img = Image.open(FILE_PATH)
                img_arr = np.array(img)

                if (img_arr.shape == (32, 32)):
                    X[LEN] = img_arr
                    Y[LEN] = [0] * 5

                    Y[LEN][k] = 1

                    LEN = LEN + 1

        X = X[: LEN]
        Y = Y[: LEN]

        X = np.array(X)
        Y = np.array(Y)
        return X, Y
    

'''
测试DATA类
data = DATA()
X, Y = data.get_train_data(10000)

x_test, y_test = data.get_test_data(450)
print(x_test.shape)
print(y_test.shape)
'''