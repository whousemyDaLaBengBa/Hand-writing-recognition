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
                    img_arr = img_arr.reshape((32, 32, 1))
                    X[LEN] = img_arr
                    Y[LEN] = [0] * 5

                    Y[LEN][k] = 1

                    LEN = LEN + 1

        X = X[: LEN]
        Y = Y[: LEN]

        X = np.array(X)
        Y = np.array(Y)

        return X, Y

    
    def get_train_data(self, size=5000):

        BRANCH_NUM = int(size/(cf.BRANCH_SIZE * 5))

        X = [None] * BRANCH_NUM
        Y = [None] * BRANCH_NUM

        for k in range(BRANCH_NUM):
            print(k)
            X[k], Y[k] = self.get_branch_data(k * cf.BRANCH_SIZE, (k + 1) * cf.BRANCH_SIZE)


        return X, Y


    def get_test_data(self):
        LEN = 0
        X = [None] * 500
        Y = [None] * 500

        for k in range(5):
            FILE_DIR = cf.DATA_PATH + '验证集/' + chr(ord('A') + k) + '/'
            FILE_LIS = os.listdir(FILE_DIR)

            for FILE_PATH_1 in FILE_LIS:

                FILE_PATH = FILE_DIR + FILE_PATH_1

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
        X = X.reshape((-1,32,32,1))
        Y = np.array(Y)
        return X, Y
    

'''
#测试DATA类
data = DATA()
X, Y = data.get_train_data(10000)

x_test, y_test = data.get_test_data(450)
print(x_test.shape)
print(y_test.shape)
'''

class MODEL():
    def __init__(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 1], name='X_input')
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, 5])
        self.data = DATA()
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        self.x_test = None


    def get_layer_t(self, pre_layer, cannel_size, conv_size, stride = 2, w_alpha = 0.01, b_alpha = 0.01):
        pre_shape = pre_layer.get_shape()

        W = pre_shape[1]
        H = pre_shape[2]
        C = pre_shape[3]#这里w,h可能写反了，mark一下

        w = tf.Variable(w_alpha*tf.random_normal([conv_size, conv_size, int(C), cannel_size]), dtype=tf.float32)
        b = tf.Variable(b_alpha*tf.random_normal([cannel_size]), dtype=tf.float32)

        conv = tf.nn.tanh(tf.nn.bias_add(tf.nn.conv2d(pre_layer, w, strides=[1, stride, stride, 1], padding='SAME'), b))
        conv = tf.nn.dropout(conv, self.keep_prob)
        return conv

    
    def get_layer_r(self, pre_layer, cannel_size, conv_size, stride = 2, w_alpha = 0.01, b_alpha = 0.01):
        pre_shape = pre_layer.get_shape()

        W = pre_shape[1]
        H = pre_shape[2]
        C = pre_shape[3]#这里w,h可能写反了，mark一下

        w = tf.Variable(w_alpha*tf.random_normal([conv_size, conv_size, int(C), cannel_size]), dtype=tf.float32)
        b = tf.Variable(b_alpha*tf.random_normal([cannel_size]), dtype=tf.float32)

        conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pre_layer, w, strides=[1, stride, stride, 1], padding='SAME'), b))
        conv = tf.nn.dropout(conv, self.keep_prob)
        return conv


    def get_layer(self, pre_layer, cannel_size, conv_size, stride = 1, w_alpha = 0.01, b_alpha = 0.01):
        pre_shape = pre_layer.get_shape()

        W = pre_shape[1]
        H = pre_shape[2]
        C = pre_shape[3]#这里w,h可能写反了，mark一下

        w = tf.Variable(w_alpha*tf.random_normal([conv_size, conv_size, int(C), cannel_size]), dtype=tf.float32)
        b = tf.Variable(b_alpha*tf.random_normal([cannel_size]), dtype=tf.float32)

        conv = tf.nn.bias_add(tf.nn.conv2d(pre_layer, w, strides=[1, stride, stride, 1], padding='SAME'), b)
        conv = tf.nn.dropout(conv, self.keep_prob)
        return conv


    def assemble_graph(self, w_alpha = 0.01, b_alpha=0.01):

        #第一层
        conv1_1 = self.get_layer_r(self.X, 16, 2, stride=1)
        conv1_2 = self.get_layer_r(conv1_1, 16, 2, stride=1)

        conv1 = tf.add(conv1_2, self.X)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        #第一层shape(size, 16, 16, 16)


        #第二层
        conv1_s = tf.stack([conv1, conv1],axis=0)
        conv1_s = tf.reshape(conv1_s, [-1, 16, 16, 32])
        conv2_1 = self.get_layer_r(conv1, 32, 2, stride=1)
        conv2_2 = self.get_layer_r(conv2_1, 32, 2, stride=1)

        conv2 = tf.add(conv2_2, conv1_s)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        #第二层shape(size, 8, 8, 32)


        #第三层
        conv2_s = tf.stack([conv2, conv2],axis=0)
        conv2_s = tf.reshape(conv2_s, [-1, 8, 8, 64])
        conv3_1 = self.get_layer_t(conv2, 64, 2, stride=1)
        conv3_2 = self.get_layer_t(conv3_1, 64, 2, stride=1)

        conv3 = tf.add(conv3_2, conv2_s)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #第三层shape(size, 4, 4, 64)

        cnn_out = tf.reshape(conv3, [-1, 4 * 4 * 64])

          
        #全连接层
        w_out = tf.Variable(w_alpha*tf.random_normal([4 * 4 * 64, 5]), dtype=tf.float32)
        b_out = tf.Variable(w_alpha*tf.random_normal([5]), dtype=tf.float32)

        Y_p =  tf.add(tf.matmul(cnn_out, w_out), b_out)
        #shape为(size, 5)
        
        '''
        #测试用代码
        x,y = self.data.get_train_data()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run([conv3], feed_dict={self.X:x[0], self.keep_prob:1.0})
            print(np.array(out).shape)
        '''
        return Y_p

    
    def get_acc(self, Y_p, Y):
        Y_p = Y_p[0]
        Y_p = np.array(Y_p)


        size, num = Y_p.shape
        print(size)
        print(num)
        
        ACC = 0
        SUM = size

        for k in range(size):
            poi = 0
            for i in range(num):
                if (Y_p[k][poi] < Y_p[k][i]):
                    poi = i
                
            if (Y[k][poi] == 1):
                ACC = ACC + 1
            else:
                '''
                img_arr = self.x_test[k]

                img_arr = img_arr.reshape(32, 32)
                img = Image.fromarray(img_arr).convert('L')
                img.show()
                print(' ')
                print(Y_p[k])
                print(Y[k])
                #print(img_arr.shape)
                '''
                pass

        return ACC/SUM 
        
        return 0
    
    def train(self, size = 5000, w_alpha = 0.01, b_alpha=0.01):
        #读入数据
        X, Y = self.data.get_train_data(size)
        x_test, y_test = self.data.get_test_data()

        self.x_test = x_test
        
        #获取神经网络输出与预测值
        BRANCH_NUM = len(X)
        print(BRANCH_NUM)
        network_output = self.assemble_graph(w_alpha, b_alpha)
        Y_p = tf.nn.softmax(network_output, name='Y_p')



        #计算loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=network_output, labels=self.Y)
        loss = tf.reduce_mean(loss)

        #设置op
        op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


        saver = tf.train.Saver()

        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())
            saver.restore(sess, cf.MODEL_PATH + 'CNN.model-2000')

            step = 0
            print('begin')
            
            while True: 
                _, L = sess.run([op,loss], feed_dict={self.X:X[step % BRANCH_NUM], self.Y:Y[step % BRANCH_NUM],self.keep_prob:0.5})
                
                if step % 20 == 0:
                    print(str(step) + '  ' + str(L))
                
                if step % 100 == 0:
                    y_p = sess.run([Y_p], feed_dict={self.X:x_test,self.keep_prob:1.0})
                    ypp = sess.run([Y_p], feed_dict={self.X:X[0],self.keep_prob:1.0})

                    acc_t = self.get_acc(ypp, Y[0])

                    acc = self.get_acc(y_p, y_test)
                    print(str(step) + ' ' + str(acc))
                    print(acc_t)
                

                if step % 2000 == 0:
                    saver.save(sess, cf.MODEL_PATH + 'CNN.model', global_step=step)

                step = step + 1



model = MODEL()
model.train()