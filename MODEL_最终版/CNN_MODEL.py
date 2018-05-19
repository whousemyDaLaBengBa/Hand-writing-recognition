import tensorflow as tf
import numpy as np
from PIL import Image
import os
import config as cf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(suppress=True)

class CNN_MODEL_ABCDE(object):
    def __init__(self):
        self.ACC = 0.9939

        self.graph=tf.Graph()

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(cf.MODEL_PATH + 'CNN.model-ABCDE.meta')


        self.sess = tf.Session(graph=self.graph)


        with self.sess.as_default():
            with self.graph.as_default():
                saver.restore(self.sess, cf.MODEL_PATH + 'CNN.model-ABCDE')

        self.x_input = self.graph.get_tensor_by_name('X_input:0')
        self.y_p = self.graph.get_tensor_by_name('Y_p:0')


    def use_model(self, img):
        x = np.array(img)
        if (x.shape != (32, 32)):
            return -1

        x = x.reshape([32, 32, 1])

        X = [x]

        Y_p = self.sess.run([self.y_p], feed_dict={self.x_input:X})
        
        Y_p = np.array(Y_p)[0][0]
        return Y_p


    #析构函数，用于归还sess
    def __del__( self ):  
        self.sess.close()

    
class CNN_MODEL_ABC(object):
    def __init__(self):
        self.ACC = 0.9918

        self.graph=tf.Graph()

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(cf.MODEL_PATH + 'CNN.model-ABC.meta')


        self.sess = tf.Session(graph=self.graph)


        with self.sess.as_default():
            with self.graph.as_default():
                saver.restore(self.sess, cf.MODEL_PATH + 'CNN.model-ABC')

        self.x_input = self.graph.get_tensor_by_name('X_input:0')
        self.y_p = self.graph.get_tensor_by_name('Y_p:0')


    def use_model(self, img):
        x = np.array(img)
        if (x.shape != (32, 32)):
            return -1

        x = x.reshape([32, 32, 1])

        X = [x]

        Y_p = self.sess.run([self.y_p], feed_dict={self.x_input:X})
        
        Y_p = np.array(Y_p)[0][0]
        return Y_p


    #析构函数，用于归还sess
    def __del__( self ):  
        self.sess.close()


class CNN_MODEL_CDE(object):
    def __init__(self):
        self.ACC = 0.9939

        self.graph=tf.Graph()

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(cf.MODEL_PATH + 'CNN.model-CDE.meta')


        self.sess = tf.Session(graph=self.graph)


        with self.sess.as_default():
            with self.graph.as_default():
                saver.restore(self.sess, cf.MODEL_PATH + 'CNN.model-CDE')

        self.x_input = self.graph.get_tensor_by_name('X_input:0')
        self.y_p = self.graph.get_tensor_by_name('Y_p:0')


    def use_model(self, img):
        x = np.array(img)
        if (x.shape != (32, 32)):
            return -1

        x = x.reshape([32, 32, 1])

        X = [x]

        Y_p = self.sess.run([self.y_p], feed_dict={self.x_input:X})
        
        Y_p = np.array(Y_p)[0][0]
        return Y_p


    #析构函数，用于归还sess
    def __del__( self ):  
        self.sess.close()


class CNN_MODEL(object):
    def __init__(self):
        self.ABCDE = CNN_MODEL_ABCDE()
        self.ABC = CNN_MODEL_ABC()
        self.CDE = CNN_MODEL_CDE()
        

    def use_model(self, img):
        x = np.array(img)
        if (x.shape != (32, 32)):
            return -1

        x = x.reshape([32, 32, 1])

        X = [x]

        Y_p_ABCDE = self.ABCDE.sess.run([self.ABCDE.y_p], feed_dict={self.ABCDE.x_input:X})
        Y_p_ABC = self.ABC.sess.run([self.ABC.y_p], feed_dict={self.ABC.x_input:X})
        Y_p_CDE = self.CDE.sess.run([self.CDE.y_p], feed_dict={self.CDE.x_input:X})
        
        Y_p_ABCDE = np.array(Y_p_ABCDE)[0][0]
        Y_p_ABC = np.array(Y_p_ABC)[0][0]
        Y_p_CDE = np.array(Y_p_CDE)[0][0]

        all_acc = self.ABCDE.ACC
        pre_acc = self.ABC.ACC
        las_acc = self.CDE.ACC

        Y_p = [0] * 5
        Y_p[0] = Y_p_ABCDE[0] * (all_acc/(all_acc + pre_acc)) + Y_p_ABC[0] * (pre_acc / (all_acc + pre_acc))
        Y_p[1] = Y_p_ABCDE[1] * (all_acc/(all_acc + pre_acc)) + Y_p_ABC[1] * (pre_acc / (all_acc + pre_acc))

        Y_p[2] = Y_p_ABCDE[2] * (all_acc/(all_acc + pre_acc + las_acc)) + Y_p_ABC[2] * (pre_acc / (all_acc + pre_acc + las_acc)) + Y_p_CDE[0] * (las_acc / (all_acc + pre_acc + las_acc))
        
        Y_p[3] = Y_p_ABCDE[3] * (all_acc/(all_acc + las_acc)) + Y_p_CDE[1] * (las_acc / (all_acc + las_acc))
        Y_p[4] = Y_p_ABCDE[4] * (all_acc/(all_acc + las_acc)) + Y_p_CDE[2] * (las_acc / (all_acc + las_acc))

        return Y_p



MODEL = CNN_MODEL()

img = Image.open('/home/ffb/Workspace/Python-srf/手写字识别/验证集/E/1081.jpg')

print(MODEL.use_model(img))