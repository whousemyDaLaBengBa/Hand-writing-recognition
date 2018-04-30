import config as cf
import numpy as np
from PIL import Image
import os


def trans_train_data():
    for k in range(5):
        print(k)
        DIR_PATH = cf.DATA_PATH  + chr(k + ord('A')) + '/'
        FILE_LIS = os.listdir(DIR_PATH)

        LEN = 0

        for PATH in FILE_LIS:
            FILE_PATH = DIR_PATH + PATH
            img = Image.open(FILE_PATH)

            if (np.array(img).shape == (90, 90, 3)):
                img = img.resize((32, 32), Image.ANTIALIAS)

                r, g, b = img.split()
                r_arr = np.array(r)
                g_arr = np.array(g)
                b_arr = np.array(b)
                img_arr = 0.2989 * r_arr + 0.5870 * g_arr + 0.1140 * b_arr

                img = Image.fromarray(img_arr).convert('L')

                IMG_SAVE_PATH = cf.SAVE_PATH + chr(k + ord('A')) + '/' + str(LEN) + '.jpg'

                img.save(IMG_SAVE_PATH)

                LEN = LEN + 1



'''
def trans_valid_data():
    for k in range(41, 46):
        print(k)
        DIR_PATH = cf.DATA_PATH + 'valid/' + str(k) + '/'
        FILE_LIS = os.listdir(DIR_PATH)

        LEN = 0

        for PATH in FILE_LIS:
            FILE_PATH = DIR_PATH + PATH
            img = Image.open(FILE_PATH)

            if (np.array(img).shape == (128, 128, 3)):
                img = img.resize((32, 32), Image.ANTIALIAS)

                r, g, b = img.split()
                r_arr = np.array(r)
                g_arr = np.array(g)
                b_arr = np.array(b)
                img_arr = 0.2989 * r_arr + 0.5870 * g_arr + 0.1140 * b_arr

                img = Image.fromarray(img_arr).convert('L')

                IMG_SAVE_PATH = cf.SAVE_PATH + chr(ord('A') + k - 41) + '/' + str(LEN) + '.jpg'
                
                img.save(IMG_SAVE_PATH)

                LEN = LEN + 1
'''

trans_train_data()
#trans_valid_data()