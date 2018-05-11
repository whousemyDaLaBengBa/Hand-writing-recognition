import tensorflow as tf
import numpy as np

a = tf.Variable([1, 2, 3, 4],dtype=tf.float32)



b = a * 5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out = sess.run([b])
    print(out)