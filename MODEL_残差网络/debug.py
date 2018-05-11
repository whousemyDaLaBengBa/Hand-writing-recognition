import tensorflow as tf
import numpy as np
a = tf.Variable([[1, 2],[3, 4],[5, 6]])

b = tf.Variable([10, 11])

c = tf.add(a, b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    out = sess.run([c])
    print(out)
