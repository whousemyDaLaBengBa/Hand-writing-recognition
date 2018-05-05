import tensorflow as tf
import numpy as np
a = tf.Variable([[1, 2],[3, 4],[5, 6]])

shape = a.get_shape()
print(shape[0])
