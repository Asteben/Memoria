import numpy as np
import tensorflow as tf

a = [1,2,3,4,5,6]
b = [2,3,4,5,6,7]

loss = tf.keras.losses.mean_squared_error(a, b)


print (loss.numpy())