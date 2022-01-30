import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model

import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

x = tf.Variable(3, name="x") 
y = tf.Variable(4, name="y") 
f=x*x*y+y+2


sess = tf.compat.v1.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

