import tensorflow as tf
import numpy as np 
from mutation import mutation , mutation_unbiased
from genetic_operators import generate_child_by_all
from layer import Layer
from datasets.datasets import get_sine_data


x = tf.placeholder("float", shape=[None, 1], name="X")
w1 = tf.get_variable('wd1',shape=(10,1,1),initializer=tf.random_normal_initializer(seed=1))
b1 = tf.get_variable('bd1',shape=(10,1),initializer=tf.random_normal_initializer(seed=1))
layer_1 = Layer(10,w1,b1,'wd')
#layer_1_saida = layer_1.run_fist_layer(x)

#resp = layer_1_saida
resp = []
for i in range(10):
    resp.append(tf.add(tf.matmul(x, w1[i]), b1[i]))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

xpts = np.random.rand(2) * 10
print( xpts[:, None])
train_x = xpts[:,None]
#train_x, train_y, test_x, test_y = get_sine_data()
resp_sess  = sess.run(resp,feed_dict={x: train_x})

print(resp_sess)

# def conv2d(x, W, b, strides=1):
#     # Conv2D wrapper, with bias and relu activation
#     x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
#     x = tf.nn.bias_add(x, b)
#     return tf.nn.relu(x)

# pai = tf.get_variable('w1', shape= (3,3,1,2), initializer=tf.random_normal_initializer(seed=1))
# pai_bias = tf.get_variable('b1', shape= (2), initializer=tf.random_normal_initializer(seed=1))
# mae = tf.get_variable('w2', shape= (3,3,1,2), initializer=tf.random_normal_initializer(seed=2))
# mae_bias = tf.get_variable('b2', shape= (2), initializer=tf.random_normal_initializer(seed=2))

# filho = tf.get_variable('w3', shape= (2,3,3,1,2), initializer=tf.random_normal_initializer(seed=2))
# filho_bias = tf.get_variable('b3', shape= (2,2), initializer=tf.random_normal_initializer(seed=2))

# X = tf.placeholder("float", shape=[None, 28, 28, 1], name="X")

# convolution = conv2d(X, pai , pai_bias)
# second_convolution = conv2d(X, mae, mae_bias)

# stack = tf.stack([pai,mae])
# #assign_filho = stack
# assign_filho = tf.assign(filho,stack)
# stack_bias = tf.stack([pai_bias,mae_bias])
# assign_filho_bias = tf.assign(filho_bias,stack_bias)
# #filho_conv = tf.map_fn( lambda x: filho[x], tf.range(2), dtype=tf.float32 )
# filho_conv = tf.map_fn( lambda x: conv2d(X, filho[x], filho_bias[x]), tf.range(2), dtype=tf.float32 )

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# train_x, train_y, test_x, test_y = get_mnist_data()
# train_x = train_x.reshape(-1, 28, 28, 1)
# saida_pai, saida_mae = sess.run([pai,mae])
# print("pai")
# print(np.mean(saida_pai))
# print("mae")
# print(np.mean(saida_mae))
# saida_conv = sess.run(convolution,feed_dict={X: train_x[0:128]})
# saida_conv_2 = sess.run(second_convolution, feed_dict={X: train_x[0:128]})
# print("Saida")
# print(np.mean(saida_conv))
# print("Saida 2")
# print(np.mean(saida_conv_2))

# session_assign = sess.run(assign_filho, feed_dict={X: train_x[0:128]} )
# print("filhos assigns")
# print(np.mean(session_assign[0]))
# print(np.mean(session_assign[1]))


# session_assign = sess.run(assign_filho_bias, feed_dict={X: train_x[0:128]} )
# print("filhos assigns")
# print(np.mean(session_assign[0]))
# print(np.mean(session_assign[1]))

# filho_session = sess.run(filho_conv, feed_dict={X: train_x[0:128]})
# print("filho session")
# print(np.mean(filho_session[0]))
# print(np.mean(filho_session[1]))