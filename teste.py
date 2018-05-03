import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


from tensorflow.examples.tutorials.mnist import input_data


#mnist = input_data.read_data_sets("MINIST_data/", one_hot=True)
def multilayer_perceptron(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


def input_evaluation_set():
    features = {'SepalLength': np.array([6.4, 5.0]),
                'SepalWidth':  np.array([2.8, 2.3]),
                'PetalLength': np.array([5.6, 3.3]),
                'PetalWidth':  np.array([2.2, 1.0])}
    labels = np.array([2, 1])
    return features, labels


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)


def input_fn(df,labels):
    feature_cols = {k:tf.constant(df[k].values,shape = [df[k].size,1]) for k in columns}
    label = tf.constant(labels.values, shape = [labels.size,1])
    return feature_cols,label

iris = pd.read_csv("./input/Iris.csv")



iris.iloc[:,1:4] = iris.iloc[:,1:4].astype(np.float32)
iris["Species"] = iris["Species"].map({"Iris-setosa":0,"Iris-virginica":1,"Iris-versicolor":2})

X_train, X_test, y_train, y_test = train_test_split(iris.iloc[:,1:5], iris["Species"], test_size=0.33, random_state=42)
columns = iris.columns[1:5]
feature_columns = [tf.contrib.layers.real_valued_column(k) for k in columns]


#learning_rate = 0.5
#epochs = 10
#batch_size = 100
#
# x = tf.placeholder(tf.float32, [1,784])
# y = tf.placeholder(tf.float32, [1,10])
#
#
# W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
# b1 = tf.Variable(tf.random_normal([300]), name='b1')
#
#
# W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
# b2 = tf.Variable(tf.random_normal([10]), name='b2')
# 
#
# hidden_out = tf.add(tf.matmul(x, W1), b1)
# hidden_out = tf.nn.relu(hidden_out)


n_hidden_1 = 38
n_input = X_train.shape[1]
n_classes = y_train.shape[0]

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

keep_prob = tf.placeholder("float")

init_op = tf.global_variables_initializer()

# start the session
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    # compute the output of the graph
    rand_array_x = np.random.rand(1, 784)
    rand_array_y = np.random.rand(1, 10)

    a_out = sess.run(hidden_out,feed_dict={x: rand_array_x,y: rand_array_y})
    print("Variable a is {}".format(a_out))