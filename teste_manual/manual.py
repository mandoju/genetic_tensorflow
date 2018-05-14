import tensorflow as tf
import numpy as np
import time
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import datasets
from sklearn.model_selection import train_test_split


def diag_block_mat_slicing(L):
    shp = L[0].shape
    N = len(L)
    r = range(N)
    out = np.zeros((N,shp[0],N,shp[1]),dtype='f')
    out[r,:,r,:] = L
    return out.reshape(np.asarray(shp)*N)

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def forwardprop_hidden(w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(w_1)  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=1)


def run_simple_graph():
    # first, create a TensorFlow constant
    const = tf.constant(2.0, name="const")

    # create TensorFlow variables
    b = tf.Variable(2.0, name='b')
    c = tf.Variable(1.0, name='c')

    # now create some operations
    d = tf.add(b, c, name='d')
    e = tf.add(c, 2, name='e')
    a = tf.multiply(d, e, name='a')

    # setup the variable initialisation
    init_op = tf.global_variables_initializer()

    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        # compute the output of the graph
        a_out = sess.run(a)
        print("Variable a is {}".format(a_out))


def run_simple_graph_multiple():
    # first, create a TensorFlow constant
    const = tf.constant(2.0, name="const")

    # create TensorFlow variables
    b = tf.placeholder(tf.float32, [None, 1], name='b')
    c = tf.Variable(1.0, name='c')

    # now create some operations
    d = tf.add(b, c, name='d')
    e = tf.add(c, 2, name='e')
    a = tf.multiply(d, e, name='a')

    # setup the variable initialisation
    init_op = tf.global_variables_initializer()

    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        # compute the output of the graph
        a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
        print("Variable a is {}".format(a_out))


def simple_with_tensor_board():
    const = tf.constant(2.0, name="const")

    # Create TensorFlow variables
    b = tf.Variable(2.0, name='b')
    c = tf.Variable(1.0, name='c')

    # now create some operations
    d = tf.add(b, c, name='d')
    e = tf.add(c, const, name='e')
    a = tf.multiply(d, e, name='a')

    # setup the variable initialisation
    init_op = tf.global_variables_initializer()

    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        # compute the output of the graph
        a_out = sess.run(a)
        print("Variable a is {}".format(a_out))
        train_writer = tf.summary.FileWriter('C:\\Users\\Andy\\PycharmProjects')
        train_writer.add_graph(sess.graph)


def nn_example(neural_networks):
    train_X, test_X, train_y, test_y = get_iris_data()

    #Defining number of layers
    number_neural_networks = len(neural_networks)
    number_neural_networks_remaining = number_neural_networks
    print(number_neural_networks)
    layer_size = 0
    for neural_network in neural_networks:
        if(len(neural_network) > layer_size):
            layer_size = len(neural_network)

    layers_sizes = []
    for neural_network in neural_networks:
        layers_sizes.append(len(neural_network))

    #Merging neural networks
    w = []
    w.append(tf.Variable(np.concatenate([item[0] for item in neural_networks],1)))
    for i in range(layer_size-1):
        w.append(tf.Variable(diag_block_mat_slicing([item[i+1] for item in neural_networks])))
        #print(diag_block_mat_slicing([item[i+1] for item in neural_networks]))

    # Layer's sizes
    x_size = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
    y_size = train_y.shape[1]  # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    #w_1 = tf.Variable(weight_1)
    #w_2 = tf.Variable(weight_2)

    # Forward propagation
    yhat = forwardprop(X, w[0], w[1])
    for i in range(layer_size-2):
        #yhat = tf.Print(yhat,[yhat],message="Yhat é: ")
        yhat = forwardprop_hidden(yhat, w[i+2])

    #yhat = tf.Print(yhat, [yhat], message="Yhat é: ")

    split0 = yhat[:, 0:int(((yhat.get_shape().as_list()[1])-1)/len(neural_networks))]
    predict = tf.argmax(split0, axis=1)
    predict_yhat = tf.argmax(yhat, axis=1)
    if(len(neural_networks) > 1):
       split1 = yhat[:, int((yhat.get_shape().as_list()[1]/len(neural_networks))):yhat.get_shape().as_list()[1]]
       predict_2 = tf.argmax(split1, axis=1)

    for i in range(number_neural_networks):
        split = yhat[:, int(i*(yhat.get_shape().as_list()[1]/len(neural_networks))):int((i+1)*((yhat.get_shape().as_list()[1]) - 1) / len(neural_networks))]
        predict = tf.argmax(split, axis=1)

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)


    for i in range(number_neural_networks):
        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict[i], feed_dict={X: train_X, y: train_y}))

        test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict[i], feed_dict={X: test_X, y: test_y}))
        print("train accuracy = %.2f%%, test accuracy = %.2f%%"
            % (100. * train_accuracy, 100. * test_accuracy))

    #if(len(neural_networks) > 1):

    #    train_accuracy_2 = np.mean(np.argmax(train_y, axis=1) ==
    #                             sess.run(predict_2, feed_dict={X: train_X, y: train_y}))

    #    test_accuracy_2 = np.mean(np.argmax(test_y, axis=1) ==
    #                             sess.run(predict_2, feed_dict={X: test_X, y: test_y}))


    #print("train accuracy = %.2f%%, test accuracy = %.2f%%"
    #      % (100. * train_accuracy, 100. * test_accuracy))
    #if(len(neural_networks) > 1):
    #    print("train accuracy = %.2f%%, test accuracy = %.2f%%"
    #          % (100. * train_accuracy_2, 100. * test_accuracy_2))
    #print(sess.run(predict, feed_dict={X: train_X, y: train_y}))
    #if(len(neural_networks) > 1):
    #    print(sess.run(predict_2, feed_dict={X: train_X, y: train_y}))

    #print("os pesos são:")
    #print(sess.run(w, feed_dict={X: train_X, y: train_y}))

    #print("o yhat devia ser:")
    #print(sess.run(yhat, feed_dict={X: train_X, y: train_y}))

    #print("o teste devia ser:")
    #print(np.argmax(train_y, axis=1))
    sess.close()

if __name__ == "__main__":
    # run_simple_graph()
    # run_simple_graph_multiple()
    # simple_with_tensor_board()

    w_1 = np.array([[1.0,2.0,3.0,4.0],[1.0,1.0,1.0,1.0],[1.0,1.0,3.0,4.0],[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0]],dtype='f')
    w_2 = np.array([[1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,10.0,1.0,1.0,1.0],[1.0,1.0,10.0,1.0,1.0,1.0],[1.0,1.0,10.0,1.0,1.0,1.0]],dtype='f')
    w_3 = np.array([[1.0,1.0,1.0],[1.0,1.0,10.0],[1.0,1.0,10.0],[1.0,1.0,10.0],[1.0,1.0,10.0],[1.0,1.0,10.0]],dtype='f')


    w_1_2 = np.array([[10.0, 0.0, 0.0, 15.0], [10.0, 5.0, 0.0, 10.0], [10.0, 10.0, 0.0, 10.0], [0.0, 3.0, 0.0, 10.0], [0.0, 0.0, 0.0, 10.0]],dtype='f')
    w_2_2 = np.array([[5.0, 3.0, 10.0, 10.0, 5.0, 6.0], [0.0, 0.0, 0.0, 10.0, 5.0, 6.0] , [0.0, 5.0, 0.0, 10.0, 5.0, 6.0], [0.0, 0.0, 0.0, 10.0, 5.0, 6.0]],dtype='f')
    w_3_2 = np.array([[1.0,1.0,1.0],[1.0,1.0,10.0],[1.0,1.0,10.0],[1.0,1.0,10.0],[1.0,1.0,10.0],[1.0,1.0,10.0]],dtype='f')

    #w_1 = diag_block_mat_slicing(( w_1,w_1_2) )
    #w_1 = np.concatenate([w_1,w_1_2],1)
    #w_2 = diag_block_mat_slicing( (w_2, w_2_2) )

    neural_network_1 = [w_1,w_2,w_3]
    neural_network_2 = [w_1_2,w_2_2,w_3_2]

    start = time.time()
    nn_example([neural_network_1,neural_network_2,neural_network_1,neural_network_2,neural_network_1,neural_network_2,neural_network_1,neural_network_2,neural_network_1,neural_network_2,neural_network_1,neural_network_2,neural_network_1,neural_network_2,neural_network_1,neural_network_2,neural_network_1,neural_network_2,neural_network_1,neural_network_2])
    end = time.time()
    print(end - start)

    start = time.time()
    nn_example(
        [neural_network_1, neural_network_2, neural_network_1, neural_network_2, neural_network_1, neural_network_2,
         neural_network_1, neural_network_2, neural_network_1, neural_network_2, neural_network_1, neural_network_2,
         neural_network_1, neural_network_2, neural_network_1, neural_network_2, neural_network_1, neural_network_2,
         neural_network_1, neural_network_2])
    end = time.time()
    print(end - start)

    start = time.time()
    nn_example([neural_network_1])
    end = time.time()
    print(end - start)

    start = time.time()
    nn_example([neural_network_2])
    end = time.time()
    print(end - start)
