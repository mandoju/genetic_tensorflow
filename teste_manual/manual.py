import tensorflow as tf
import numpy as np
import time
from itertools import filterfalse
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import datasets
from sklearn.model_selection import train_test_split



def block_diagonal(matrices, dtype=tf.float32):
  r"""Constructs block-diagonal matrices from a list of batched 2D tensors.

  Args:
    matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
      matrices with the same batch dimension).
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.
  Returns:
    A matrix with the input matrices stacked along its main diagonal, having
    shape [..., \sum_i N_i, \sum_i M_i].

  """
  matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
  blocked_rows = tf.Dimension(0)
  blocked_cols = tf.Dimension(0)
  batch_shape = tf.TensorShape(None)
  for matrix in matrices:
    full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
    batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
    blocked_rows += full_matrix_shape[-2]
    blocked_cols += full_matrix_shape[-1]
  ret_columns_list = []
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    ret_columns_list.append(matrix_shape[-1])
  ret_columns = tf.add_n(ret_columns_list)
  row_blocks = []
  current_column = 0
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    row_before_length = current_column
    current_column += matrix_shape[-1]
    row_after_length = ret_columns - current_column
    row_blocks.append(tf.pad(
        tensor=matrix,
        paddings=tf.concat(
            [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
             [(row_before_length, row_after_length)]],
            axis=0)))
  blocked = tf.concat(row_blocks, -2)
  blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
  return blocked

def diag_block_mat_slicing(L):
    shp = L[0].shape
    N = len(L)
    r = range(N)
    out = np.zeros((N, shp[0], N, shp[1]), dtype='f')
    out[r, :, r, :] = L
    return out.reshape(np.asarray(shp) * N)

def diag_block_mat_boolindex(L):
    shp = L[0].shape
    mask = np.kron(np.eye(len(L)), np.ones(shp))==1
    out = np.zeros(np.asarray(shp)*len(L),dtype='f')
    out[mask] = np.concatenate(L).ravel()
    return out


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)


def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


def forwardprop_hidden(w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h = tf.nn.sigmoid(w_1)  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris = datasets.load_iris()
    data = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M = data.shape
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


def run_matmul_test():
    # first, create a TensorFlow constant
    const2d = tf.constant([[1,0,0],[0,1,0],[0,0,1]], name="2d")
    const3d = tf.constant([[[0,0,0],[1,1,1],[2,2,2]],[[3,3,3],[4,4,4],[5,5,5]]], name="3d")

    # create TensorFlow variables
    #b = tf.Variable(2.0, name='b')
    #c = tf.Variable(1.0, name='c')

    # now create some operations
    #d = tf.add(b, c, name='d')
    #e = tf.add(c, 2, name='e')
    #a = tf.multiply(d, e, name='a')

    a = tf.matmul(const2d,const3d,name="matmul")
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
    print("com estrutura")
    train_X, test_X, train_y, test_y = get_iris_data()

    # Defining number of layers
    #predict = []
    number_neural_networks = len(neural_networks)
    number_neural_networks_remaining = number_neural_networks
    print("número de redes: %d" % (number_neural_networks))
    layer_size = 0
    for neural_network in neural_networks:
        if (len(neural_network) > layer_size):
            layer_size = len(neural_network)

    layers_sizes = []
    neural_networks.sort(key=len)
    for neural_network in neural_networks:
        layers_sizes.append(len(neural_network))

    # Merging neural networks
    start = time.time()

    w = []

    w.append(tf.Variable(np.concatenate([item[0] for item in neural_networks], 1)))
    neural_networks_remaining = list(neural_networks)
    i=0

    while(1):
        w.append(tf.Variable(block_diagonal(
            [item[i + 1] for item in neural_networks_remaining])))
        for remaining_neural_network in list(neural_networks_remaining):
            if len(remaining_neural_network) < (i + 3):
                neural_networks_remaining.remove(remaining_neural_network)
        if len(neural_networks_remaining) == 0:
            break
        #print(w[i+1])
        i = i + 1

    end = time.time()
    tempo = (end - start)
    #print(f"demorei {tempo} fazendo merge na estrutura")

    # Layer's sizes
    x_size = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
    y_size = train_y.shape[1]  # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations


    # Forward propagation
    yhat = forwardprop(X, w[0], w[1])
    predict_list = []
    remaining_neural_network = len(neural_networks)
    for i in range(layer_size - 2):
        if yhat.get_shape().as_list()[1] != w[i + 2].get_shape().as_list()[0]:
            #print("---- problema ----")
            #print("Yhat possui shape: ",yhat.get_shape().as_list())
            #print("Weight ", (i+2) ," possui shape: ", w[i + 2].get_shape().as_list())
            #print("---- problema ----")
            #print("tentando reduzir")
            size_of_slice = int(yhat.get_shape().as_list()[1] - w[i + 2].get_shape().as_list()[0])
            slice = yhat[:,size_of_slice]

            for j in range(int(size_of_slice/y_size)):
                #print(y_size)
                remaining_neural_network -= 1

                split = yhat[:, int(j*y_size):int(y_size + (y_size*j))]
                predict_list.append(split)


            yhat = yhat[:,size_of_slice:yhat.get_shape().as_list()[1]]
            #print("O shape agora ficou", yhat.get_shape())
        yhat = forwardprop_hidden(yhat, w[i + 2])

    splits = []
    #print("O shape do yhat remanecente é ", yhat.get_shape())
    for i in range(remaining_neural_network):
        split = yhat[:, int(i * (yhat.get_shape().as_list()[1] / remaining_neural_network)):int(
            (i + 1) * ((yhat.get_shape().as_list()[1]) ) / remaining_neural_network)]
        splits.append(split)

    predict_list = predict_list + splits
    predict = tf.stack(predict_list,1)



    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_accuracies = sess.run(predict, feed_dict={X: train_X, y: train_y})
    test_accuracies = sess.run(predict, feed_dict={X: test_X, y: test_y})
    final_yhat = sess.run(yhat,feed_dict={X: test_X, y: test_y})

    for i in range(number_neural_networks):
        train_accuracy = np.mean(np.argmax(train_y, axis=1) == train_accuracies[i])

        test_accuracy = np.mean(np.argmax(test_y, axis=1) == test_accuracies[i])
        #print("train accuracy = %.2f%%, test accuracy = %.2f%%"
        #      % (100. * train_accuracy, 100. * test_accuracy))

        #print(len(train_accuracies[i]))

        #print(len(final_yhat))

    sess.close()
def nn_example_same_size(neural_networks):
    print("com estrutura")
    train_X, test_X, train_y, test_y = get_iris_data()

    # Defining number of layers
    #predict = []
    number_neural_networks = len(neural_networks)
    number_neural_networks_remaining = number_neural_networks
    print("número de redes: %d" % (number_neural_networks))
    layer_size = 0
    for neural_network in neural_networks:
        if (len(neural_network) > layer_size):
            layer_size = len(neural_network)

    layers_sizes = []
    neural_networks.sort(key=len)
    for neural_network in neural_networks:
        layers_sizes.append(len(neural_network))

    # Merging neural networks
    start = time.time()

    w = []

    w.append(tf.Variable(np.concatenate([item[0] for item in neural_networks], 1)))
    neural_networks_remaining = list(neural_networks)
    i=0

    while(1):
        w.append(tf.Variable(diag_block_mat_boolindex(
            [item[i + 1] for item in neural_networks_remaining])))
        for remaining_neural_network in list(neural_networks_remaining):
            if len(remaining_neural_network) < (i + 3):
                neural_networks_remaining.remove(remaining_neural_network)
        if len(neural_networks_remaining) == 0:
            break
        #print(w[i+1])
        i = i + 1

    end = time.time()
    tempo = (end - start)
    #print(f"demorei {tempo} fazendo merge na estrutura")

    # Layer's sizes
    x_size = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
    y_size = train_y.shape[1]  # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations


    # Forward propagation
    yhat = forwardprop(X, w[0], w[1])
    predict_list = []
    remaining_neural_network = len(neural_networks)
    for i in range(layer_size - 2):
        yhat = forwardprop_hidden(yhat, w[i + 2])

    splits = []
    #print("O shape do yhat remanecente é ", yhat.get_shape())
    for i in range(remaining_neural_network):
        split = yhat[:, int(i * (yhat.get_shape().as_list()[1] / remaining_neural_network)):int(
            (i + 1) * ((yhat.get_shape().as_list()[1]) ) / remaining_neural_network)]
        splits.append(split)

    predict_list = predict_list + splits
    predict = tf.stack(predict_list,1)



    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_accuracies = sess.run(predict, feed_dict={X: train_X, y: train_y})
    test_accuracies = sess.run(predict, feed_dict={X: test_X, y: test_y})
    final_yhat = sess.run(yhat,feed_dict={X: test_X, y: test_y})

    for i in range(number_neural_networks):
        train_accuracy = np.mean(np.argmax(train_y, axis=1) == train_accuracies[i])

        test_accuracy = np.mean(np.argmax(test_y, axis=1) == test_accuracies[i])
        #print("train accuracy = %.2f%%, test accuracy = %.2f%%"
        #      % (100. * train_accuracy, 100. * test_accuracy))

        #print(len(train_accuracies[i]))

        #print(len(final_yhat))

    sess.close()


def nn_example_without_struct(neural_networks):
    print("sem estrutura")
    train_X, test_X, train_y, test_y = get_iris_data()

    # Defining number of layers
    #predict = []
    number_neural_networks = len(neural_networks)
    number_neural_networks_remaining = number_neural_networks
    print("número de redes: %d" % (number_neural_networks))
    layer_size = 0
    for neural_network in neural_networks:
        if (len(neural_network) > layer_size):
            layer_size = len(neural_network)

    layers_sizes = []
    neural_networks.sort(key=len)
    for neural_network in neural_networks:
        layers_sizes.append(len(neural_network))

    # Merging neural networks


    # Layer's sizes
    x_size = train_X.shape[1]  # Number of input nodes: 4 features and 1 bias
    y_size = train_y.shape[1]  # Number of outcomes (3 iris flowers)

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations


    for neural_network in neural_networks:
        w = neural_network

        # Forward propagation
        yhat = forwardprop(X, w[0], w[1])
        predict_list = []
        remaining_neural_network = len(neural_networks)
        for i in range(len(w) - 2):

            yhat = forwardprop_hidden(yhat, w[i + 2])


#        predict_list = predict_list + splits
        predict = tf.stack(yhat)



    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_accuracies = sess.run(predict, feed_dict={X: train_X, y: train_y})
    test_accuracies = sess.run(predict, feed_dict={X: test_X, y: test_y})
    final_yhat = sess.run(yhat,feed_dict={X: test_X, y: test_y})

    for i in range(number_neural_networks):
        train_accuracy = np.mean(np.argmax(train_y, axis=1) == train_accuracies[i])

        test_accuracy = np.mean(np.argmax(test_y, axis=1) == test_accuracies[i])
        #print("train accuracy = %.2f%%, test accuracy = %.2f%%"
        #      % (100. * train_accuracy, 100. * test_accuracy))

        #print(len(train_accuracies[i]))

        #print(len(final_yhat))

    print("fim");
    sess.close()

if __name__ == "__main__":
    # run_simple_graph()
    # run_simple_graph_multiple()
    # simple_with_tensor_board()

    run_matmul_test();
#layer_size = [4,5,6,7,8,9,10,3];
#
#w_1 = np.array(
#    [[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
#    dtype='f')
#w_1 = np.random.randn(5, 4 ).astype('f') * 0.01;
#w_2 = np.random.randn(4, 10000 ).astype('f') * 0.01;
#w_3 = np.random.randn(10000, 3 ).astype('f') * 0.01;
#
#
#w_1_2 = np.random.randn(5, 4 ).astype('f') * 0.01;
#w_2_2 = np.random.randn(4, 10000 ).astype('f') * 0.01;
#w_3_2 = np.random.randn(10000, 3 ).astype('f') * 0.01;
#w_4_2 = np.random.randn(3, 3).astype('f') * 0.01;
#
#neural_network_1 = [w_1, w_2, w_3]
#neural_network_2 = [w_1_2, w_2_2, w_3_2]
#
#start = time.time()
#nn_example(
#    [neural_network_1, neural_network_2])
#end = time.time()
#print(end - start)
#
#start = time.time()
#nn_example_without_struct(
#    [neural_network_1, neural_network_2])
#end = time.time()
#print(end - start)
#
#
#start = time.time()
#nn_example(
#    [neural_network_1, neural_network_2])
#end = time.time()
#print(end - start)
#
#start = time.time()
#nn_example(
#    [neural_network_1, neural_network_1, neural_network_1, neural_network_1, neural_network_1, neural_network_1,
#     neural_network_1, neural_network_1, neural_network_1, neural_network_1, neural_network_2, neural_network_2,
#     neural_network_2, neural_network_2, neural_network_2, neural_network_2, neural_network_2, neural_network_2,
#     neural_network_2, neural_network_2])
#end = time.time()
#print(end - start)
#
#start = time.time()
#nn_example_without_struct(
#    [neural_network_1, neural_network_2, neural_network_1, neural_network_2, neural_network_1, neural_network_2,
#     neural_network_1, neural_network_2, neural_network_1, neural_network_2, neural_network_1, neural_network_2,
#     neural_network_1, neural_network_2, neural_network_1, neural_network_2, neural_network_1, neural_network_2,
#     neural_network_1, neural_network_2])
#end = time.time()
#print(end - start)
#
#