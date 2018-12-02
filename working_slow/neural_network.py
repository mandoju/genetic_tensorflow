import tensorflow as tf
import numpy as np


def calculate_fitness(neural_networks):
    return nn_example_without_struct(neural_networks)


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
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype)
                for matrix in matrices]
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
    mask = np.kron(np.eye(len(L)), np.ones(shp)) == 1
    out = np.zeros(np.asarray(shp) * len(L), dtype='f')
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

    with tf.name_scope('foward_propagation') as scope:
        h = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
        yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


def forwardprop_hidden(w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    with tf.name_scope('foward_propagation_hidden') as scope:
        h = tf.nn.sigmoid(w_1)  # The \sigma function
        yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat


# def get_iris_data():
#    """ Read the iris data set and split them into training and test sets """
#    iris = datasets.load_iris()
#    data = iris["data"]
#    target = iris["target"]

# Prepend the column of 1s for bias
#    N, M = data.shape
#    all_X = np.ones((N, M + 1 ))
#    all_X[:, 1:] = data

# Convert into one-hot vectors
#    num_labels = len(np.unique(target))
#    all_Y = np.eye(num_labels)[target]  # One liner trick!
#    return train_test_split(all_X, all_Y, test_size=0.33, random_state=1)

def get_mnist_data():
    mnist = tf.keras.datasets.mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = np.array(train_X).reshape(len(train_X), 784)
    # Prepend the column of 1s for bias
    N, M = train_X.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = train_X

    num_labels = len(np.unique(train_y))
    all_Y = np.eye(num_labels)[train_y]  # One liner trick!
    # a,b,c,d = train_test_split(all_X, all_Y, test_size=0.00, random_state=0)
    return (all_X, all_X, all_Y, all_Y)


def nn_example_without_struct(neural_networks):
    print("Rodando rede neural")

    # iris
    # train_X, test_X, train_y, test_y = get_iris_data()

    # mnist
    with tf.name_scope('Fitness') as scope:

        train_X, test_X, train_y, test_y = get_mnist_data()
        # Defining number of layers
        number_neural_networks = len(neural_networks)
        number_neural_networks_remaining = number_neural_networks
        print("numero de redes: %d" % (number_neural_networks))

        # Layer's sizes
        # Number of input nodes: x features and 1 bias
        x_size = train_X.shape[1]
        y_size = train_y.shape[1]  # Number of outcomes (3 iris flowers)
        # Symbols
        X = tf.placeholder("float", shape=[None, x_size], name="X")
        y = tf.placeholder("float", shape=[None, y_size], name="Y")
        # Weight initializations
        i = 0
        predicts = []

        for neural_network in neural_networks:
            with tf.name_scope('rede_neural_' + str(i)) as scope:
                w = neural_network

                # Forward propagation
                yhat = forwardprop(X, w[0], w[1])
                for i in range(len(w) - 2):
                    yhat = forwardprop_hidden(yhat, w[i + 2])

                predicts.append(tf.argmax(yhat, axis=1))
            i = i + 1

            # Backward propagation
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
            # updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

        train_accuracies = []
        for predict in predicts:
            with tf.name_scope('calculo_da_acuracia') as scope:
                # train_accuracy = tf.reduce_mean(np.argmax(train_y, axis=1) == predict)
                label_train = tf.argmax(
                    train_y, axis=1, name="label_train_argmax")
                train_accuracy = tf.metrics.accuracy(
                    labels=label_train, predictions=predict)
                train_accuracies.append(train_accuracy[1])

                # test_accuracy = tf.reduce_mean(np.argmax(test_y, axis=1) == predict)
                label_test = tf.argmax(
                    test_y, axis=1, name="label_test_argmax")
                test_accuracy = tf.metrics.accuracy(
                    labels=label_test, predictions=predict)

        # Run SGD
        sess = tf.Session()

        writer = tf.summary.FileWriter("./log/", sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_accuracies_sess = sess.run(train_accuracies, feed_dict={X: train_X},
                                         options=run_options, run_metadata=run_metadata)
        writer.close()
        sess.close()

        return train_accuracies_sess


def propagation(neural_network, X):
    # with tf.name_scope('rede_neural_' + str(i)) as scope:
    w = neural_network

    print(tf.shape(w[0]));
    print(tf.shape(w[1]));
    # Forward propagation
    yhat = forwardprop(X, w[0], w[1])
    for i in range(len(w) - 2):
        yhat = forwardprop_hidden(yhat, w[i + 2])

    #predicts.append(tf.argmax(yhat, axis=1))
    print(tf.argmax,axis=1);
    return tf.argmax(yhat, axis=1)


def nn_example_with_map_fn(neural_networks):
    print("Rodando rede neural")

    # iris
    # train_X, test_X, train_y, test_y = get_iris_data()

    # mnist
    with tf.name_scope('Fitness') as scope:

        train_X, test_X, train_y, test_y = get_mnist_data()
        # Defining number of layers
        number_neural_networks = len(neural_networks)
        number_neural_networks_remaining = number_neural_networks
        print("numero de redes: %d" % (number_neural_networks))

        # Layer's sizes
        # Number of input nodes: x features and 1 bias
        x_size = train_X.shape[1]
        y_size = train_y.shape[1]  # Number of outcomes (3 iris flowers)
        # Symbols
        X = tf.placeholder("float", shape=[None, x_size], name="X")
        y = tf.placeholder("float", shape=[None, y_size], name="Y")
        # Weight initializations
        i = 0
        predicts = []

        predicts = tf.map_fn(lambda x: propagation(x, X), neural_networks)
        print(predicts)

        # Backward propagation
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
        # updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

        train_accuracies = []
        for predict in predicts:
            with tf.name_scope('calculo_da_acuracia') as scope:
                # train_accuracy = tf.reduce_mean(np.argmax(train_y, axis=1) == predict)
                label_train = tf.argmax(
                    train_y, axis=1, name="label_train_argmax")
                train_accuracy = tf.metrics.accuracy(
                    labels=label_train, predictions=predict)
                train_accuracies.append(train_accuracy[1])

                # test_accuracy = tf.reduce_mean(np.argmax(test_y, axis=1) == predict)
                label_test = tf.argmax(
                    test_y, axis=1, name="label_test_argmax")
                test_accuracy = tf.metrics.accuracy(
                    labels=label_test, predictions=predict)

        # Run SGD
        sess = tf.Session()

        writer = tf.summary.FileWriter("~/log/", sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_accuracies_sess = sess.run(train_accuracies, feed_dict={X: train_X},
                                         options=run_options, run_metadata=run_metadata)
        writer.close()
        sess.close()

        return train_accuracies_sess
