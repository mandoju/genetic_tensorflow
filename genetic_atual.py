import tensorflow as tf
import numpy as np
import time
from itertools import filterfalse
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.python import debug as tf_debug
from random import randint
import random
import copy


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

def get_iris_data():
    """ Read the iris data set and split them into training and test sets """
    iris = datasets.load_iris()
    data = iris["data"]
    target = iris["target"]

    # Prepend the column of 1s for bias
    N, M = data.shape
    all_X = np.ones((N, M + 1 ))
    all_X[:, 1:] = data

    # Convert into one-hot vectors
    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target]  # One liner trick!
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=1)

def get_mnist_data():
    mnist = tf.keras.datasets.mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data();
    train_X = np.array(train_X).reshape(len(train_X),784)
    # Prepend the column of 1s for bias
    N, M = train_X.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = train_X

    num_labels = len(np.unique(train_y))
    all_Y = np.eye(num_labels)[train_y]  # One liner trick!
    a,b,c,d = train_test_split(all_X, all_Y, test_size=0.00, random_state=1)
    return (all_X,all_X,all_Y,all_Y)



def nn_example_without_struct(neural_networks):
    print("sem estrutura")


    #iris
    #train_X, test_X, train_y, test_y = get_iris_data()

    #mnist
    train_X, test_X, train_y, test_y = get_mnist_data()

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
    x_size = train_X.shape[1]  # Number of input nodes: x features and 1 bias
    print(x_size)
    y_size = train_y.shape[1]  # Number of outcomes (3 iris flowers)
    print(y_size)
    # Symbols
    X = tf.placeholder("float", shape=[None, x_size], name="X")
    y = tf.placeholder("float", shape=[None, y_size], name="Y")
    # Weight initializations
    i = 0
    predicts = []
    for neural_network in neural_networks:
        with tf.name_scope('rede_neural_'+str(i)) as scope:
            w = neural_network

            # Forward propagation
            yhat = forwardprop(X, w[0], w[1])
            for i in range(len(w) - 2):
                yhat = forwardprop_hidden(yhat, w[i + 2])
                print(tf.shape(yhat))
            final_yhat = tf.cast(tf.argmax(yhat, axis=1), tf.float32)

            predicts.append(tf.argmax(yhat, axis=1))
        i = i + 1

        # Backward propagation
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
        #updates = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

    train_accuracies = []
    for predict in predicts:
        with tf.name_scope('calculo_da_acuracia') as scope:
            #train_accuracy = tf.reduce_mean(np.argmax(train_y, axis=1) == predict)
            label_train = tf.argmax(train_y, axis=1, name="label_train_argmax")
            train_accuracy = tf.metrics.accuracy(labels = label_train , predictions= predict)
            train_accuracies.append(train_accuracy[1]);

            #test_accuracy = tf.reduce_mean(np.argmax(test_y, axis=1) == predict)
            label_test = tf.argmax(test_y, axis=1, name="label_test_argmax")
            test_accuracy = tf.metrics.accuracy(labels = label_test, predictions= predict)


    # Run SGD
    sess = tf.Session()
    writer = tf.summary.FileWriter("/home/jorge/graph/log", sess.graph)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    train_accuracies_sess = sess.run(train_accuracies, feed_dict={X: train_X},
                                options=run_options, run_metadata=run_metadata)
    writer.close()

    #test_accuracies = sess.run(test_accuracy, feed_dict={X: test_X},
    #                           options=run_options, run_metadata=run_metadata)
    #predict_sess = sess.run(predicts, feed_dict={X: test_X},
    #                           options=run_options, run_metadata=run_metadata)

    #label_train_sess = sess.run(label_train, feed_dict={X: test_X},
    #                           options=run_options, run_metadata=run_metadata)

    #label_test_sess = sess.run(label_test, feed_dict={X: test_X},
    #                           options=run_options, run_metadata=run_metadata)


    #updates_sess = sess.run(updates,feed_dict={X: train_X},options=run_options, run_metadata=run_metadata)

    #print(predict_sess)
    #print(predict_sess)
    #print(updates_sess)
    #print(label_train_sess)
    #print(label_test_sess)


    #print(train_accuracies)
    #for i in range(number_neural_networks):

    #    print(train_accuracies)
    #    print(test_accuracies)

        #train_accuracy = np.mean(np.argmax(train_y, axis=1) == train_accuracies[i])

        #test_accuracy = np.mean(np.argmax(test_y, axis=1) == test_accuracies[i])
        #print("train accuracy = %.2f%%, test accuracy = %.2f%%"
        #      % (100. * train_accuracy, 100. * test_accuracy))

        #print(len(train_accuracies[i]))

        #print(len(final_yhat))

    sess.close()
    #print("train_accuracies");
    #print(train_accuracies);
    return train_accuracies_sess;


def create_population(populationSize):

    population = []
    for i in range(populationSize):
        #w_1 = np.random.rand(5, 10).astype('f') * 0.01;
        #w_2 = np.random.rand(10, 10000).astype('f') * 0.01;
        #w_3 = np.random.rand(10000, 500).astype('f') * 0.01;
        #w_4 = np.random.rand(500, 3).astype('f') * 0.01;

        w_1 = tf.Variable(tf.random_normal((785,1000), stddev=0.1));
        #w_2 = tf.random_normal((10, 10000), stddev=0.1);
        #w_3 = tf.random_normal((10000, 500), stddev=0.1);
        w_4 = tf.Variable(tf.random_normal((1000, 9), stddev=0.1));
        population.append([w_1,w_4]);
    return population


def calculate_fitness(neural_networks):
    return nn_example_without_struct(neural_networks);

def choose_best(neural_networks,fitnesses):
    print(fitness)
    top_2_idx = np.argsort(fitnesses)[-2:]
    top_2_values = [neural_networks[i] for i in top_2_idx]
    print([fitnesses[i] for i in top_2_idx])
    return top_2_values;


def crossover(neural_networks,population_size):

    new_population = neural_networks[:] ;
    size_neural_networks = len(neural_networks)
    crossover_point = np.random.choice(np.arange(1, 784, step=1), population_size - size_neural_networks)

    for i in range(population_size - size_neural_networks):
        start_op = tf.global_variables_initializer()
        #comparison = tf.equal(temp_neural_network , tf.constant(1))
        #conditional_assignment_op = temp_neural_network.assign(tf.where(comparison, tf.zeros_like(temp_neural_network), temp_neural_network))
        father = neural_networks[0];
        mother = neural_networks[1];

        temp_neural_network = [];
        #for i in range(len(temp_neural_network)):
        # Gather parents by shuffled indices, expand back out to pop_size too
        for weight_idx in range(len(mother)):

            rows_size = mother[weight_idx].get_shape()[0];
            columns_size =  mother[weight_idx].get_shape()[1];
            weight = tf.Variable(tf.zeros([rows_size,columns_size],tf.float32));
            weight.assign(mother[weight_idx]);

            print('peso ' + str(weight_idx));
            print('rows_size ' + str(rows_size));



            for row_idx in range(rows_size - 1):
                if(random.uniform(0,1) < 0.5):
                    weight[row_idx].assign(father[weight_idx][row_idx]);
                else:
                    weight[row_idx].assign(mother[weight_idx][row_idx]);

                #column_to_cut_start = randint(0,columns_size - 1)
                #column_to_cut_end = randint(column_to_cut_start,columns_size - 1);
                #cut_size = column_to_cut_end - column_to_cut_start;
                #cut_indexes = range(column_to_cut_start,column_to_cut_end);


                #print(temp_neural_network[weight_idx][row_idx][column_to_cut_start:column_to_cut_end]);
                #print(father[weight_idx][row_idx][column_to_cut_start:column_to_cut_end]);
                #temp_neural_network[weight_idx][row_idx][column_to_cut_start:column_to_cut_end] = father[weight_idx][row_idx][column_to_cut_start:column_to_cut_end];
                #print(temp_neural_network[weight_idx]);
                #tf.concat([temp_neural_network[weight_idx][row_idx][:column_to_cut_start], father[weight_idx][row_idx][column_to_cut_start:column_to_cut_end], temp_neural_network[weight_idx][row_idx][column_to_cut_end:]], axis=0);
                #print(new_row);
                #temp_neural_network[weight_idx][row_idx] = temp_neural_network[weight_idx][row_idx][column_to_cut_start:column_to_cut_end].assign(father[weight_idx][row_idx][column_to_cut_start:column_to_cut_end]);
                #tf.scatter_update(temp_neural_network[weight_idx][row_idx], cut_indexes , father[weight_idx][row_idx][column_to_cut_start:column_to_cut_end])
                #tf.scatter_update(temp_neural_network[weight_idx], 0 , new_row)
            temp_neural_network.append(weight)
        new_population.append(temp_neural_network[:]);

        sess = tf.Session()
        writer = tf.summary.FileWriter("/home/jorge/graph/log", sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.initialize_all_variables())
        sess.run(new_population);
    return new_population;

def mutation(neural_networks,population_size,mutationRate):

    #depois fazer matrix mascara (a.k.a recomendação do gabriel)

    size_neural_networks = len(neural_networks)
    new_neural_networks = []
    for i in range(population_size - size_neural_networks):
        temp_neural_network = neural_networks[i%size_neural_networks];
        start_op = tf.global_variables_initializer()
        comparison = tf.equal(temp_neural_network , tf.constant(1))
        conditional_assignment_op = temp_neural_network.assign(tf.where(comparison, tf.zeros_like(temp_neural_network), temp_neural_network))

        for weight in temp_neural_network:
            for x in weight:
                for y in x:
                    if(np.random < mutationRate):
                        y = np.random;
        new_neural_networks.append(temp_neural_network);

    for neural_network in neural_networks:
        new_neural_networks.append(neural_network);

    return new_neural_networks;

if __name__ == "__main__":
    genetic_pool_settings = {
        'populationSize': 30,
        'tournamentSize': 4,
        'memberDimensions': [4, 3, 2, 3, 4],
        'mutationRate': 0.05,
        'averagesCount': 1,
        'maxEpochs': 10
    };

    population = create_population(3);
    print(population);
    fitness = calculate_fitness(population);
    best_ones = choose_best(population,fitness);
    population = crossover(best_ones,3);
    print(population);
    fitness = calculate_fitness(population);
    best_ones = choose_best(population, fitness);
    population = crossover(best_ones, 3); print(population);
    fitness = calculate_fitness(population);
    best_ones = choose_best(population,fitness);
    population = crossover(best_ones,3); print(population);
    fitness = calculate_fitness(population);
    best_ones = choose_best(population,fitness);
    population = crossover(best_ones,3);