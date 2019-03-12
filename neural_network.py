import tensorflow as tf
import numpy as np
import time
from utils import variable_summaries

    
def block_diagonal(matrices, dtype=tf.float32):
    """Constructs block-diagonal matrices from a list of batched 2D tensors.

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
    all_X = np.ones((N, M))
    all_X[:, :] = train_X

    num_labels = len(np.unique(train_y))
    train_y_eye = np.eye(num_labels)[train_y]  # One liner trick!
    test_y_eye = np.eye(num_labels)[test_y]  # One liner trick!
    # a,b,c,d = train_test_split(all_X, all_Y, test_size=0.00, random_state=0)
    #return (all_X, all_X, all_Y, all_Y)
    return train_X,train_y_eye,test_X,test_y_eye

def nn_example_without_struct(neural_networks):
    #print("Rodando rede neural")

    # iris
    # train_X, test_X, train_y, test_y = get_iris_data()

    # mnist
    with tf.name_scope('Fitness') as scope:

        train_X, test_X, train_y, test_y = get_mnist_data()
        # Defining number of layers
        number_neural_networks = len(neural_networks)
        number_neural_networks_remaining = number_neural_networks
        #print("numero de redes: %d" % (number_neural_networks))

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


class Neural_network:

    def __init__(self,populationSize,layers,convulations,biases,logdir):
        #self.neural_networks = neural_networks
        self.layers = layers
        self.logdir = logdir
        #self.train_x, self.test_x, self.train_y,self.test_y = get_mnist_data()
        self.train_x, self.train_y, self.test_x, self.test_y = get_mnist_data()
        print(self.train_y)
        self.train_x = self.train_x.reshape(-1, 28, 28, 1)
        self.test_x =  self.test_x.reshape(-1, 28, 28, 1)
        self.convulations = convulations
        self.biases = biases
        self.populationSize = populationSize

    
    def conv2d(self,x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x) 

    def maxpool2d(self,x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

    def conv_net(self):  

        weights = self.convulations
        biases = self.biases
        # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
        conv1 = tf.map_fn( lambda x: self.conv2d(self.X, weights['wc1'][x], biases['bc1'][x]) , tf.range(self.populationSize) , dtype=tf.float32 )
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
        conv1 = tf.map_fn( lambda x: self.maxpool2d(conv1[x], k=2) ,  tf.range(self.populationSize) , dtype=tf.float32 )

        # # Convolution Layer
        # # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
        # conv2 = tf.map_fn( lambda x: self.conv2d(conv1[x], weights['wc2'][x], biases['bc2'][x]) , tf.range(self.populationSize) , dtype=tf.float32 )
        # # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
        # conv2 = tf.map_fn( lambda x: self.maxpool2d(x, k=2) , conv2 , dtype=tf.float32 )

        # conv3 = tf.map_fn( lambda x: self.conv2d(conv2[x], weights['wc3'][x], biases['bc3'][x]), tf.range(self.populationSize) , dtype=tf.float32 )
        # # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
        # conv3 =  tf.map_fn( lambda x: self.maxpool2d(x, k=2) , conv3 , dtype=tf.float32 )

        convs = []
        convs.append(conv1)
        for i in range(len(weights.keys()) - 3):
            conv = tf.map_fn( lambda x: self.conv2d(convs[i][x], weights['wc' + str(i+2)][x], biases['bc' + str(i+2)][x]) , tf.range(self.populationSize) , dtype=tf.float32 )
            conv = tf.map_fn( lambda x: self.maxpool2d(x, k=2) , conv , dtype=tf.float32 )
            convs.append(conv[:])
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        last_conv = convs.pop()
        fc1 = tf.map_fn( lambda x: tf.reshape(last_conv[x], [-1, weights['wd1'][x].get_shape().as_list()[0]]), tf.range(self.populationSize) , dtype=tf.float32 )
        fc1 = tf.map_fn(lambda x:tf.add(tf.matmul(fc1[x], weights['wd1'][x]), biases['bd1'][x]),  tf.range(self.populationSize) , dtype=tf.float32 )
        fc1 = tf.map_fn(lambda x: tf.nn.relu(fc1[x]) , tf.range(self.populationSize)  , dtype=tf.float32 )
        
        # Output, class prediction
        # finally we multiply the fully connected layer with the weights and add a bias term. 
        out = tf.map_fn(lambda x: tf.add(tf.matmul(fc1[x], weights['out'][x]), biases['out'][x]),tf.range(self.populationSize) , dtype=tf.float32 )
        return out

    def conv_net_best(self):  

        weights = self.best_conv
        biases = self.best_bias
        # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
        conv1 = self.conv2d(self.X, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
        conv1 = self.maxpool2d(conv1, k=2)

        # Convolution Layer
        # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
        conv2 = self.maxpool2d(conv2, k=2)

        conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'])
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
        conv3 = self.maxpool2d(conv3, k=2)


        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Output, class prediction
        # finally we multiply the fully connected layer with the weights and add a bias term. 
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out
    def forwardprop(self,X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """

        with tf.name_scope('foward_propagation') as scope:
            h = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
            yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat


    def forwardprop_hidden(self,w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        with tf.name_scope('foward_propagation_hidden') as scope:
            h = tf.nn.sigmoid(w_1)  # The \sigma function
            yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    def get_accuracies(self,predict):

            # arg = tf.cast(tf.argmax(
            #     self.Y, axis=1, name="label_test_argmax_sme"),tf.float32)
            # arg2 = tf.cast(tf.argmax(
            #     predict, axis=1, name="label_test_argmax_sme"),tf.float32)
            # correct_prediction = tf.equal(arg, tf.cast(arg2,tf.float32))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                #Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
            correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(self.Y, 1))

            #calculate accuracy across all the given images and average them out. 
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return accuracy

    def get_square_mean_error(self,predict):
        
        with tf.name_scope('calculo_square_mean_error') as scope:

            label_test = tf.cast(tf.argmax(
                self.Y, axis=1, name="label_test_argmax_sme"),tf.float32)
            square_mean_error = tf.metrics.mean_squared_error(labels=label_test,predictions=predict)
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=self.Y))

        return square_mean_error[0]
        #return cost

    def get_predicts(self,neural_network, X,layers):

        with tf.name_scope('rede_neural_') as scope:

            # Forward propagation
            yhat = forwardprop(X, tf.slice(neural_network[0],[0,0],[layers[0],layers[1]]), tf.slice(neural_network[1],[0,0],[layers[1],layers[2]]))
            for i in range(neural_network.shape[0] - 2):
                yhat = forwardprop_hidden(yhat, tf.slice(neural_network[i+2],[0,0],[layers[i+1],layers[i+2]]))

            return tf.cast(tf.argmax(yhat, axis=1),tf.float32)

    def convulation(self,input_data,weights,bias,pool_shape):
        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

        # add the bias
        out_layer += bias

        # apply a ReLU non-linear activation
        out_layer = tf.nn.relu(out_layer)

        # now perform max pooling
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        strides = [1, 2, 2, 1]
        out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, 
                                padding='SAME')

        return out_layer

    def get_cost_functions(self,predict,train,test):
        with tf.name_scope('calculo_da_acuracia') as scope:
            # train_accuracy = tf.reduce_mean(np.argmax(train_y, axis=1) == predict)
            
            #label_train = tf.argmax(
            #    train, axis=1, name="label_train_argmax")
            #train_cost = -tf.reduce_sum( label_train * tf.log(predict) )  
            
            #tf.metrics.accuracy(
                #labels=label_train, predictions=predict)

            # test_accuracy = tf.reduce_mean(np.argmax(test_y, axis=1) == predict)
            #label_test = tf.argmax(
            #    test, axis=1, name="label_test_argmax")
            #test_cost = -tf.reduce_sum( test * tf.log(predict) )  

            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train, logits=predict))

            #return test_cost

    def run(self):
        #print("Rodando rede neural")

        # iris
        # train_X, test_X, train_y, test_y = get_iris_data()

        # mnist
        with tf.name_scope('Fitness') as scope:

            #train_X, test_X, train_y, test_y = get_mnist_data()
            # Defining number of layers
            #print(self.neural_networks.shape)

            #number_neural_networks = self.neural_networks.shape[0]
            #number_neural_networks_remaining = number_neural_networks
            #print("numero de redes: %d" % (number_neural_networks))

            # Layer's sizes
            # Number of input nodes: x features and 1 bias
            #x_size = 786 #self.train_x.shape[1]
            print(self.train_y.shape)
            y_size = self.train_y.shape[1]  
            #print(y_size)
            
            # Symbols
            self.X = tf.placeholder("float", shape=[None, 28, 28, 1], name="X")
            self.Y = tf.placeholder("float", shape=[None,y_size], name="Y")
            
            X = self.X
            Y = self.Y

            i = 0

            with tf.name_scope('predicts') as scope:


                predicts = self.conv_net()
                # predicts = tf.map_fn(lambda x: self.get_predicts(x,X,self.layers), self.neural_networks)
                

                # with tf.name_scope('convulation_') as scope:
                #     X_shaped = tf.reshape(X[:,1:], [28, 28, 1 , 1])
                #     yhat = tf.map_fn(lambda x: self.convulation(X_shaped,x,self.biases,[28,28]) ,self.convulations)
                
                ## -- ESTA PARTE FUNCIONA --
                # with tf.name_scope('rede_neural_') as scope:

                #     # Forward propagation
                #     yhat = tf.map_fn(lambda x: forwardprop(X, tf.slice(x[0],[0,0],[self.layers[0],self.layers[1]]), tf.slice(x[1],[0,0],[self.layers[1],self.layers[2]])), self.neural_networks )
                #     for i in range(self.neural_networks[0].shape[0] - 2):
                #         yhat = tf.map_fn(lambda x: forwardprop_hidden(x, tf.slice(self.neural_networks[i+2],[0,0],[self.layers[i+1],self.layers[i+2]])), yhat)
                #     predicts = tf.map_fn(lambda x: tf.cast(tf.argmax(x, axis=1),tf.float32), yhat )
                ## -- ESTA PARTE FUNCIONA --

                
                # #print(predicts)
            
            with tf.name_scope('accuracies') as scope:

                train_accuracies = tf.map_fn(lambda x: self.get_accuracies(x),predicts)

                #train_accuracies = self.get_accuracies(predicts[0])
            with tf.name_scope('cost') as cost: 
                
                
                cost = tf.map_fn(lambda pred: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=self.Y)),predicts)
                #cost = tf.map_fn(lambda pred: -tf.reduce_sum(self.Y * tf.log(pred)), predicts)

            with tf.name_scope('square_mean_error') as scope:

                square_mean_error = tf.map_fn(lambda pred: tf.reduce_mean(tf.squared_difference(pred, Y)),predicts)
            with tf.name_scope('root_square_mean_error') as scope:

                root_square_mean_error = tf.map_fn(lambda pred: tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, pred)))), predicts)
            #with tf.name_scope('square_mean_errors') as scope:

           #     square_mean_errors = tf.map_fn(lambda x: self.get_square_mean_error(x),predicts)
            
            #with tf.name_scope('cost_function') as scope:
                
            #    cost_function = tf.map_fn(lambda x: self.get_cost_functions(x,train_y,test_y),predicts)

            # Run SGD

            #predict_sess = sess.run(predicts, feed_dict={X: train_X,Y: train_y},
            #                                options=run_options, run_metadata=run_metadata)

            #print('predict_sess temos:')
            #print(predict_sess)                     
            


            ## Aqui temos a sessão para a retirada das acurácias e afins
            # sess = tf.Session()

            # writer = tf.summary.FileWriter(self.logdir, sess.graph)
            # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # run_metadata = tf.RunMetadata()


            # start = time.time()
            # sess.run(tf.global_variables_initializer())
            # print("Global variables:", time.time() - start)

            # start = time.time()
            # sess.run(tf.local_variables_initializer())
            # print("local variables:", time.time() - start)

            # start = time.time()
            # predicts_session,train_accuracies_session = sess.run([predicts,train_accuracies], feed_dict={X: self.train_x, Y: self.train_y},
            #                                 options=run_options, run_metadata=run_metadata)
            # print("predicts e acuracias:", time.time() - start )

            # print('acuracia temos:')
            # print(train_accuracies_session)

            #cost_sess = sess.run(cost_function, feed_dict={X: train_X},
            #                                options=run_options, run_metadata=run_metadata)

            ## Fim da retirada das acurácias e afins

            ## Utilizacao das acuracias e predicts como tensores
            self.predicts = predicts
            self.accuracies = train_accuracies
            self.cost = cost
            self.square_mean_error = square_mean_error
            self.root_square_mean_error = root_square_mean_error
            self.label_argmax = tf.cast(tf.argmax(
                self.Y, axis=1, name="label_test_argmax_sme"),tf.float32)

            tf.summary.scalar('acuracia',tf.reduce_max(self.accuracies))
            #variable_summaries(self.accuracies)
            #tf.summary.scalar('predicts', self.predicts)

            # writer.close()
            # sess.close()

            # self.predicts = predicts_session
            # self.accuracies = train_accuracies_session


            # return train_accuracies_session
            return self.accuracies


    def run_best(self):

        with tf.name_scope('Fitness') as scope:
            #train_X, test_X, train_y, test_y = get_mnist_data()
            # Defining number of layers
            #print(self.neural_networks.shape)

            #number_neural_networks = self.neural_networks.shape[0]
            #number_neural_networks_remaining = number_neural_networks
            #print("numero de redes: %d" % (number_neural_networks))

            # Layer's sizes
            # Number of input nodes: x features and 1 bias
            #x_size = 786 #self.train_x.shape[1]
            print(self.train_y.shape)
            y_size = self.train_y.shape[1]  
            #print(y_size)
            
            # Symbols
            #self.X = tf.placeholder("float", shape=[None, 28, 28, 1], name="X")
            #self.Y = tf.placeholder("float", shape=[None,y_size], name="Y")
            
            X = self.X
            Y = self.Y

            i = 0

            with tf.name_scope('predicts') as scope:


                predicts = self.conv_net_best()

            with tf.name_scope('accuracies') as scope:

                train_accuracies = tf.map_fn(lambda x: self.get_accuracies(x),predicts)
            
            # self.predicts = predicts
            # self.accuracies = train_accuracies

            # self.label_argmax = tf.cast(tf.argmax(
            #     self.Y, axis=1, name="label_test_argmax_sme"),tf.float32)

        
        return train_accuracies
def calculate_fitness(neural_networks,layers,logdir):
    #return nn_cube(neural_networks,layers)
    neural_structure = neural_network(neural_networks,layers,logdir)
    return neural_structure.run()
    

