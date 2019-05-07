import tensorflow as tf
import numpy as np
import time
from utils import variable_summaries
from layer import Layer


class Neural_network:

    def __init__(self, populationSize, layers, convulations, biases, train_x, train_y,test_x,test_y, logdir):
        #self.neural_networks = neural_networks
        self.layers = layers
        self.logdir = logdir
        #self.train_x, self.test_x, self.train_y,self.test_y = get_mnist_data()
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.convulations = convulations
        self.biases = biases
        self.populationSize = populationSize
        self.classification = False

    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(
            x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    """ def conv_net(self):

        weights = self.convulations
        biases = self.biases
        # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.


        # conv1 = tf.map_fn(lambda x: self.conv2d(self.X, weights['wc1'][x], biases['bc1'][x]), tf.range(
        #     self.populationSize), dtype=tf.float32)
        # # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
        # conv1 = tf.map_fn(lambda x: self.maxpool2d(
        #     conv1[x], k=2),  tf.range(self.populationSize), dtype=tf.float32)

        conv1 = []
        for i in range(self.populationSize):
            conv1.append( self.conv2d(self.X, weights['wc1'][i], biases['bc1'][i]) )
        max_pool_1 = []
        for conv in conv1:
            max_pool_1.append( self.maxpool2d(conv, k=2) )

        convs = []
        #convs = tf.stack([convs])
        #convs.append(max_pool_1)
        last_conv  = max_pool_1
        for i in range(len(weights.keys()) - 3):
            #print(last_conv[0])
            #print(weights['wc' + str(i+2)][tf.constant(0)])
            #print(biases['bc' + str(i+2)][0])
            print(last_conv)
            convs = []
            for j in range(self.populationSize): 
                convs.append(self.conv2d(last_conv[j], weights['wc' + str(i+2)][j], biases['bc' + str(i+2)][j]))
            # conv = tf.map_fn(lambda x: self.conv2d(last_conv[x], weights['wc' + str(
            #     i+2)][x], biases['bc' + str(i+2)][x]), tf.range(self.populationSize), dtype=tf.float32)
            max_pool = []
            for j in range(self.populationSize): 
                max_pool.append(self.maxpool2d(convs[j],k=2))
            last_conv = max_pool
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        #last_conv = convs.pop()
        
        fc1 = []
        reshaped_conv = []
        for i in range(self.populationSize):
            reshaped_conv.append(tf.reshape(last_conv[i], [-1, weights['wd1'][i].get_shape().as_list()[0]]))
        
        fc1 = []
        for x in range(self.populationSize):
            fc1.append(tf.reshape(last_conv[x], [-1, weights['wd1'][x].get_shape().as_list()[0]]))
        fc2 = []
        for x in range(self.populationSize):
            fc2.append(tf.add(tf.matmul(fc1[x], weights['wd1'][x]), biases['bd1'][x]))

        fc3 = []
        for x in range(self.populationSize):
            fc3.append(tf.nn.relu(fc2[x]))
        out = []
        for x in range(self.populationSize):
            out.append(tf.add(tf.matmul(fc3[x], weights['out'][x]), biases['out'][x]))
        # fc1 = tf.map_fn(lambda x: tf.reshape(last_conv[x], [-1, weights['wd1'][x].get_shape(
        # ).as_list()[0]]), tf.range(self.populationSize), dtype=tf.float32)
        # fc1 = tf.map_fn(lambda x: tf.add(tf.matmul(fc1[x], weights['wd1'][x]), biases['bd1'][x]),  tf.range(
        #     self.populationSize), dtype=tf.float32)
        # fc1 = tf.map_fn(lambda x: tf.nn.relu(fc1[x]), tf.range(
        #     self.populationSize), dtype=tf.float32)

        # Output, class prediction
        # finally we multiply the fully connected layer with the weights and add a bias term.
        # out = tf.map_fn(lambda x: tf.add(tf.matmul(fc1[x], weights['out'][x]), biases['out'][x]), tf.range(
        #     self.populationSize), dtype=tf.float32)
        return tf.stack(out) """

    def conv_net(self):
        
        weights = self.convulations
        biases = self.biases
        
        layer_1 = Layer(self.populationSize,weights['wd1'], biases['bc1'],'wd1','relu')
        layer_2 = Layer(self.populationSize,weights['wd2'], biases['bc2'],'wd2','relu')
        layer_3 = Layer(self.populationSize,weights['out'], biases['out'],'wdout')

        layer_1_out = layer_1.run(self.X)
        layer_2_out = layer_2.run(layer_1_out)
        layer_3_out = layer_3.run(layer_2_out)

        return tf.stack(layer_3_out)
    def conv_net_best(self):

        weights = self.convulations
        biases = self.biases
        # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
        conv1 = tf.map_fn(lambda x: self.conv2d(self.X, weights['wc1'][x], biases['bc1'][x]), tf.range(1), dtype=tf.float32)
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
        conv1 = tf.map_fn(lambda x: self.maxpool2d(
            conv1[x], k=2),  tf.range(1), dtype=tf.float32)

        convs = []
        convs.append(conv1)
        for i in range(len(weights.keys()) - 3):
            conv = tf.map_fn(lambda x: self.conv2d(convs[i][x], weights['wc' + str(
                i+2)][x], biases['bc' + str(i+2)][x]), tf.range(1), dtype=tf.float32)
            conv = tf.map_fn(lambda x: self.maxpool2d(
                x, k=2), conv, dtype=tf.float32)
            convs.append(conv[:])
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        last_conv = convs.pop()
        fc1 = tf.map_fn(lambda x: tf.reshape(last_conv[x], [-1, weights['wd1'][x].get_shape(
        ).as_list()[0]]), tf.range(1), dtype=tf.float32)
        fc1 = tf.map_fn(lambda x: tf.add(tf.matmul(fc1[x], weights['wd1'][x]), biases['bd1'][x]),  tf.range(1) , dtype=tf.float32)
        fc1 = tf.map_fn(lambda x: tf.nn.relu(fc1[x]), tf.range(1), dtype=tf.float32)

        # Output, class prediction
        # finally we multiply the fully connected layer with the weights and add a bias term.
        out = tf.map_fn(lambda x: tf.add(tf.matmul(fc1[x], weights['out'][x]), biases['out'][x]), tf.range(1), dtype=tf.float32)
        return out


    def forwardprop(self, X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """

        with tf.name_scope('foward_propagation') as scope:
            h = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
            yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    def forwardprop_hidden(self, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        with tf.name_scope('foward_propagation_hidden') as scope:
            h = tf.nn.sigmoid(w_1)  # The \sigma function
            yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    def get_accuracies(self, predict):

            # arg = tf.cast(tf.argmax(
            #     self.Y, axis=1, name="label_test_argmax_sme"),tf.float32)
            # arg2 = tf.cast(tf.argmax(
            #     predict, axis=1, name="label_test_argmax_sme"),tf.float32)
            # correct_prediction = tf.equal(arg, tf.cast(arg2,tf.float32))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                # Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image. and both will be a column vector.
        if(self.classification):
            correct_prediction = tf.equal(
                tf.argmax(predict, 1), tf.argmax(self.Y, 1))

            # calculate accuracy across all the given images and average them out.
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return accuracy
        else:
            correct_prediction = tf.equal(
                predict, self.Y)

            # calculate accuracy across all the given images and average them out.
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return accuracy
    def get_square_mean_error(self, predict):

        with tf.name_scope('calculo_square_mean_error') as scope:

            label_test = tf.cast(tf.argmax(
                self.Y, axis=1, name="label_test_argmax_sme"), tf.float32)
            square_mean_error = tf.metrics.mean_squared_error(
                labels=label_test, predictions=predict)
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=self.Y))

        return square_mean_error[0]
        # return cost

    def get_predicts(self, neural_network, X, layers):

        with tf.name_scope('rede_neural_') as scope:

            # Forward propagation
            yhat = forwardprop(X, tf.slice(neural_network[0], [0, 0], [
                               layers[0], layers[1]]), tf.slice(neural_network[1], [0, 0], [layers[1], layers[2]]))
            for i in range(neural_network.shape[0] - 2):
                yhat = forwardprop_hidden(yhat, tf.slice(
                    neural_network[i+2], [0, 0], [layers[i+1], layers[i+2]]))

            return tf.cast(tf.argmax(yhat, axis=1), tf.float32)

    def convulation(self, input_data, weights, bias, pool_shape):
        # setup the convolutional layer operation
        out_layer = tf.nn.conv2d(input_data, weights, [
                                 1, 1, 1, 1], padding='SAME')

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

    def get_cost_functions(self, predict, train, test):
        with tf.name_scope('calculo_da_acuracia') as scope:
            # train_accuracy = tf.reduce_mean(np.argmax(train_y, axis=1) == predict)

            # label_train = tf.argmax(
            #    train, axis=1, name="label_train_argmax")
            #train_cost = -tf.reduce_sum( label_train * tf.log(predict) )

            # tf.metrics.accuracy(
                # labels=label_train, predictions=predict)

            # test_accuracy = tf.reduce_mean(np.argmax(test_y, axis=1) == predict)
            # label_test = tf.argmax(
            #    test, axis=1, name="label_test_argmax")
            #test_cost = -tf.reduce_sum( test * tf.log(predict) )

            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=train, logits=predict))

            # return test_cost

    def run(self):
        #print("Rodando rede neural")

        # mnist
        with tf.name_scope('Fitness') as scope:

            print(self.train_y.shape)
            y_size = self.train_y.shape[1]

            #self.X = tf.placeholder("float", shape=[None, 28, 28, 1], name="X")
            #self.Y = tf.placeholder("float", shape=[None, y_size], name="Y")

            self.X = tf.placeholder("float", shape=[None, 1], name="X")
            self.Y = tf.placeholder("float", shape=[None, y_size], name="Y")

            X = self.X
            Y = self.Y

            i = 0

            with tf.name_scope('predicts') as scope:

                predicts = self.conv_net()
                print(predicts)

            with tf.name_scope('accuracies') as scope:

                train_accuracies = tf.map_fn(
                    lambda x: self.get_accuracies(x), predicts)
                
                
                #train_accuracies = self.get_accuracies(predicts[0])
            with tf.name_scope('cost') as cost:

                cost = tf.map_fn(lambda pred: tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=self.Y), predicts)
                print(cost)
                #cost = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicts, labels=self.Y, axis=1)
                if(self.classification):
                    cost = tf.reduce_mean(cost,1)

                #cost = tf.map_fn(lambda pred: -tf.reduce_sum(self.Y * tf.log(pred)), predicts)

            with tf.name_scope('square_mean_error') as scope:

                square_mean_error = tf.map_fn(lambda pred: tf.reduce_mean(tf.squared_difference(tf.cast(tf.argmax(
                pred, axis=1, name="label_test_argmax_sme"),tf.float32), tf.cast(tf.argmax(self.Y, axis=1, name="label_test_argmax_sme"), tf.float32))), predicts)
            with tf.name_scope('root_square_mean_error') as scope:

                root_square_mean_error = tf.map_fn(lambda pred: tf.sqrt(
                    tf.reduce_mean(tf.square(tf.subtract(Y, pred)))), predicts, dtype=tf.float32)
           
            # Utilizacao das acuracias e predicts como tensores
            self.predicts = predicts
            if(self.classification):
                self.argmax_predicts = tf.argmax(predicts[0], 1)
            else:
                self.argmax_predicts = predicts[0]
            self.accuracies = train_accuracies
            self.cost = cost
            self.square_mean_error = square_mean_error
            self.root_square_mean_error = root_square_mean_error
            self.label_argmax = tf.cast(tf.argmax(
                self.Y, axis=1, name="label_test_argmax_sme"), tf.float32)

            #tf.summary.scalar('acuracia', tf.reduce_max(self.accuracies))
            variable_summaries(self.accuracies[0])
            tf.summary.scalar('predicts', self.predicts[0])

            # writer.close()
            # sess.close()

            # self.predicts = predicts_session
            # self.accuracies = train_accuracies_session

            # return train_accuracies_session
            return self.accuracies

    def run_best(self):

        with tf.name_scope('Fitness') as scope:

            print(self.train_y.shape)
            y_size = self.train_y.shape[1]

            self.X = tf.placeholder("float", shape=[None, 28, 28, 1], name="X")
            self.Y = tf.placeholder("float", shape=[None, y_size], name="Y")

            X = self.X
            Y = self.Y

            i = 0

            with tf.name_scope('predicts') as scope:

                predicts = self.conv_net_best()


            with tf.name_scope('accuracies') as scope:

                train_accuracies = tf.map_fn(
                    lambda x: self.get_accuracies(x), predicts)

                #train_accuracies = self.get_accuracies(predicts[0])
            with tf.name_scope('cost') as cost:

                cost = tf.map_fn(lambda pred: tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=self.Y)), predicts)
                #cost = tf.map_fn(lambda pred: -tf.reduce_sum(self.Y * tf.log(pred)), predicts)

            with tf.name_scope('square_mean_error') as scope:

                square_mean_error = tf.map_fn(lambda pred: tf.reduce_mean(tf.squared_difference(tf.cast(tf.argmax(
                pred, axis=1, name="label_test_argmax_sme"),tf.float32), tf.cast(tf.argmax(self.Y, axis=1, name="label_test_argmax_sme"), tf.float32))), predicts)
            with tf.name_scope('root_square_mean_error') as scope:

                root_square_mean_error = tf.map_fn(lambda pred: tf.sqrt(
                    tf.reduce_mean(tf.square(tf.subtract(Y, pred)))), predicts, dtype=tf.float32)
           
            # Utilizacao das acuracias e predicts como tensores
            self.predicts = predicts
            self.argmax_predicts = tf.argmax(predicts[0], 1)
            self.accuracies = train_accuracies
            self.cost = cost
            self.square_mean_error = square_mean_error
            self.root_square_mean_error = root_square_mean_error
            self.label_argmax = tf.cast(tf.argmax(
                self.Y, axis=1, name="label_test_argmax_sme"), tf.float32)

            #tf.summary.scalar('acuracia', tf.reduce_max(self.accuracies))
            # variable_summaries(self.accuracies)
            #tf.summary.scalar('predicts', self.predicts)

            # writer.close()
            # sess.close()

            # self.predicts = predicts_session
            # self.accuracies = train_accuracies_session

            # return train_accuracies_session
            return self.accuracies


def calculate_fitness(neural_networks, layers, logdir):
    # return nn_cube(neural_networks,layers)
    neural_structure = neural_network(neural_networks, layers, logdir)
    return neural_structure.run()
