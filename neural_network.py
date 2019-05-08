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

    def conv_net(self,X):
        
        weights = self.convulations
        biases = self.biases
        
        layer_1 = Layer(self.populationSize,weights['wd1'], biases['bc1'],'wd',tf.math.tanh)
        layer_2 = Layer(self.populationSize,weights['wd2'], biases['bc2'],'wd',tf.math.tanh)
        layer_3 = Layer(self.populationSize,weights['out'], biases['out'],'wd')

        layer_1_out = layer_1.run_fist_layer(X)
        layer_2_out = layer_2.run(layer_1_out)
        layer_3_out = layer_3.run(layer_2_out)

        return layer_3_out
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

    def get_accuracies(self, predict):

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

    def get_cost_functions(self, predict, train, test):
        with tf.name_scope('calculo_da_acuracia') as scope:

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

            with tf.name_scope('predicts') as scope:

                self.predicts = self.conv_net(X)
                predicts = tf.stack(self.predicts)
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
            #self.predicts = predicts
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
