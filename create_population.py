
import tensorflow as tf
import numpy as np
import utils


def create_population(layers, populationSize):

    # createGraph = tf.Graph()
    # with createGraph.as_default() as create_bootstrap:
    #     populationTemp = []
    #     for i in range(populationSize):
    #         neuralNetwork_temp = []

           
    #         array_temp = (np.random.randn(
    #             layers[0], layers[1])*np.sqrt(2/1000)).astype(np.float32)

    #         neuralNetwork = tf.constant(
    #             array_temp, name='Populacao_peso_1_' + str(i))
    #         neuralNetwork = utils.pad_up_to(neuralNetwork,[max(layers),max(layers)],0)
            
    #         neuralNetwork_temp.append(neuralNetwork)

    #         for layer in range(len(layers) - 2):
    #             currentLayer = layer+1
    #             nextLayer = layer+2

    #             array_temp = (np.random.randn(layers[currentLayer], layers[nextLayer]) *
    #                         np.sqrt(2/9)).astype(np.float32)

    #             neuralNetwork = tf.constant(array_temp,
    #                                                 name='Populacao_peso_' + str(currentLayer) + '_' + str(i))

    #             #neuralNetwork = tf.pad( neuralNetwork, paddings, 'CONSTANT', constant_values=0 ) ]
    #             neuralNetwork = utils.pad_up_to(neuralNetwork,[max(layers),max(layers)],0)
    #             neuralNetwork_temp.append(neuralNetwork)

    #         populationTemp.append(neuralNetwork_temp)
    #         #    resp = tf.SparseTensor(population,name='geracao_1')
    #     #population = tf.Variable(population,name='geracao_1')
    #     populationTemp = tf.stack(populationTemp, name="geracao_1")
        
    #     sess = tf.Session()
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    #     population_session, population_session_shape = sess.run([populationTemp,tf.shape(populationTemp)])
    #     sess.close()

    tf.reset_default_graph()
    
    population = tf.Variable([30,30],name='populacao')
    population_session_shape = [30,30]

    #filter_shape = [28,28]
    #conv_filt_shape = [populationSize, 1, 28, 28, 1]

    #convulations_weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
    #                                  name='conv_W')
    #bias = tf.Variable(tf.truncated_normal([populationSize,28]), name='conv_b')

    # convulations_weights = {
    #     'wc1': tf.get_variable('W0', shape=(populationSize,3,3,1,4), initializer=tf.contrib.layers.xavier_initializer()), 
    #     'wc2': tf.get_variable('W1', shape=(populationSize,3,3,4,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'wc3': tf.get_variable('W2', shape=(populationSize,3,3,4,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'wc4': tf.get_variable('W3', shape=(populationSize,3,3,4,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'wc5': tf.get_variable('W4', shape=(populationSize,3,3,4,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'wc6': tf.get_variable('W5', shape=(populationSize,3,3,4,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'wc7': tf.get_variable('W6', shape=(populationSize,3,3,4,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'wc8': tf.get_variable('W7', shape=(populationSize,3,3,4,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'wc9': tf.get_variable('W8', shape=(populationSize,3,3,4,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'wc10': tf.get_variable('W9', shape=(populationSize,3,3,4,8), initializer=tf.contrib.layers.xavier_initializer()),
    #     'wc11': tf.get_variable('W10', shape=(populationSize,3,3,8,16), initializer=tf.contrib.layers.xavier_initializer()),
    #     'wc12': tf.get_variable('W11', shape=(populationSize,3,3,16,32), initializer=tf.contrib.layers.xavier_initializer()), 
    #     'wc13': tf.get_variable('W12', shape=(populationSize,3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    #     'wc14': tf.get_variable('W13', shape=(populationSize,3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
      
    #     'wd1': tf.get_variable('Wd1', shape=(populationSize,128,16), initializer=tf.contrib.layers.xavier_initializer()), 
    #     'out': tf.get_variable('Wout', shape=(populationSize,16,10), initializer=tf.contrib.layers.xavier_initializer()), 
    # }
    # biases = {
    #     'bc1': tf.get_variable('B0', shape=(populationSize,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'bc2': tf.get_variable('B1', shape=(populationSize,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'bc3': tf.get_variable('B2', shape=(populationSize,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'bc4': tf.get_variable('B3', shape=(populationSize,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'bc5': tf.get_variable('B4', shape=(populationSize,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'bc6': tf.get_variable('B5', shape=(populationSize,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'bc7': tf.get_variable('B6', shape=(populationSize,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'bc8': tf.get_variable('B7', shape=(populationSize,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'bc9': tf.get_variable('B8', shape=(populationSize,4), initializer=tf.contrib.layers.xavier_initializer()),
    #     'bc10': tf.get_variable('B9', shape=(populationSize,8), initializer=tf.contrib.layers.xavier_initializer()),
    #     'bc11': tf.get_variable('B10', shape=(populationSize,16), initializer=tf.contrib.layers.xavier_initializer()),
    #     'bc12': tf.get_variable('B11', shape=(populationSize,32), initializer=tf.contrib.layers.xavier_initializer()),
    #     'bc13': tf.get_variable('B12', shape=(populationSize,64), initializer=tf.contrib.layers.xavier_initializer()),
    #     'bc14': tf.get_variable('B13', shape=(populationSize,128), initializer=tf.contrib.layers.xavier_initializer()),
        
    #     'bd1': tf.get_variable('Bd1', shape=(populationSize,16), initializer=tf.contrib.layers.xavier_initializer()),
    #     'out': tf.get_variable('Bout', shape=(populationSize,10), initializer=tf.contrib.layers.xavier_initializer()),
    # }

    convulations_weights = {
    # 'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    # 'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    # 'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wc1': tf.get_variable('W0', shape=(populationSize,3,3,1,2), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(populationSize,3,3,2,4), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.get_variable('W2', shape=(populationSize,3,3,4,16), initializer=tf.contrib.layers.xavier_initializer()),
    'wc4': tf.get_variable('W3', shape=(populationSize,3,3,16,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc5': tf.get_variable('W4', shape=(populationSize,3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc6': tf.get_variable('W5', shape=(populationSize,3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wc7': tf.get_variable('W6', shape=(populationSize,3,3,128,256), initializer=tf.contrib.layers.xavier_initializer()),
    'wc8': tf.get_variable('W7', shape=(populationSize,3,3,256,256), initializer=tf.contrib.layers.xavier_initializer()),
    'wc9': tf.get_variable('W8', shape=(populationSize,3,3,256,256), initializer=tf.contrib.layers.xavier_initializer()),
    'wc10': tf.get_variable('W9', shape=(populationSize,3,3,256,256), initializer=tf.contrib.layers.xavier_initializer()),
    'wc11': tf.get_variable('W10', shape=(populationSize,3,3,256,256), initializer=tf.contrib.layers.xavier_initializer()),
    'wc12': tf.get_variable('W11', shape=(populationSize,3,3,256,256), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc13': tf.get_variable('W12', shape=(populationSize,3,3,256,256), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc14': tf.get_variable('W13', shape=(populationSize,3,3,256,256), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('Wd1', shape=(populationSize,256,16), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('Wout', shape=(populationSize,16,10), initializer=tf.contrib.layers.xavier_initializer()), 
    }
    biases = {
        # 'bc1': tf.get_variable('B0', shape=(populationSize,32), initializer=tf.contrib.layers.xavier_initializer()),
        # 'bc2': tf.get_variable('B1', shape=(populationSize,64), initializer=tf.contrib.layers.xavier_initializer()),
        # 'bc3': tf.get_variable('B2', shape=(populationSize,128), initializer=tf.contrib.layers.xavier_initializer()),
        'bc1': tf.get_variable('B0', shape=(populationSize,2), initializer=tf.contrib.layers.xavier_initializer()),
        'bc2': tf.get_variable('B1', shape=(populationSize,4), initializer=tf.contrib.layers.xavier_initializer()),
        'bc3': tf.get_variable('B2', shape=(populationSize,16), initializer=tf.contrib.layers.xavier_initializer()),
        'bc4': tf.get_variable('B3', shape=(populationSize,32), initializer=tf.contrib.layers.xavier_initializer()),
        'bc5': tf.get_variable('B4', shape=(populationSize,64), initializer=tf.contrib.layers.xavier_initializer()),
        'bc6': tf.get_variable('B5', shape=(populationSize,128), initializer=tf.contrib.layers.xavier_initializer()),
        'bc7': tf.get_variable('B6', shape=(populationSize,256), initializer=tf.contrib.layers.xavier_initializer()),
        'bc8': tf.get_variable('B7', shape=(populationSize,256), initializer=tf.contrib.layers.xavier_initializer()),
        'bc9': tf.get_variable('B8', shape=(populationSize,256), initializer=tf.contrib.layers.xavier_initializer()),
        'bc10': tf.get_variable('B9', shape=(populationSize,256), initializer=tf.contrib.layers.xavier_initializer()),
        'bc11': tf.get_variable('B10', shape=(populationSize,256), initializer=tf.contrib.layers.xavier_initializer()),
        'bc12': tf.get_variable('B11', shape=(populationSize,256), initializer=tf.contrib.layers.xavier_initializer()),
        'bc13': tf.get_variable('B12', shape=(populationSize,256), initializer=tf.contrib.layers.xavier_initializer()),
        'bc14': tf.get_variable('B13', shape=(populationSize,256), initializer=tf.contrib.layers.xavier_initializer()),
        
        'bd1': tf.get_variable('Bd1', shape=(populationSize,16), initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('Bout', shape=(populationSize,10), initializer=tf.contrib.layers.xavier_initializer()),
    }
    return population, population_session_shape, convulations_weights, biases