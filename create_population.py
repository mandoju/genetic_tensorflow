
import tensorflow as tf
import numpy as np
import utils


def create_population(layers, populationSize):

    createGraph = tf.Graph()
    with createGraph.as_default() as g:
        populationTemp = []
        for i in range(populationSize):
            neuralNetwork_temp = []

            array_temp = (np.random.randn(
                layers[0], layers[1])*np.sqrt(2/1000)).astype(np.float32)

            neuralNetwork = tf.constant(
                array_temp, name='Populacao_peso_1_' + str(i))
            neuralNetwork = utils.pad_up_to(neuralNetwork,[1000,1000],0)
            
            neuralNetwork_temp.append(neuralNetwork)

            for layer in range(len(layers) - 2):
                currentLayer = layer+1
                nextLayer = layer+2

                array_temp = (np.random.randn(layers[currentLayer], layers[nextLayer]) *
                            np.sqrt(2/9)).astype(np.float32)

                neuralNetwork = tf.constant(array_temp,
                                                    name='Populacao_peso_' + str(currentLayer) + '_' + str(i))

                #neuralNetwork = tf.pad( neuralNetwork, paddings, 'CONSTANT', constant_values=0 ) ]
                neuralNetwork = utils.pad_up_to(neuralNetwork,[1000,1000],0)
                neuralNetwork_temp.append(neuralNetwork)

            populationTemp.append(neuralNetwork_temp)
            #    resp = tf.SparseTensor(population,name='geracao_1')
        #population = tf.Variable(population,name='geracao_1')
        populationTemp = tf.stack(populationTemp, name="geracao_1")
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        population_session = sess.run(populationTemp)
        sess.close()

    tf.reset_default_graph()
    
    population = tf.Variable(population_session,name='populacao')

    return population