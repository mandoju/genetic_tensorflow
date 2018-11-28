
import tensorflow as tf
import numpy as np


def create_population(layers, populationSize):
    population = []
    for i in range(populationSize):
        neuralNetwork_temp = []

        array_temp = (np.random.randn(
            layers[0], layers[1])*np.sqrt(2/1000)).astype(np.float32)

        neuralNetwork_temp.append(tf.constant(
            array_temp, name='Populacao_peso_1_' + str(i)))

        for layer in range(len(layers) - 2):
            currentLayer = layer+1
            nextLayer = layer+2

            array_temp = (np.random.randn(layers[currentLayer], layers[nextLayer]) *
                          np.sqrt(2/9)).astype(np.float32)

            neuralNetwork_temp.append(tf.constant(array_temp,
                                                  name='Populacao_peso_' + str(currentLayer) + '_' + str(i)))

        population.append(neuralNetwork_temp)

    return population
