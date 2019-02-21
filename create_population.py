
import tensorflow as tf
import numpy as np
import utils


def create_population(layers, populationSize):

    tf.reset_default_graph()
    
    population = tf.Variable([30,30],name='populacao')
    population_session_shape = [30,30]

    convulations_weights = {
    'wc1': tf.get_variable('W0', shape=(populationSize,3,3,1,2), initializer=tf.keras.initializers.he_normal()), 
    'wc2': tf.get_variable('W1', shape=(populationSize,3,3,2,4), initializer=tf.keras.initializers.he_normal()),
    'wc3': tf.get_variable('W2', shape=(populationSize,3,3,4,16), initializer=tf.keras.initializers.he_normal()),
    'wc4': tf.get_variable('W3', shape=(populationSize,3,3,16,32), initializer=tf.keras.initializers.he_normal()),
    'wc5': tf.get_variable('W4', shape=(populationSize,3,3,32,64), initializer=tf.keras.initializers.he_normal()),
    'wc6': tf.get_variable('W5', shape=(populationSize,3,3,64,128), initializer=tf.keras.initializers.he_normal()),
    'wc7': tf.get_variable('W6', shape=(populationSize,3,3,128,256), initializer=tf.keras.initializers.he_normal()),
    'wc8': tf.get_variable('W7', shape=(populationSize,3,3,256,256), initializer=tf.keras.initializers.he_normal()),
    'wc9': tf.get_variable('W8', shape=(populationSize,3,3,256,256), initializer=tf.keras.initializers.he_normal()),
    'wc10': tf.get_variable('W9', shape=(populationSize,3,3,256,256), initializer=tf.keras.initializers.he_normal()),
    'wc11': tf.get_variable('W10', shape=(populationSize,3,3,256,256), initializer=tf.keras.initializers.he_normal()),
    'wc12': tf.get_variable('W11', shape=(populationSize,3,3,256,256), initializer=tf.keras.initializers.he_normal()), 
    'wc13': tf.get_variable('W12', shape=(populationSize,3,3,256,256), initializer=tf.keras.initializers.he_normal()), 
    'wc14': tf.get_variable('W13', shape=(populationSize,3,3,256,256), initializer=tf.keras.initializers.he_normal()), 
    'wc15': tf.get_variable('W14', shape=(populationSize,3,3,256,256), initializer=tf.keras.initializers.he_normal()), 
    'wc16': tf.get_variable('W15', shape=(populationSize,3,3,256,256), initializer=tf.keras.initializers.he_normal()), 
    'wc17': tf.get_variable('W16', shape=(populationSize,3,3,256,256), initializer=tf.keras.initializers.he_normal()), 
    'wc18': tf.get_variable('W17', shape=(populationSize,3,3,256,256), initializer=tf.keras.initializers.he_normal()), 
    'wc19': tf.get_variable('W18', shape=(populationSize,3,3,256,256), initializer=tf.keras.initializers.he_normal()), 
    'wc20': tf.get_variable('W19', shape=(populationSize,3,3,256,256), initializer=tf.keras.initializers.he_normal()), 




    'wd1': tf.get_variable('Wd1', shape=(populationSize,256,16), initializer=tf.keras.initializers.he_normal()), 
    'out': tf.get_variable('Wout', shape=(populationSize,16,10), initializer=tf.keras.initializers.he_normal()), 
    }
    biases = {
        # 'bc1': tf.get_variable('B0', shape=(populationSize,32), initializer=tf.keras.initializers.he_normal()),
        # 'bc2': tf.get_variable('B1', shape=(populationSize,64), initializer=tf.keras.initializers.he_normal()),
        # 'bc3': tf.get_variable('B2', shape=(populationSize,128), initializer=tf.keras.initializers.he_normal()),
        'bc1': tf.get_variable('B0', shape=(populationSize,2), initializer=tf.keras.initializers.he_normal()),
        'bc2': tf.get_variable('B1', shape=(populationSize,4), initializer=tf.keras.initializers.he_normal()),
        'bc3': tf.get_variable('B2', shape=(populationSize,16), initializer=tf.keras.initializers.he_normal()),
        'bc4': tf.get_variable('B3', shape=(populationSize,32), initializer=tf.keras.initializers.he_normal()),
        'bc5': tf.get_variable('B4', shape=(populationSize,64), initializer=tf.keras.initializers.he_normal()),
        'bc6': tf.get_variable('B5', shape=(populationSize,128), initializer=tf.keras.initializers.he_normal()),
        'bc7': tf.get_variable('B6', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        'bc8': tf.get_variable('B7', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        'bc9': tf.get_variable('B8', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        'bc10': tf.get_variable('B9', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        'bc11': tf.get_variable('B10', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        'bc12': tf.get_variable('B11', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        'bc13': tf.get_variable('B12', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        'bc14': tf.get_variable('B13', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        'bc15': tf.get_variable('B14', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        'bc16': tf.get_variable('B15', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        'bc17': tf.get_variable('B16', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        'bc18': tf.get_variable('B17', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        'bc19': tf.get_variable('B18', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        'bc20': tf.get_variable('B19', shape=(populationSize,256), initializer=tf.keras.initializers.he_normal()),
        
        
        'bd1': tf.get_variable('Bd1', shape=(populationSize,16), initializer=tf.keras.initializers.he_normal()),
        'out': tf.get_variable('Bout', shape=(populationSize,10), initializer=tf.keras.initializers.he_normal()),
    }
    return population, population_session_shape, convulations_weights, biases