
import tensorflow as tf
import numpy as np
import utils
import itertools

def create_population(layers, in_weights, in_biases, populationSize):

    tf.reset_default_graph()
    
    population = tf.Variable([30,30],name='populacao')
    population_session_shape = [30,30]

    # convulations_weights = {
    # 'wc1': tf.get_variable('W0', shape=(populationSize,3,3,1,2), initializer=tf.random_normal_initializer()), 
    # 'wc2': tf.get_variable('W1', shape=(populationSize,3,3,2,4), initializer=tf.random_normal_initializer()),
    # 'wc3': tf.get_variable('W2', shape=(populationSize,3,3,4,16), initializer=tf.random_normal_initializer()),
    # 'wc4': tf.get_variable('W3', shape=(populationSize,3,3,16,32), initializer=tf.random_normal_initializer()),
    # 'wc5': tf.get_variable('W4', shape=(populationSize,3,3,32,64), initializer=tf.random_normal_initializer()),
    # 'wc6': tf.get_variable('W5', shape=(populationSize,3,3,64,128), initializer=tf.random_normal_initializer()),
    # 'wc7': tf.get_variable('W6', shape=(populationSize,3,3,128,256), initializer=tf.random_normal_initializer()),
    # 'wc8': tf.get_variable('W7', shape=(populationSize,3,3,256,256), initializer=tf.random_normal_initializer()),
    # 'wc9': tf.get_variable('W8', shape=(populationSize,3,3,256,256), initializer=tf.random_normal_initializer()),
    # 'wc10': tf.get_variable('W9', shape=(populationSize,3,3,256,256), initializer=tf.random_normal_initializer()),
    # # 'wc11': tf.get_variable('W10', shape=(populationSize,3,3,256,256), initializer=tf.random_normal_initializer()),
    # # 'wc12': tf.get_variable('W11', shape=(populationSize,3,3,256,256), initializer=tf.random_normal_initializer()), 
    # # 'wc13': tf.get_variable('W12', shape=(populationSize,3,3,256,256), initializer=tf.random_normal_initializer()), 
    # # 'wc14': tf.get_variable('W13', shape=(populationSize,3,3,256,256), initializer=tf.random_normal_initializer()), 
    # # 'wc15': tf.get_variable('W14', shape=(populationSize,3,3,256,256), initializer=tf.random_normal_initializer()), 
    # # 'wc16': tf.get_variable('W15', shape=(populationSize,3,3,256,256), initializer=tf.random_normal_initializer()), 
    # # 'wc17': tf.get_variable('W16', shape=(populationSize,3,3,256,256), initializer=tf.random_normal_initializer()), 
    # # 'wc18': tf.get_variable('W17', shape=(populationSize,3,3,256,256), initializer=tf.random_normal_initializer()), 
    # # 'wc19': tf.get_variable('W18', shape=(populationSize,3,3,256,256), initializer=tf.random_normal_initializer()), 
    # # 'wc20': tf.get_variable('W19', shape=(populationSize,3,3,256,256), initializer=tf.random_normal_initializer()), 




    # 'wd1': tf.get_variable('Wd1', shape=(populationSize,256,16), initializer=tf.random_normal_initializer()), 
    # 'out': tf.get_variable('Wout', shape=(populationSize,16,10), initializer=tf.random_normal_initializer()), 
    # }

    convulations_weights = {}
    initializer = tf.contrib.layers.variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN', dtype=tf.float32)
    for key,val in in_weights.items():
        
        #convulations_weights[key] = tf.Variable(initial_value=tf.stack(tf.map_fn(lambda x: initializer(list(val)) , tf.range(populationSize), dtype=tf.float32 ) ),dtype=tf.float32,name='w'+ key)#tf.get_variable('w'+ key, shape= (populationSize,) + val, initializer=tf.map_fn(lambda x: tf.keras.initializers.he_normal(), tf.range(populationSize) ) )
        convulations_weights[key] = tf.get_variable('w'+ key, shape= (populationSize,) + val, initializer=tf.random_normal_initializer())
    # biases = {
    #     # 'bc1': tf.get_variable('B0', shape=(populationSize,32), initializer=tf.random_normal_initializer()),
    #     # 'bc2': tf.get_variable('B1', shape=(populationSize,64), initializer=tf.random_normal_initializer()),
    #     # 'bc3': tf.get_variable('B2', shape=(populationSize,128), initializer=tf.random_normal_initializer()),
    #     'bc1': tf.get_variable('B0', shape=(populationSize,2), initializer=tf.random_normal_initializer()),
    #     'bc2': tf.get_variable('B1', shape=(populationSize,4), initializer=tf.random_normal_initializer()),
    #     'bc3': tf.get_variable('B2', shape=(populationSize,16), initializer=tf.random_normal_initializer()),
    #     'bc4': tf.get_variable('B3', shape=(populationSize,32), initializer=tf.random_normal_initializer()),
    #     'bc5': tf.get_variable('B4', shape=(populationSize,64), initializer=tf.random_normal_initializer()),
    #     'bc6': tf.get_variable('B5', shape=(populationSize,128), initializer=tf.random_normal_initializer()),
    #     'bc7': tf.get_variable('B6', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
    #     'bc8': tf.get_variable('B7', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
    #     'bc9': tf.get_variable('B8', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
    #     'bc10': tf.get_variable('B9', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
    #     # 'bc11': tf.get_variable('B10', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
    #     # 'bc12': tf.get_variable('B11', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
    #     # 'bc13': tf.get_variable('B12', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
    #     # 'bc14': tf.get_variable('B13', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
    #     # 'bc15': tf.get_variable('B14', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
    #     # 'bc16': tf.get_variable('B15', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
    #     # 'bc17': tf.get_variable('B16', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
    #     # 'bc18': tf.get_variable('B17', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
    #     # 'bc19': tf.get_variable('B18', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
    #     # 'bc20': tf.get_variable('B19', shape=(populationSize,256), initializer=tf.random_normal_initializer()),
        
        
    #     'bd1': tf.get_variable('Bd1', shape=(populationSize,16), initializer=tf.random_normal_initializer()),
    #     'out': tf.get_variable('Bout', shape=(populationSize,10), initializer=tf.random_normal_initializer()),
    # }

    biases = {}
    for key,val in in_biases.items():
        print(val)
        biases[key] = tf.get_variable(key, shape= (populationSize,val), initializer=tf.random_normal_initializer())

        #biases[key] = tf.Variable(initial_value=tf.stack(tf.map_fn(lambda x: initializer([val]), tf.range(populationSize), dtype=tf.float32 )) ,dtype=tf.float32, name= key)  
        #biases[key] = tf.get_variable(key, shape= (populationSize,val), initializer=tf.map_fn(lambda x: tf.keras.initializers.he_normal(), tf.range(populationSize) ) )
    
    return population, population_session_shape, convulations_weights, biases