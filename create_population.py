
import tensorflow as tf
import numpy as np


def create_population(populationSize):
    population = []
    for i in range(populationSize):
        # w_1 = np.random.rand(5, 10).astype('f') * 0.01;
        # w_2 = np.random.rand(10, 10000).astype('f') * 0.01;
        # w_3 = np.random.rand(10000, 500).astype('f') * 0.01;
        # w_4 = np.random.rand(500, 3).astype('f') * 0.01;

        # w_1 = tf.Variable(tf.random_normal((785,1000), stddev=0.1,seed=1));
        #w_1 = tf.constant(np.random.normal(loc=0, scale=0.1, size=(785, 1000)).astype(np.float32),
        array_temp = (np.random.randn(785,1000)*np.sqrt(2/1000)).astype(np.float32);
        #print(array_temp);
        w_1 = tf.constant( array_temp ,
                          name='Populacao_peso_1_' + str(i));

        # w_2 = tf.random_normal((10, 10000), stddev=0.1);
        # w_3 = tf.random_normal((10000, 500), stddev=0.1);

        # w_4 = tf.Variable(tf.random_normal((1000, 9), stddev=0.1,seed=1));
        # w_4 = tf.constant(np.random.normal(loc=0, scale=0.1, size=(1000, 9)).astype(np.float32) * ,
        array_temp = (np.random.randn(1000,9)*np.sqrt(2/9)).astype(np.float32);
        #print(array_temp);

        w_4 = tf.constant(array_temp,
                          name='Populacao_peso_2_' + str(i));
        population.append([w_1, w_4]);
        # print(population)
    return population