import tensorflow as tf
import numpy as np
import time
from neural_network import calculate_fitness
from crossover import  crossover
from create_population import create_population
from choose_best import  choose_best, create_constants
# from itertools import filterfalse
# import pandas as pd
# from tensorflow.examples.tutorials.mnist import input_data
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from tensorflow.python import debug as tf_debug
# from random import randint
import random


# import copy


if __name__ == "__main__":
    genetic_pool_settings = {
        'populationSize': 30,
        'tournamentSize': 4,
        'memberDimensions': [4, 3, 2, 3, 4],
        'mutationRate': 0.05,
        'averagesCount': 1,
        'maxEpochs': 10
    };

    population_size = 4;

    start_time = time.time()

    g1 = tf.Graph()
    with g1.as_default() as g:
        with g.name_scope("g1") as g1_scope:
            population = create_population(population_size);
            print("--- Population: %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

            fitness = calculate_fitness(population);

            print("--- Fitness: %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

            best_ones = choose_best(population, fitness);

            print("--- Best Ones: %s seconds ---" % (time.time() - start_time))
            start_time = time.time()

    for i in range(1):
        g1 = tf.Graph()
        with g1.as_default() as g:
            with g.name_scope("g" + str(i)) as g1_scope:

                constants = create_constants(best_ones);
                population = crossover(constants, population_size);

                print("--- Crossover: %s seconds ---" % (time.time() - start_time))
                start_time = time.time()

                fitness = calculate_fitness(population);

                print("--- Fitness: %s seconds ---" % (time.time() - start_time))
                start_time = time.time()

                best_ones = choose_best(population, fitness);

                print("--- Best Ones: %s seconds ---" % (time.time() - start_time))
                start_time = time.time()

#    fitness = calculate_fitness(population);
#    best_ones = choose_best(population,fitness);
#    population = crossover(best_ones,3);
