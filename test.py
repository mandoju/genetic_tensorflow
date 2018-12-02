import tensorflow as tf
import numpy as np
import time
from neural_network import calculate_fitness
from crossover import  crossover
from create_population import create_population
from choose_best import  choose_best, create_constants

geneticSettings = {
        'populationSize': 10,
        'epochs': 10,
        'layers': [785,900,9],
        'mutationRate': 0.20
}

populationSize = geneticSettings['populationSize']
epochs = geneticSettings['epochs']
layers = geneticSettings['layers']
mutationRate = geneticSettings['mutationRate']

start_time = time.time()
begin_time = start_time

g1 = tf.Graph()
with g1.as_default() as g:
    with g.name_scope("g1") as g1_scope:
        population = create_population(layers,populationSize)
        print("--- Population: %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        fitness = calculate_fitness(population,layers)

        print("--- Fitness: %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        #best_ones = choose_best(population, fitness)

        #print("--- Best Ones: %s seconds ---" % (time.time() - start_time))
        #start_time = time.time()
