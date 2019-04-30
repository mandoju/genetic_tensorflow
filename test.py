import tensorflow as tf
import numpy as np
import time
from crossover import crossover
from create_population import create_population
from choose_best import choose_best, create_constants, choose_best_tensor
# from win10toast import ToastNotifier
from genetic_class import Population
from test_packs import get_biases, get_weight_convolution
import traceback
import sys

weights_convulation_input = {
    # ('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc1': (3, 3, 1, 2),
    # ('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': (3, 3, 2, 4),
    # ('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': (3, 3, 4, 16),
    'wc4': (3, 3, 16, 32),
    'wc5': (3, 3, 32, 64),
    'wc6': (3, 3, 64, 128),
    'wc7': (3, 3, 128, 256),
    'wc8': (3, 3, 256, 256),
    'wc9': (3, 3, 256, 256),
    'wc10': (3, 3, 256, 256),

            'wd1': (256,16),
            'out': (16,10) , #('W6', shape=(128,9), initializer=tf.contrib.layers.xavier_initializer()), 
        },
        'biases': {
            'bc1': (2),
            'bc2': (4),
            'bc3': (16),
            'bc4': (32),
            'bc5': (64),
            'bc6': (128),
            'bc7': (256),
            'bc8': (256),
            'bc9': (256),
            'bc10': (256),
            'bd1': (16),
            'out': (10),
        },
        'fitness': 'cross_entropy',
        'selection': 'tournament',
        'elite': 0.20,
        'genetic_operators': [['crossover',0.10],['mutation_unbiased',0.10],['mutation',0.01],['mutation',0.001],['mutation',0.0001]],
        'genetic_operators_size': [0.10,0.10,0.20,0.20,0.20],
	    'fineTuningRate': 0.05,
        'layers': [785,10],
        'mutationRate': 0.10,
        'logdir': './log/',
        'fineTuning': True
}

biases_input = {
    'bc1': (2),
    'bc2': (4),
    'bc3': (16),
    'bc4': (32),
    'bc5': (64),
    'bc6': (128),
    'bc7': (256),
    'bc8': (256),
    'bc9': (256),
    'bc10': (256),
    'bd1': (16),
    'out': (10)}

if(sys.argv[2]):
    weights_convulation_input = get_weight_convolution(sys.argv[2])
    biases_input = get_biases(sys.argv[2])


geneticSettings = {
    'populationSize': int(sys.argv[1]),
    'epochs': 10,
    'inner_loop': 10,
    'weights_convulation': weights_convulation_input,
    'biases': biases_input,
    'fitness': 'cross_entropy',
    'selection': 'tournament',
    'elite': 0.20,
    'genetic_operators': [['crossover', 0.10], ['mutation_unbiased', 0.10], ['mutation', 0.01], ['mutation', 0.001], ['mutation', 0.0001]],
    'genetic_operators_size': [0.10, 0.10, 0.20, 0.20, 0.20],
    'fineTuningRate': 0.05,
    'layers': [785, 10],
    'mutationRate': 0.10,
    'logdir': './log/',
    'fineTuning': False
}

populationSize = geneticSettings['populationSize']
epochs = geneticSettings['epochs']
layers = geneticSettings['layers']
mutationRate = geneticSettings['mutationRate']
logdir = geneticSettings['logdir']
weights_convulation = geneticSettings['weights_convulation'],
biases = geneticSettings['biases']

start_time = time.time()
begin_time = start_time

# toaster = ToastNotifier()
# toaster.show_toast("Programa iniciado","Rodando programa")

try:

    genetic = Population(geneticSettings)
    genetic.run_epoch()
    print(genetic.neural_networks.accuracies)

    # toaster.show_toast("Sucesso!","Programa finalizado com sucesso")
except Exception as e:
    print(traceback.format_exc())
    print(e)
    # toaster.show_toast("Erro!","Ocorreu um erro")
