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


if(len(sys.argv) > 2):
    weights_convulation_input = get_weight_convolution(int(sys.argv[2]))
    biases_input = get_biases(int(sys.argv[2]))
print(weights_convulation_input);

geneticSettings = {
    'populationSize': int(sys.argv[1]),
    'epochs': 1,
    'inner_loop': 10,
    'weights_convulation': weights_convulation_input,
    'biases': biases_input,
    'fitness': 'cross_entropy',
    'selection': 'truncate',
    'elite': 0.20,
    'genetic_operators': [['crossover', 0.10], ['mutation', 0.10], ['mutation_by_layer', 0.00001], ['mutation_by_layer', 0.000001], ['mutation_by_layer', 0.0000001]],
    'genetic_operators_size': [0.10, 0.10, 0.20, 0.20, 0.10],
    'fineTuningRate': 0.05,
    'layers': [785, 10],
    'mutationRate': 0.10,
    'logdir': './log/',
    'fineTuning': True
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
