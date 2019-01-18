import tensorflow as tf
import numpy as np
import time
from neural_network import calculate_fitness
from crossover import  crossover
from create_population import create_population
from choose_best import  choose_best, create_constants, choose_best_tensor
from win10toast import ToastNotifier
from genetic_class import Population
import traceback

geneticSettings = {
        'populationSize': 10,
        'epochs': 10,
        'layers': [785,900,9],
        'mutationRate': 0.20,
        'logdir': './log/'
}

populationSize = geneticSettings['populationSize']
epochs = geneticSettings['epochs']
layers = geneticSettings['layers']
mutationRate = geneticSettings['mutationRate']
logdir = geneticSettings['logdir']

start_time = time.time()
begin_time = start_time

toaster = ToastNotifier()
toaster.show_toast("Programa iniciado","Rodando programa")

try:

    genetic = Population(populationSize,layers,mutationRate)
    genetic.run_epoch()
    print(genetic.neural_networks.accuracies)


    toaster.show_toast("Sucesso!","Programa finalizado com sucesso")
except Exception as e:
    print(traceback.format_exc())
    print(e)
    toaster.show_toast("Erro!","Ocorreu um erro")