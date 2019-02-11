import tensorflow as tf
import numpy as np
import time
from crossover import  crossover
from create_population import create_population
from choose_best import  choose_best, create_constants, choose_best_tensor
# from win10toast import ToastNotifier
from genetic_class import Population
import traceback

geneticSettings = {
        'populationSize': 100,
        'epochs': 10,
        'weights_convulation': {
            'wc1': (3,3,3,1,312) ,#tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc2': (3,3,32,64) , #tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
            'wc3': (3,3,64,128) ,#tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
            'wd1': (4*4*128,128) ,#tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
            'out': (128,9) ,#tf.get_variable('W6', shape=(128,9), initializer=tf.contrib.layers.xavier_initializer()), 
        },
        'biases': {
            'bc1': 32 ,#tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
            'bc2': 64 ,#tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
            'bc3': 128 ,#tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'bd1': 128 ,#tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
            'out': 10 ,#tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
        },
        'layers': [785,10],
        'mutationRate': 0.20,
        'logdir': './log/'
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

    genetic = Population(populationSize,layers,mutationRate,weights_convulation,biases)
    genetic.run_epoch()
    print(genetic.neural_networks.accuracies)


    # toaster.show_toast("Sucesso!","Programa finalizado com sucesso")
except Exception as e:
    print(traceback.format_exc())
    print(e)
    # toaster.show_toast("Erro!","Ocorreu um erro")