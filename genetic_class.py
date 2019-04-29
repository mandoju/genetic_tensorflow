from neural_network import Neural_network
from create_population import create_population
from choose_best import choose_best
from crossover import crossover, crossover_conv
from utils import variable_summaries
from tensorflow.python.client import timeline
from genetic_operators import apply_genetic_operatos
from graph import Graph
import numpy as np
import tensorflow as tf
import time
import matplotlib
import pickle
matplotlib.use('Agg')

import matplotlib.pyplot as plt


class Population:

    def __init__(self, geneticSettings):
        self.populationSize = geneticSettings['populationSize']
        self.layers = geneticSettings['layers']
        self.mutationRate = geneticSettings['mutationRate']
        self.population, self.populationShape, self.convulations, self.bias = create_population(geneticSettings['layers'], geneticSettings['weights_convulation'],geneticSettings['biases'],geneticSettings['populationSize'])
        self.neural_networks = Neural_network(
           geneticSettings['populationSize'] , geneticSettings['layers'], self.convulations,self.bias, './log/')
        self.geneticSettings = geneticSettings
        self.current_epoch = 0
        self.eliteSize = int(geneticSettings['elite'] * self.populationSize)
        self.slice_sizes = [self.populationSize * x for x in geneticSettings['genetic_operators_size'] ]
        self.genetic_operators_size = geneticSettings['genetic_operators_size']
        self.fineTuningRate = geneticSettings['fineTuningRate'] 
        self.fineTuningBoolean = geneticSettings['fineTuning']
        #self.slice_sizes.append(self.eliteSize)
    def run_epoch(self):

        #print("neural networks fitness run:")
        #fitness_operator = tf.placeholder(tf.int16)
        start = time.time()
        
        self.neural_networks.run()
        
        self.mutationRate = tf.placeholder(tf.float32,shape=[])
        self.operatorSize = tf.placeholder(tf.float32,shape=[len(self.geneticSettings['genetic_operators_size'])])
        if(self.geneticSettings['fitness'] == 'cross_entropy'):
            fitness = -self.neural_networks.cost
        elif(self.geneticSettings['fitness'] == 'square_mean_error'):
            fitness = -self.neural_networks.square_mean_error 
        elif(self.geneticSettings['fitness'] == 'root_square_mean_error'):
            fitness = -self.neural_networks.root_square_mean_error
        elif(self.geneticSettings['fitness'] == 'cross_entropy_mix_accuracies'):
            fitness = self.neural_networks.accuracies  + self.neural_networks.cost
        else:
            fitness = self.neural_networks.accuracies
        
        best_conv, best_bias, the_best_conv, the_best_bias, mutate_conv, mutate_bias = choose_best(self.geneticSettings['selection'],self.neural_networks.convulations, self.neural_networks.biases, fitness, self.eliteSize)
        
        finish_conv, finish_bias = apply_genetic_operatos(self.geneticSettings['genetic_operators'],self.operatorSize,self.eliteSize,self.convulations,self.bias, best_conv, best_bias, self.populationShape , self.populationSize, self.mutationRate,2,len(self.layers))
            
        merged = tf.summary.merge_all()

        self.current_epoch += 1
       
        sess = tf.Session()
        # writer = tf.summary.FileWriter(self.neural_networks.logdir, sess.graph)
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        start = time.time()
        sess.run(tf.global_variables_initializer())
        print("Global variables:", time.time() - start)

        start = time.time()
        sess.run(tf.local_variables_initializer())
        print("local variables:", time.time() - start)

        pop_array = []
        
        finished_array = []
        
        train_x = self.neural_networks.train_x
        #test_x = self.neural_networks.test_x
        train_y = self.neural_networks.train_y
        #test_y = self.neural_networks.test_y
        # print(len(train_x))
        start_time = time.time()
        acuracias = []
        fitnesses = []
        tempos = []
        print("batchs: " + str(len(train_x)//125))
        mutate = self.geneticSettings['mutationRate']
        print(mutate)
        last_accuracy = 0
        for i in range(self.geneticSettings['epochs']):
            
            print("época: " + str(i))
            start_generation = time.time()

            batch_size = 4000

            for batch in range(len(train_x)//batch_size):
                    #for j in range(self.geneticSettings['inner_loop']):
                        print("batch: " + str(batch))
                        start_batch = time.time()
                        batch_x = train_x[batch*batch_size:min((batch+1)*batch_size,len(train_x))]
                        batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]  

                        print("Mutação atual: " + str(mutate) )
                        print(self.genetic_operators_size)

                        predicts,label_argmax,accuracies,cost,finished_conv,finished_bias = sess.run([self.neural_networks.argmax_predicts,self.neural_networks.label_argmax,self.neural_networks.accuracies,fitness,finish_conv,finish_bias], feed_dict={
                            self.neural_networks.X: batch_x, self.neural_networks.Y: batch_y, self.mutationRate: mutate, self.operatorSize: self.genetic_operators_size} )
                        msg = "Batch: " + str(batch)
                        np.savetxt('predicts_save.txt',predicts)
                        np.savetxt('Y.txt',label_argmax)
                        print("Accuracy: ")
                        print(accuracies)
                        print("Cost: ")
                        print(cost)
                        print("tempo atual: " + str(time.time() - start_time))
                        # if(max(cost) < 3):
                        acuracias.append(max(cost))
                        tempos.append(time.time() - start_time)
                        if(self.fineTuningBoolean):
                            if(max(accuracies) <= last_accuracy):
                                mutate += 0.1
                                if(mutate > 0.7):
                                    mutate = 0.7
                            else:
                                mutate -= 0.1
                                if(mutate < 0.1):
                                    mutate = 0.1
                            last_accuracy = max(accuracies)

                            last_population_slice = 0
                            operators_max = []
                            print(self.genetic_operators_size)
                            for population_slice in self.slice_sizes:
                                slice_finish = int(last_population_slice+population_slice-1)
                                operators_max.append(max(cost[last_population_slice:slice_finish]))
                                last_population_slice += int(population_slice)
                            
                            max_fitness_operator_index = operators_max.index(max(operators_max))
                            min_fitness_operator_index = operators_max.index(min(operators_max))
                            print(self.genetic_operators_size[max_fitness_operator_index])
                            print(self.genetic_operators_size[min_fitness_operator_index])
                            if(self.genetic_operators_size[max_fitness_operator_index] <= (1 - self.fineTuningRate ) and self.genetic_operators_size[min_fitness_operator_index] >= self.fineTuningRate):
                                self.genetic_operators_size[max_fitness_operator_index] += self.fineTuningRate
                                self.genetic_operators_size[min_fitness_operator_index] -= self.fineTuningRate
            mutate = mutate * 2
        sess.close()
        
        with open('./graphs/' + str(self.populationSize) + '_he_menor.pckl', 'wb') as save_graph_file:
            save_graph = Graph(tempos,acuracias)
            pickle.dump(save_graph,save_graph_file)
            print('salvei em: ' + '.\graphs\\' + str(self.populationSize) + '.pckl')
        plt.plot(tempos, acuracias, '-', lw=2)
        plt.grid(True)
        plt.savefig('acuracias.png')
