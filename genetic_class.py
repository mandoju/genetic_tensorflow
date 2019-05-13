from neural_network import Neural_network
from create_population import create_population
from choose_best import choose_best
from crossover import crossover, crossover_conv
from utils import variable_summaries
from tensorflow.python.client import timeline
from genetic_operators import apply_genetic_operatos
from graph import Graph
from statistics import mean 
from tensorflow.python import debug as tf_debug
import tkinter as tk
import numpy as np
import tensorflow as tf
import time
import matplotlib
import pickle
import sys

#matplotlib.use('Agg')

import matplotlib.pyplot as plt


class Population:

    def __init__(self, geneticSettings):
        self.populationSize = geneticSettings['populationSize']
        self.layers = geneticSettings['layers']
        self.mutationRate = geneticSettings['mutationRate']
        self.population, self.populationShape, self.convulations, self.bias = create_population(geneticSettings['layers'], geneticSettings['weights_convulation'],geneticSettings['biases'],geneticSettings['populationSize'])
        self.neural_networks = Neural_network(
           geneticSettings['populationSize'] , geneticSettings['layers'], self.convulations,self.bias, geneticSettings['train_x'], geneticSettings['train_y'], geneticSettings['test_x'], geneticSettings['test_y'],'./log/')
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
        
        best_conv, best_bias = choose_best(self.geneticSettings['selection'],self.neural_networks.convulations, self.neural_networks.biases, fitness, self.eliteSize)
        
        finish_conv, finish_bias = apply_genetic_operatos(self.geneticSettings['genetic_operators'],self.operatorSize,self.eliteSize,self.convulations,self.bias, best_conv, best_bias, self.populationShape , self.populationSize, self.mutationRate,2,len(self.layers))
            
        merged = tf.summary.merge_all()

        self.current_epoch += 1
        sess = tf.Session()

        writer = tf.summary.FileWriter(self.neural_networks.logdir, sess.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        start = time.time()
        sess.run(tf.global_variables_initializer())
        print("Global variables:", time.time() - start)

        start = time.time()
        sess.run(tf.local_variables_initializer())
        print("local variables:", time.time() - start)

        pop_array = []
        
        finished_array = []
        
        train_x = self.neural_networks.train_x
        train_y = self.neural_networks.train_y
        
        acuracias = []
        fitnesses = []
        validation_fitnesses = []
        validation_acuracias = []
        tempos = []
        tempos_validation = []
        fine_tuning_graph = []
        session_times = []

        print("batchs: " + str(len(train_x)//125))
        mutate = self.geneticSettings['mutationRate']
        print(mutate)
        last_cost = 0
        last_best_cost = -99999
        start_time = time.time()
        for i in range(self.geneticSettings['epochs']):
            
            print("época: " + str(i))
            start_generation = time.time()

            batch_size = 10000

            for batch in range(1):
            #for batch in range( (len(train_x)//batch_size ) - 1 ):
                    #for j in range(self.geneticSettings['inner_loop']):
                        print("batch: " + str(batch))
                        start_batch = time.time()
                        batch_x = train_x[batch*batch_size:min((batch+1)*batch_size,len(train_x))]
                        batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]  

                        #print("Mutação atual: " + str(mutate) )
                        print(self.slice_sizes)
                        fine_tuning_graph.append(self.slice_sizes[:])

                        session_time = time.time()


                        accuracies,cost,finished_conv,finished_bias = sess.run([self.neural_networks.accuracies,fitness,finish_conv,finish_bias], feed_dict={
                               self.neural_networks.X: batch_x, self.neural_networks.Y: batch_y, self.mutationRate: mutate, self.operatorSize: self.slice_sizes})#, options=run_options, run_metadata=run_metadata )
               

                        #predicts,label_argmax,accuracies,cost,sess_best_conv , sess_best_bias = sess.run([self.neural_networks.predicts,self.neural_networks.label_argmax,self.neural_networks.accuracies,fitness,best_conv,best_bias], feed_dict={
                        #        self.neural_networks.X: batch_x, self.neural_networks.Y: batch_y, self.mutationRate: mutate, self.operatorSize: self.slice_sizes})#, options=run_options, run_metadata=run_metadata )

                        #finished_conv = [1]
                        #finished_bias = [2]
                        #print("sessao demorou: " +  str(time.time() - session_time))
                        session_times.append(str(time.time() - session_time))
                        writer.add_run_metadata(run_metadata,'step%s' % (str(batch) + '_' +str(i)) )
                        msg = "Batch: " + str(batch)
                        #np.savetxt('predicts_save.txt',predicts)
                        #np.savetxt('Y.txt',label_argmax)
                        print("Accuracy: ")
                        print(accuracies)
                        print("Cost: ")
                        print(cost)
                        print("tempo atual: " + str(time.time() - start_time))
                        # if(max(cost) < 3):
                        fitnesses.append(max(cost))
                        acuracias.append(max(accuracies))
                        tempos.append(time.time() - start_time)
                        if(self.fineTuningBoolean):
                            # if(max(accuracies) <= last_accuracy):
                            #     mutate += 0.1
                            #     if(mutate > 0.7):
                            #         mutate = 0.7
                            # else:
                            #     mutate -= 0.1
                            #     if(mutate < 0.1):
                            #         mutate = 0.1
                            last_cost = max(cost)

                            last_population_slice = self.eliteSize
                            operators_max = []
                            for population_slice in self.slice_sizes:
                                slice_finish = int(last_population_slice+population_slice-1)
                                if(population_slice > 1):
                                    operators_max.append(np.mean(cost[last_population_slice:slice_finish]))
                                else:
                                    operators_max.append(cost[last_population_slice])
                                last_population_slice += int(population_slice)
                            
                            max_fitness_operator_index = operators_max.index(max(operators_max))
                            
                            #possible_slices_remove = self.slice_sizes
                            #minimum_not_one = possible_slices[operators_max.index(min(operators_max))]
                            slice_with_operator = np.column_stack((self.slice_sizes,operators_max,range(len(self.slice_sizes))))
                            slice_with_operator = list(filter(lambda x: x[0] > self.populationSize * 0.05,slice_with_operator))
                            print(slice_with_operator)
                            min_fitness_slice = min(slice_with_operator,key=lambda x: x[1])
                            print(min_fitness_slice)
                            min_fitness_operator_index = int(min_fitness_slice[2])
                            print(min_fitness_operator_index)
                            if(self.slice_sizes[min_fitness_operator_index] > 1):
                                self.slice_sizes[max_fitness_operator_index] += 1
                                self.slice_sizes[min_fitness_operator_index] -= 1
                            # if(batch % 30 == 0):
                            #     if(self.slice_sizes[1] > 18 or self.slice_sizes[2] > 18):
                            #         mutate = mutate * 10
                            #         print('mutei')
                            #     if(self.slice_sizes[0] > 18 or self.slice_sizes[4] > 18):
                            #         mutate = mutate / 10
                            #         print('mutei')
                            #     last_best_cost = max(cost)

                            

            #batch = (len(train_x)//batch_size ) - 1
            #batch_x = train_x[batch*batch_size:min((batch+1)*batch_size,len(train_x))]
            #batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]  
            final_predict,accuracies,cost = sess.run([self.neural_networks.predicts,self.neural_networks.accuracies,fitness], feed_dict={
                                self.neural_networks.X: train_x, self.neural_networks.Y: train_y})
            print("acuracia:" + str(accuracies[0]))
            validation_acuracias.append(accuracies[0])
            print("fitness:" + str(cost[0]))
            validation_fitnesses.append(cost[0])
            tempos_validation.append(time.time() - start_time)
            
            # mutate = mutate * 2
        sess.close()
        file_string = []
        if(len(sys.argv) > 2):
          file_string = './graphs/' + str(self.populationSize)  + '_' +  sys.argv[2] +   '.pckl'
        else:
          file_string = './graphs/' + str(self.populationSize)  + '_10.pckl'
        with open(file_string, 'wb') as save_graph_file:
            save_graph = Graph(tempos,fitnesses,acuracias,tempos_validation,validation_fitnesses,validation_acuracias,fine_tuning_graph)
            pickle.dump(save_graph,save_graph_file)
            print('salvei em: ' + '.\graphs\\' + str(self.populationSize) + '.pckl')
        
        if(len(sys.argv) > 2):
          file_string = './session_times/all_' + str(self.populationSize)  + '_' +  sys.argv[2] +   '.pckl'
        else:
          file_string = './session_times/all_' + str(self.populationSize)  + '_10.pckl'
        with open(file_string, 'wb') as save_graph_file:
            save_graph = session_times
            pickle.dump(save_graph,save_graph_file)
            print('salvei em: ' + '.\session_times\\' + str(self.populationSize) + '.pckl')

        #plt.plot(tempos, acuracias, '-', lw=2)
        #plt.grid(True)
        #plt.savefig('acuracias.png')
        #plt.plot(train_x,train_y,'-',label="seno")
        #plt.plot(train_x,final_predict[0],'-',label="neural_network")
        #plt.grid(True)
        #plt.show()