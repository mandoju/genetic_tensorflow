from neural_network import Neural_network
from create_population import create_population
from choose_best import choose_best_tensor, choose_best_tensor_conv
from crossover import crossover, crossover_conv
from utils import variable_summaries
from tensorflow.python.client import timeline
import numpy as np
import tensorflow as tf
import time
#import matplotlib.pyplot as plt


class Population:

    def __init__(self, populationSize, layers, mutationRate, weights_convulation, biases):
        self.populationSize = populationSize
        self.layers = layers
        self.mutationRate = mutationRate
        self.population, self.populationShape, self.convulations, self.bias = create_population(layers, populationSize)
        self.neural_networks = Neural_network(
           populationSize , layers, self.convulations,self.bias, './log/')
        self.current_epoch = 0
        

    def run_epoch(self):

        #print("neural networks fitness run:")
        start = time.time()
        for i in range(1):
            self.neural_networks.run()
            
            #print("neural network : ", time.time() - start)
            #best = choose_best_tensor(
            #    self.neural_networks.neural_networks, self.neural_networks.accuracies)

            inverted_cost = -self.neural_networks.cost #tf.multiply(self.neural_networks.cost,tf.constant(-0.1),name="inverted_costs")
            inverted_sqe = -self.neural_networks.square_mean_error #tf.multiply(self.neural_networks.square_mean_error , tf.constant(-0.1),name="inverted_sqe")
            #fitness = self.neural_networks.accuracies * 100  + inverted_sqe + inverted_cost
            #fitness = self.neural_networks.accuracies
            fitness = inverted_cost  #+ inverted_sqe / 2
            best_conv, best_bias, the_best_conv, the_best_bias, mutate_conv, mutate_bias = choose_best_tensor_conv(self.neural_networks.convulations, self.neural_networks.biases, fitness, self.populationSize // 10)
            # self.neural_networks.best_conv = the_best_conv
            # self.neural_networks.best_bias = the_best_bias
            # best_accuracies = self.neural_networks.run_best()

            #new_population = crossover(best,self.population, self.populationShape , self.populationSize, self.mutationRate,2,len(self.layers))
            finish_conv, finish_bias = crossover_conv(best_conv,best_bias,mutate_conv,mutate_bias,self.convulations,self.bias, self.populationShape , self.populationSize, self.mutationRate,2,len(self.layers))
            #self.neural_networks.convulations = finish_conv
            #self.neural_networks.biases = finish_bias

        #finish = finish_conv.append(finish_bias)

        #variable_summaries(self.population)
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
        tempos = []
        print("batchs: " + str(len(train_x)//125))
        for i in range(10):
            
            print("época: " + str(i))
            start_generation = time.time()

            batch_size = 1000
            for batch in range(len(train_x)//batch_size):
                print("batch: " + str(batch))
                start_batch = time.time()
                batch_x = train_x[batch*batch_size:min((batch+1)*batch_size,len(train_x))]
                batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]  

                accuracies,cost,finished_conv,finished_bias = sess.run([self.neural_networks.accuracies,fitness,finish_conv,finish_bias], feed_dict={
                    self.neural_networks.X: batch_x, self.neural_networks.Y: batch_y} )
                
                # options=run_options, run_metadata=run_metadata )
                # print("Accuracy: ")
                # print(accuracies)
                # print("Cost: ").
                # print(cost)
                # print("tempo batch: " + str(time.time() - start_batch))

                # if(batch == (len(train_x)//batch_size) - 1 ):
                #     print(accuracies)
                #     print("tempo:" + str(time.time() - start_generation))

                print("Accuracy: ")
                print(accuracies)
                acuracias.append(accuracies)
                print("Cost: ")
                print(cost)
                print("tempo atual: " + str(time.time() - start_time))
                tempos.append(time.time() - start_time)
            
                # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                # with open('./log/timeline.ctf.json', 'w') as trace_file:
                #     trace_file.write(trace.generate_chrome_trace_format())
            #print(pop)
            #print("---------")
            #print(finished)

        sess.close()
        # plt.plot(tempos, acuracias, '-', lw=2)
        # plt.grid(True)
        # plt.show()
    
        # tf.reset_default_graph()
        # createGraph = tf.Graph()
        # with createGraph.as_default() as test_graph:
        #     self.neural_networks = Neural_network(
        #         self.population , self.layers, the_best_conv, the_best_bias, './log/')
        #     final_accuracies = self.neural_networks.run()
        #     test_accuracies = sess.run(final_accuracies, feed_dict={
        #         self.neural_networks.X: test_x, self.neural_networks.Y: test_y})
        #     print(test_accuracies)


        #writer.close()

        # if(np.all(pop_array[0] == pop_array[1])):
        #     print("populações iguais")
        # else:
        #     print("mudou")
        # if(np.all(pop_array[1] == finished_array[0])):
        #     print("população secundária funcionando")
        # else:
        #     print("população secundária não funcionando")