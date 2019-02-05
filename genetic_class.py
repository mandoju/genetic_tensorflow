from neural_network import Neural_network
from create_population import create_population
from choose_best import choose_best_tensor
from crossover import crossover
from utils import variable_summaries
from tensorflow.python.client import timeline
import numpy as np
import tensorflow as tf
import time


class Population:

    def __init__(self, populationSize, layers, mutationRate):
        self.populationSize = populationSize
        self.layers = layers
        self.mutationRate = mutationRate
        self.population, self.populationShape, self.convulations, self.bias = create_population(layers, populationSize)
        self.neural_networks = Neural_network(
           self.population , layers, self.convulations,self.bias, './log/')
        self.current_epoch = 0

    def run_epoch(self):

        #print("neural networks fitness run:")
        start = time.time()
        self.neural_networks.run()
        #print("neural network : ", time.time() - start)
        print(self.neural_networks.accuracies)
        best = choose_best_tensor(
            self.neural_networks.neural_networks, self.neural_networks.accuracies)

        new_population = crossover(best,self.population, self.populationShape , self.populationSize, 0.01,2,len(self.layers))

        finish = new_population

        #variable_summaries(self.population)
        merged = tf.summary.merge_all()

        self.current_epoch += 1
       
        sess = tf.Session()
       # writer = tf.summary.FileWriter(self.neural_networks.logdir, sess.graph)
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()

        start = time.time()
        sess.run(tf.global_variables_initializer())
        print("Global variables:", time.time() - start)

        start = time.time()
        sess.run(tf.local_variables_initializer())
        print("local variables:", time.time() - start)

        pop_array = []
        
        finished_array = []
        
        for i in range(100):

            print("época: " + str(i))
            start = time.time()
            
            pop,accuracies,finished = sess.run([self.population,self.neural_networks.accuracies,finish], feed_dict={
                self.neural_networks.X: self.neural_networks.train_x, self.neural_networks.Y: self.neural_networks.train_y})    

            print(accuracies)
            print("tempo:" + str(time.time() - start))

            #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            #with open('./log/timeline.ctf.json', 'w') as trace_file:
            #    trace_file.write(trace.generate_chrome_trace_format())
            #print(pop)
            #print("---------")
            #print(finished)

        sess.close()
        #writer.close()

        # if(np.all(pop_array[0] == pop_array[1])):
        #     print("populações iguais")
        # else:
        #     print("mudou")
        # if(np.all(pop_array[1] == finished_array[0])):
        #     print("população secundária funcionando")
        # else:
        #     print("população secundária não funcionando")