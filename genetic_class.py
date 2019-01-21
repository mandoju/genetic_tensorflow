from neural_network import Neural_network
from create_population import create_population
from choose_best import choose_best_tensor
from crossover import crossover
from utils import variable_summaries
import tensorflow as tf
import time


class Population:

    def __init__(self, populationSize, layers, mutationRate):
        self.populationSize = populationSize
        self.layers = layers
        self.mutationRate = mutationRate
        self.population, self.populationShape = create_population(layers, populationSize)
        self.neural_networks = Neural_network(
           self.population , layers, './log/')
        self.current_epoch = 0

    def run_epoch(self):

        #print("neural networks fitness run:")
        start = time.time()
        self.neural_networks.run()
        #print("neural network : ", time.time() - start)
        best = choose_best_tensor(
            self.neural_networks.neural_networks, self.neural_networks.accuracies)

        new_population = crossover(best,self.population, self.populationShape , self.populationSize, 0.01,2,len(self.layers))


        finish = new_population
        variable_summaries(self.population)
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

        for i in range(10):
            finished,accuracies, mergedSess = sess.run([tf.shape(finish),self.neural_networks.accuracies,merged], feed_dict={
                self.neural_networks.X: self.neural_networks.train_x, self.neural_networks.Y: self.neural_networks.train_y}, options=run_options, run_metadata=run_metadata)
            writer.add_summary(mergedSess,i) 
            print(accuracies)
            #print(finished)
            #print(finished)

        sess.close()
        writer.close()
