import tensorflow as tf
import numpy as np
from mutation import mutation

def crossover(neural_networks,population,populationShape,population_size,mutationRate,tournamentSize,layers):
    with tf.name_scope('Crossover'):

        size_neural_networks = tournamentSize 

        #population = tf.get_variable('populacao',populationShape)
        father_tensor = neural_networks[0]
        mother_tensor = neural_networks[1]

        new_population = []
        new_population.append(father_tensor)
        new_population.append(mother_tensor)


        
        for i in range(population_size - size_neural_networks):

            with tf.name_scope('Passagem_Genes'):

                temp_neural_network = []
                
                for weight_idx_range in range(layers - 1):
                    weight_idx = weight_idx_range - 1
                    father_tensor_process = mother_tensor[weight_idx]
                    mother_tensor_process = father_tensor[weight_idx]

                    shape_size = tf.shape(mother_tensor[weight_idx])


                    #Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
                    random_array_binary = tf.cast(
                        tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=[shape_size[0]]), tf.float32)
                        

                    #Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
                    random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)

                    #Criação o array de taxa de mistura para ambos
                    random_array_start = tf.cast(
                        tf.random_uniform(dtype=tf.int32, minval=0, maxval=1, shape=[shape_size[0]]), tf.float32)

                    #Fazendo o crossover do pai + mãe
                    #child_weight_tensor = tf.Variable(tf.multiply(father_tensor_process, random_array_start[:, tf.newaxis]) + tf.multiply( mother_tensor_process, random_array_inverse[:, tf.newaxis]))

                    #mutation(child_weight_tensor,mutationRate)
                    temp_neural_network.append(tf.multiply(father_tensor_process, random_array_start[:, tf.newaxis]) + tf.multiply( mother_tensor_process, random_array_inverse[:, tf.newaxis]))
                    print(temp_neural_network)
            #population = population[i + tournamentSize].assign( temp_neural_network)
            new_population.append(temp_neural_network[:])

        finish = tf.assign(tf.get_variable('populacao',populationShape), tf.stack(new_population))
        return finish
        #sess = tf.Session()
        # writer = tf.summary.FileWriter("log/graph", sess.graph)

        # init_g = tf.global_variables_initializer()
        # init_l = tf.local_variables_initializer()

        # sess.run(init_g)
        # sess.run(init_l)

        # resultado_session = sess.run(new_population)
        # sess.close()
        # writer.close()

        #        pop = []
        #        for resultado in resultado_session:
        #            neural = []
        #            for peso in resultado:
        #                neural.append(tf.constant(peso))
        #            pop.append(neural)
        #        for neural in neural_networks:
        #            pop.append(neural[:])