import tensorflow as tf
import numpy as np


def crossover(neural_networks, population_size):
    with tf.name_scope('Crossover') as scope:

        # new_population = neural_networks[:] ;
        new_population = [];
        size_neural_networks = len(neural_networks)
        # crossover_point = np.random.choice(np.arange(1, 784, step=1), population_size - size_neural_networks)

        father_tensor = neural_networks[0];
        mother_tensor = neural_networks[1];
        sess = tf.Session()

        #   sess = tf.Session();
        #    init = tf.global_variables_initializer();
        #    sess.run(init)
        #    inicializar_tensor_array(neural_networks,sess)
        #    father , mother = sess.run([father_tensor,mother_tensor]);
        #    sess.close()

        #    father_tensor = tf.Variable(father);
        #    mother_tensor = tf.Variable(mother);

        for i in range(population_size - size_neural_networks):
            # comparison = tf.equal(temp_neural_network , tf.constant(1))
            # conditional_assignment_op = temp_neural_network.assign(tf.where(comparison, tf.zeros_like(temp_neural_network), temp_neural_network))

            with tf.name_scope('Passagem_Genes') as scope:

                temp_neural_network = [];
                
                for weight_idx in range(len(mother_tensor)):
                    # child_tensor = tf.Variable(tf.zeros(tf.shape(mother_tensor[weight_idx])));
                    father_tensor_process = mother_tensor[weight_idx];
                    mother_tensor_process = father_tensor[weight_idx];
                    # mother_size = len(mother[weight_idx]) # tf.shape(mother[weight_idx]).run()
                    # random_array_start = np.random.randint(2, size=mother_size)
                    # random_array_start = tf.cast(tf.random_uniform(dtype=tf.int32,minval=0,maxval=1,shape=[shape_size[0]]),tf.float32)

                    shape_size = tf.shape(mother_tensor[weight_idx])
                    #                    random_array_binary = tf.constant(random_array_start,dtype=tf.float32);
                    random_array_binary = tf.cast(
                        tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=[shape_size[0]]), tf.float32)
                    random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)
                    # child_tensor = tf.mapfn(lambda x: x.assign(father_tensor[weight_idx][i] * ) )

                    # father_tensor_process = tf.multiply(father_tensor_process,random_array_start[:,tf.newaxis]);
                    # random_array_start = np.random.randint(2, size=mother_size)
                    random_array_start = tf.cast(
                        tf.random_uniform(dtype=tf.int32, minval=0, maxval=1, shape=[shape_size[0]]), tf.float32)

                    # mother_tensor_process = tf.multiply(mother_tensor_process,random_array_inverse[:,tf.newaxis]);
                    child_weight_tensor = tf.Variable(tf.multiply(father_tensor_process, random_array_start[:, tf.newaxis]) + tf.multiply( mother_tensor_process, random_array_inverse[:, tf.newaxis]));

                    temp_neural_network.append(child_weight_tensor[:]);
                    # child_weight = tf.Variable(child_tensor);
                    # temp_neural_network.append(child_weight);
            new_population.append(temp_neural_network[:]);
            # print(new_population);

        sess = tf.Session()
        writer = tf.summary.FileWriter("~/graph", sess.graph)
        #    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #    run_metadata = tf.RunMetadata()
        #    tf.reset_default_graph()
        #    inicializar_tensor_array(neural_networks,sess)

        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        #    init = tf.initialize_all_variables();
        #    inicializar_tensor_array(neural_networks,sess)

        sess.run(init_g)
        sess.run(init_l)
        #    sess.run(init)
        resultado_session = sess.run(new_population)
        sess.close()
        writer.close()

        pop = []
        for resultado in resultado_session:
            neural = []
            for peso in resultado:
                neural.append(tf.constant(peso))
            pop.append(neural)
        for neural in neural_networks:
            pop.append(neural[:])
        # pop.append(neural_networks[:])
        # print(resultado_session)
        # print(new_population)

        return pop;
