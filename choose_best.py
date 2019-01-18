
import tensorflow as tf
import numpy as np


def choose_best(neural_networks, fitnesses):
    with tf.name_scope('Choose_best') as scope:

        print("fitness totais")
        print(fitnesses)
        top_2_idx = np.argsort(fitnesses)[-2:]
        print(top_2_idx)
        top_2_values = [neural_networks[i] for i in top_2_idx]
        print("fitness escolhidos")
        print([fitnesses[i] for i in top_2_idx])
        sess = tf.Session()
        sess.run(tf.global_variables_initializer());
        neural_networks_selected = [];
        neural_networks_selected.append(sess.run(top_2_values[0]));
        neural_networks_selected.append(sess.run(top_2_values[1]));

        print("fim choose best")
        #tf.reset_default_graph;


        return neural_networks_selected;
        #neural_networs_output = []
        #for neural_network in neural_networks_selected:
        #    temp_neural_network = []
        #    for weight in neural_network:
        #        temp_neural_network.append(tf.constant(weight))
        #    neural_networs_output.append(temp_neural_network[:])

        #return neural_networs_output;

def create_constants(neural_networks):
    neural_networs_output = []

    for current_neural_network in neural_networks:
        temp_neural_network = []
        #print("NEURAL NETWORK")
        i = 0
        for weight in current_neural_network:
            #if (type(weight) != tf.Tensor):
                #print(type(weight))
            temp_neural_network.append(tf.constant(weight))
            i += 1
        neural_networs_output.append(temp_neural_network[:])

    return neural_networs_output;


def choose_best_tensor(neural_networks, fitnesses):
    with tf.name_scope('Choose_best') as scope:

        top_values, top_indices = tf.math.top_k(tf.reshape(fitnesses, (-1,)), 2)
        #new_neural_networks = tf.gather()
        #top_2_idx = np.argsort(fitnesses)[-2:]
        #print("indices")
        #print(top_indices)
        #top_2_values = [neural_networks[i] for i in top_2_idx]
        #sess = tf.Session()
        #sess.run(tf.global_variables_initializer());
        #neural_networks_selected = sess.run(top_2_values);
        #print(top_2_values)

        #neural_networs_output = []
        #for neural_network in neural_networks_selected:
        #    temp_neural_network = []
        #    print("NEURAL NETWORK")
        #    for weight in neural_network:
        #        temp_neural_network.append(tf.constant(weight))
        #    neural_networs_output.append(temp_neural_network[:])

        neural_networks_output = tf.stack([neural_networks[top_indices[0]],neural_networks[top_indices[1]]])
        return neural_networks_output;


