
import tensorflow as tf
import numpy as np
import copy



# def choose_best(neural_networks, fitnesses):
#     with tf.name_scope('Choose_best') as scope:

#         # Choosing top 2 fitnesses
#         #top_2_idx = np.argsort(fitnesses)[-2:]
#         # Choosing bottom 2 fitnesses
#         top_2_idx = np.argsort(fitnesses)[2:]

#         # Getting top 2 neural networks
#         top_2_values = [neural_networks[i] for i in top_2_idx]
#         neural_networks_selected = []
#         neural_networks_selected.append(top_2_values[0])
#         neural_networks_selected.append(top_2_values[1])

#         # tf.reset_default_graph;

#         print("selecionado")
#         print(neural_networks_selected)
#         return neural_networks_selected
#         #neural_networs_output = []
#         # for neural_network in neural_networks_selected:
#         #    temp_neural_network = []
#         #    for weight in neural_network:
#         #        temp_neural_network.append(tf.constant(weight))
#         #    neural_networs_output.append(temp_neural_network[:])

#         # return neural_networs_output;


def create_constants(neural_networks):
    neural_networs_output = []

    for current_neural_network in neural_networks:
        temp_neural_network = []
        #print("NEURAL NETWORK")
        i = 0
        for weight in current_neural_network:
            # if (type(weight) != tf.Tensor):
                # print(type(weight))
            temp_neural_network.append(tf.constant(weight))
            i += 1
        neural_networs_output.append(temp_neural_network[:])

    return neural_networs_output


def choose_best_tensor(neural_networks, fitnesses):
    with tf.name_scope('Choose_best') as scope:

        top_values, top_indices = tf.math.top_k(
            tf.reshape(fitnesses, (-1,)), 4)
        #new_neural_networks = tf.gather()
        #top_2_idx = np.argsort(fitnesses)[-2:]
        # print("indices")
        # print(top_indices)
        #top_2_values = [neural_networks[i] for i in top_2_idx]
        #sess = tf.Session()
        # sess.run(tf.global_variables_initializer());
        #neural_networks_selected = sess.run(top_2_values);
        # print(top_2_values)

        #neural_networs_output = []
        # for neural_network in neural_networks_selected:
        #    temp_neural_network = []
        #    print("NEURAL NETWORK")
        #    for weight in neural_network:
        #        temp_neural_network.append(tf.constant(weight))
        #    neural_networs_output.append(temp_neural_network[:])

        neural_networks_output = tf.stack([neural_networks[top_indices[0]], neural_networks[top_indices[1]],
                                           neural_networks[top_indices[2]], neural_networks[top_indices[3]]])
        return neural_networks_output


def choose_best_tensor_conv(convulations, biases, fitnesses, chooseNumber):
    with tf.name_scope('Choose_best') as scope:

        print(chooseNumber)
        top_values, top_indices = tf.math.top_k(
            fitnesses, chooseNumber)
            #tf.reshape(fitnesses, (-1,)), 4)
        
        print('-----')
        print(top_indices)
        top_mutate_values, top_mutate_indices = tf.math.top_k(
            fitnesses, chooseNumber*8)
        #new_neural_networks = tf.gather()
        #top_2_idx = np.argsort(fitnesses)[-2:]
        # print("indices")
        # print(top_indices)
        #top_2_values = [neural_networks[i] for i in top_2_idx]
        #sess = tf.Session()
        # sess.run(tf.global_variables_initializer());
        #neural_networks_selected = sess.run(top_2_values);
        # print(top_2_values)

        #neural_networs_output = []
        # for neural_network in neural_networks_selected:
        #    temp_neural_network = []
        #    print("NEURAL NETWORK")
        #    for weight in neural_network:
        #        temp_neural_network.append(tf.constant(weight))
        #    neural_networs_output.append(temp_neural_network[:])

        convulations_weights_keys = list(convulations.keys())
        convulation_weights_output = {
            key: None for key in convulations_weights_keys}
        biases_output_keys = list(biases.keys())
        biases_output = {key: None for key in biases_output_keys}

        convulation_weights_best_output = {
            key: None for key in convulations_weights_keys}
        biases_output_best = {key: None for key in biases_output_keys}

        convulation_weights_mutate_output = {
            key: None for key in convulations_weights_keys}
        biases_output_mutate = {key: None for key in biases_output_keys}
        
        for key in convulations:
            #array_to_stack = tf.map_fn(lambda x: convulations[key][top_indices[x]], tf.range(chooseNumber))
            convulation_weights_output[key] = tf.gather(convulations[key],top_indices) #tf.slice(convulation_weights_best_output[key],conv_shape_zero, conv_shape_altered )
            convulation_weights_mutate_output[key] = tf.gather(convulations[key],top_mutate_indices) #tf.slice(convulation_weights_best_output[key],conv_shape_zero, conv_shape_altered )

            # tf.stack(
            #     [convulations[key][top_indices[0]], convulations[key][top_indices[1]], convulations[key][top_indices[2]], convulations[key][top_indices[3]]])
            convulation_weights_best_output[key] = convulations[key][top_indices[0]]
        for key in biases:
            # print(idx);
            biases_output[key] = tf.gather(biases[key], top_indices) #[top_indices[0]:top_indices[chooseNumber - 1]]
            biases_output_mutate[key] = tf.gather(biases[key], top_mutate_indices) #[top_indices[0]:top_indices[chooseNumber - 1]]

            # tf.stack(
            #     [biases[key][top_indices[0]], biases[key][top_indices[1]], biases[key][top_indices[2]], biases[key][top_indices[3]]])
            biases_output_best[key] = biases[key][top_indices[0]]

        #neural_networks_output = tf.stack([neural_networks[top_indices[0]],neural_networks[top_indices[1]]])
        # return neural_networks_output;
        print(top_indices)
        return convulation_weights_output, biases_output, convulation_weights_best_output, biases_output_best, convulation_weights_mutate_output, biases_output_mutate


def tournament(fitnesses, indexes):
    #convulation_weights_output[key] = tf.gather(convulations[key],top_indices)
    tournament_fitnesses = tf.gather(fitnesses, indexes)
    get_best_value, get_best_index = tf.math.top_k(tournament_fitnesses)
    return indexes[get_best_index[0]]

def choose_best_tensor_tournament(convulations, biases, fitnesses, chooseNumber):
    with tf.name_scope('Choose_best') as scope:


        tournamentSize = tf.shape(fitnesses)[0] // chooseNumber

        numbers_to_tournament = tf.range(tf.shape(fitnesses)[0])
        numbers_to_tournament = tf.random.shuffle(numbers_to_tournament)
        numbers_to_tournament = tf.reshape(numbers_to_tournament,[tf.shape(numbers_to_tournament)[0]//tournamentSize, tournamentSize])
        #top_indices = tf.map_fn( lambda x: get_tournament_result(numbers_to_tournament) , tf.range(numbers_to_tournament // tournamentSize ))


        top_indices =  tf.map_fn(lambda x: tournament(fitnesses,x) , numbers_to_tournament) 
        #top_indices = tf.reshape(top_indices, [tf.shape(top_indices)[0]])
        top_mutate_indices = tf.tile(top_indices, [8])
        # top_values, top_indices = tf.math.top_k(
        #     fitnesses, chooseNumber)
        # #tf.reshape(fitnesses, (-1,)), 4)
        
        # top_mutate_values, top_mutate_indices = tf.math.top_k(
        #     fitnesses, chooseNumber*8)
        #new_neural_networks = tf.gather()
        #top_2_idx = np.argsort(fitnesses)[-2:]
        # print("indices")
        # print(top_indices)
        #top_2_values = [neural_networks[i] for i in top_2_idx]
        #sess = tf.Session()
        # sess.run(tf.global_variables_initializer());
        #neural_networks_selected = sess.run(top_2_values);
        # print(top_2_values)

        #neural_networs_output = []
        # for neural_network in neural_networks_selected:
        #    temp_neural_network = []
        #    print("NEURAL NETWORK")
        #    for weight in neural_network:
        #        temp_neural_network.append(tf.constant(weight))
        #    neural_networs_output.append(temp_neural_network[:])

        convulations_weights_keys = list(convulations.keys())
        convulation_weights_output = {
            key: None for key in convulations_weights_keys}
        biases_output_keys = list(biases.keys())
        biases_output = {key: None for key in biases_output_keys}

        convulation_weights_best_output = {
            key: None for key in convulations_weights_keys}
        biases_output_best = {key: None for key in biases_output_keys}

        convulation_weights_mutate_output = {
            key: None for key in convulations_weights_keys}
        biases_output_mutate = {key: None for key in biases_output_keys}
        
        for key in convulations:
            #array_to_stack = tf.map_fn(lambda x: convulations[key][top_indices[x]], tf.range(chooseNumber))
            convulation_weights_output[key] = tf.gather(convulations[key],top_indices) #tf.slice(convulation_weights_best_output[key],conv_shape_zero, conv_shape_altered )
            convulation_weights_mutate_output[key] = tf.gather(convulations[key],top_mutate_indices) #tf.slice(convulation_weights_best_output[key],conv_shape_zero, conv_shape_altered )

            # tf.stack(
            #     [convulations[key][top_indices[0]], convulations[key][top_indices[1]], convulations[key][top_indices[2]], convulations[key][top_indices[3]]])
            convulation_weights_best_output[key] = convulations[key][top_indices[0]]
        for key in biases:
            # print(idx);
            biases_output[key] = tf.gather(biases[key], top_indices) #[top_indices[0]:top_indices[chooseNumber - 1]]
            biases_output_mutate[key] = tf.gather(biases[key], top_mutate_indices) #[top_indices[0]:top_indices[chooseNumber - 1]]

            # tf.stack(
            #     [biases[key][top_indices[0]], biases[key][top_indices[1]], biases[key][top_indices[2]], biases[key][top_indices[3]]])
            biases_output_best[key] = biases[key][top_indices[0]]

        #neural_networks_output = tf.stack([neural_networks[top_indices[0]],neural_networks[top_indices[1]]])
        # return neural_networks_output;
        return convulation_weights_output, biases_output, convulation_weights_best_output, biases_output_best, convulation_weights_mutate_output, biases_output_mutate


def choose_best(chooseType,convulations, biases, fitnesses, chooseNumber):
    if(chooseType == 'tournament'):
        return choose_best_tensor_tournament(convulations, biases, fitnesses, chooseNumber)
    else:
        return choose_best_tensor_conv(convulations, biases, fitnesses, chooseNumber)