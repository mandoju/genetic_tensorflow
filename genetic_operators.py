import tensorflow as tf
import numpy as np
from mutation import mutation, mutation_unbiased, mutation_by_node
from itertools import chain
from collections import defaultdict

def select_operator_and_apply(genetic_operator,genetic_operator_param,genetic_operator_population_size,elite_size,mutatioRate,best_conv,best_bias):
    if(genetic_operator == 'crossover'): 
        return crossover_operator(best_conv,best_bias,elite_size,genetic_operator_population_size)
    elif(genetic_operator == 'mutation'):
        return mutation_operator(best_conv,best_bias,elite_size,mutatioRate,genetic_operator_param,genetic_operator_population_size)
    elif(genetic_operator == 'mutate_by_layer'):
        return mutation_operator_by_layer(best_conv,best_bias,elite_size,mutatioRate,genetic_operator_param,genetic_operator_population_size)
    elif(genetic_operator == 'mutation_unbiased'):
        return mutation_unbiased_operator(best_conv,best_bias,elite_size,mutatioRate,genetic_operator_param,genetic_operator_population_size)
def apply_genetic_operatos(genetic_operators, genetic_operators_size, elite_size, input_convulations, input_bias, best_convulations, best_biases, populationShape , populationSize, mutationRate,tournamentSize,layers):
    
    conv_operators_results = []
    bias_operators_results = []
    conv_result_dict = {}
    bias_result_dict = {}
    assigns_conv = []
    assigns_bias = []

    
    conv_operators_results, bias_operators_results = zip(*[select_operator_and_apply(genetic_operator[0],genetic_operator[1],tf.cast(genetic_operators_size[idx] ,dtype=tf.int32),elite_size,mutationRate,best_convulations,best_biases) for idx, genetic_operator in enumerate(genetic_operators)])

    conv_operators_results = list(conv_operators_results)
    bias_operators_results = list(bias_operators_results)

    conv_operators_results.append(best_convulations)
    bias_operators_results.append(best_biases)
    for item in conv_operators_results:
        for k, v in item.items():
            if(k in conv_result_dict):
                    conv_result_dict[k].append(v)
            else:
                    conv_result_dict[k] = [v]

    for item in bias_operators_results:
        for k, v in item.items():
            if(k in bias_result_dict):
                    bias_result_dict[k].append(v)
            else:
                    bias_result_dict[k] = [v]

    for key, value in conv_result_dict.items():
        assigns_conv.append(input_convulations[key].assign(tf.concat(value,0) ) )
    
    for key, value in bias_result_dict.items():
        assigns_bias.append(input_bias[key].assign(tf.concat(value,0) ) )
    
    return assigns_conv, assigns_bias

##Todos aleatorios
def generate_child_by_all(mother_tensor,father_tensor):
    with tf.name_scope('Passagem_Genes'):

        temp_neural_network = []
        

        shape_size = tf.shape(mother_tensor)

        #Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
        random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=shape_size)
        random_array_select = tf.random_uniform(dtype=tf.float32, minval=0,maxval=1,shape=shape_size)
        random_array_select = tf.math.round(random_array_select)

        #Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
        #random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)
        random_array_inverse = tf.scalar_mul(-1, random_array_binary) + tf.ones_like(random_array_binary)

        crossoved = tf.multiply(father_tensor, random_array_binary) + tf.multiply( mother_tensor, random_array_inverse)
        #temp_neural_network.append(mutation(crossoved,mutationRate))
        #Criação o array de taxa de mistura para ambos
        #random_array_start = tf.cast(
        #    tf.random_uniform(dtype=tf.int32, minval=0, maxval=1, shape=[shape_size[0]]), tf.float32)

        # for weight_idx_range in range(layers - 1):
        #     weight_idx = weight_idx_range - 1
        #     father_tensor_process = mother_tensor[weight_idx]
        #     mother_tensor_process = father_tensor[weight_idx]

        #     shape_size = tf.shape(mother_tensor[weight_idx])

        #     # #Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
        #     # random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=[shape_size[0]])
                

        #     # #Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
        #     # random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)

        #     # #Criação o array de taxa de mistura para ambos
        #     # random_array_start = tf.cast(
        #     #     tf.random_uniform(dtype=tf.int32, minval=0, maxval=1, shape=[shape_size[0]]), tf.float32)


        #     #Fazendo o crossover do pai + mãe
        #     #child_weight_tensor = tf.Variable(tf.multiply(father_tensor_process, random_array_start[:, tf.newaxis]) + tf.multiply( mother_tensor_process, random_array_inverse[:, tf.newaxis]))

        #     #mutation(child_weight_tensor,mutationRate)
        #     crossoved = tf.multiply(father_tensor_process, random_array_binary[weight_idx]) + tf.multiply( mother_tensor_process, random_array_inverse[weight_idx])
        #     temp_neural_network.append(mutation(crossoved,mutationRate))

        return crossoved

def generate_child_by_mixed(mother_tensor,father_tensor,mutationRate):
    with tf.name_scope('Passagem_Genes'):

        temp_neural_network = []
        

        shape_size = tf.shape(mother_tensor)

        #Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
        random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=shape_size)
        
        random_array_select =  tf.random_uniform(dtype=tf.float32, minval=0,maxval=1,shape=[shape_size[0]])
        random_array_select = tf.math.round(random_array_select)

        random_array_binary = tf.multiply(random_array_select[:,tf.newaxis],random_array_binary)
        #Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
        #random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)
        random_array_inverse = tf.scalar_mul(-1, random_array_binary) + tf.ones_like(random_array_binary)

        crossoved = tf.multiply(father_tensor, random_array_binary) + tf.multiply( mother_tensor, random_array_inverse)
        #temp_neural_network.append(mutation(crossoved,mutationRate))
        #Criação o array de taxa de mistura para ambos
        #random_array_start = tf.cast(
        #    tf.random_uniform(dtype=tf.int32, minval=0, maxval=1, shape=[shape_size[0]]), tf.float32)

        

        # for weight_idx_range in range(layers - 1):
        #     weight_idx = weight_idx_range - 1
        #     father_tensor_process = mother_tensor[weight_idx]
        #     mother_tensor_process = father_tensor[weight_idx]

        #     shape_size = tf.shape(mother_tensor[weight_idx])

        #     # #Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
        #     # random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=[shape_size[0]])
                

        #     # #Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
        #     # random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)

        #     # #Criação o array de taxa de mistura para ambos
        #     # random_array_start = tf.cast(
        #     #     tf.random_uniform(dtype=tf.int32, minval=0, maxval=1, shape=[shape_size[0]]), tf.float32)


        #     #Fazendo o crossover do pai + mãe
        #     #child_weight_tensor = tf.Variable(tf.multiply(father_tensor_process, random_array_start[:, tf.newaxis]) + tf.multiply( mother_tensor_process, random_array_inverse[:, tf.newaxis]))

        #     #mutation(child_weight_tensor,mutationRate)
        #     crossoved = tf.multiply(father_tensor_process, random_array_binary[weight_idx]) + tf.multiply( mother_tensor_process, random_array_inverse[weight_idx])
        #     temp_neural_network.append(mutation(crossoved,mutationRate))

        return crossoved

##Apenas as layers
def generate_child_by_layer(mother_tensor,father_tensor,mutationRate,layers):
    with tf.name_scope('Passagem_Genes'):

        temp_neural_network = []
        

        shape_size = tf.shape(mother_tensor)

        #Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
        random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=[shape_size[0]])
        random_array_select = tf.random_uniform(dtype=tf.float32, minval=0,maxval=1,shape=[shape_size[0]])
        random_array_select = tf.math.round(random_array_select)

        #Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
        random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)

        #Criação o array de taxa de mistura para ambos
        #random_array_start = tf.cast(
        #    tf.random_uniform(dtype=tf.int32, minval=0, maxval=1, shape=[shape_size[0]]), tf.float32)

        

        for weight_idx_range in range(layers - 1):
            weight_idx = weight_idx_range - 1
            father_tensor_process = mother_tensor[weight_idx]
            mother_tensor_process = father_tensor[weight_idx]

            shape_size = tf.shape(mother_tensor[weight_idx])

            # #Criação do array binário para definir quais são os genes que irão receber a mistura do  mãe
            # random_array_binary = tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=[shape_size[0]])
                

            # #Criando o array inverso para definir o número ao contrário para criar a quantidade recebida pelo pai
            # random_array_inverse = tf.map_fn(lambda x: (1 - x), random_array_binary, dtype=tf.float32)

            # #Criação o array de taxa de mistura para ambos
            # random_array_start = tf.cast(
            #     tf.random_uniform(dtype=tf.int32, minval=0, maxval=1, shape=[shape_size[0]]), tf.float32)


            #Fazendo o crossover do pai + mãe
            #child_weight_tensor = tf.Variable(tf.multiply(father_tensor_process, random_array_start[:, tf.newaxis]) + tf.multiply( mother_tensor_process, random_array_inverse[:, tf.newaxis]))

            #mutation(child_weight_tensor,mutationRate)
            crossoved = tf.multiply(father_tensor_process, random_array_binary[weight_idx]) + tf.multiply( mother_tensor_process, random_array_inverse[weight_idx])
            temp_neural_network.append(mutation(crossoved,mutationRate))

        return tf.stack(temp_neural_network)

def crossover_operator(best_conv, best_bias, tamanhoElite, tamanhoCrossover):

    with tf.name_scope('Crossover'):

        #size_neural_networks = tournamentSize 
        size_neural_networks = 3

        finish = []
        finish_conv = {}
        finish_bias = {}
#        tamanhoCrossover = tamanhoElite
        permutations = tf.concat( [ tf.range(tamanhoElite) , tf.range(tamanhoElite) ], 0 )
        permutations = tf.reshape(permutations, [tamanhoElite,2])
        keys = best_conv.keys()


        for key in best_conv: 
                #population = best_conv[key] #, best_conv[key][2], best_conv[key][3] ])
                #new_population = 
                finish_conv[key] = tf.map_fn(lambda permutation: generate_child_by_all(best_conv[key][permutation[0]],best_conv[key][permutation[1]]) ,permutations[0:(tamanhoCrossover)], dtype=tf.float32)

        for key in best_bias: 
                #population = best_bias[key] #, best_bias[key][2] ,best_bias[key][3] ])
                #new_population = 
                finish_bias[key] = tf.map_fn(lambda permutation: generate_child_by_all(best_bias[key][permutation[0]],best_bias[key][permutation[1]]) ,permutations[0:(tamanhoCrossover)], dtype=tf.float32)

        return finish_conv, finish_bias

def mutation_operator(best_conv,best_bias,tamanhoElite,mutationRate,mutationPercent,tamanhoMutacoes):

        with tf.name_scope('Crossover'):
                
            finish = []
            finish_conv = {}
            finish_bias = {}
            tamanhoCrossover = tamanhoElite
            permutations = tf.range(tamanhoElite)
            permutations = tf.reshape(permutations, [tamanhoElite//2,2])
            keys = best_conv.keys()
            
            for key in best_conv: 
                    shape_module = tf.shape(best_conv[key])[0]
                    finish_conv[key] = tf.map_fn(lambda x: mutation_by_node(best_conv[key][x%shape_module],mutationRate,mutationPercent),tf.range( tamanhoMutacoes), dtype=tf.float32)

            for key in best_bias: 
                    shape_module = tf.shape(best_bias[key])[0]
                    finish_bias[key] = tf.map_fn(lambda x: mutation(best_bias[key][x%shape_module],mutationRate,mutationPercent),tf.range( tamanhoMutacoes), dtype=tf.float32)

        return finish_conv, finish_bias

def mutation_operator_by_layer(best_conv,best_bias,tamanhoElite,mutationRate,mutationPercent,tamanhoMutacoes):

        with tf.name_scope('Crossover'):
                
            finish = []
            finish_conv = {}
            finish_bias = {}
            tamanhoCrossover = tamanhoElite
            permutations = tf.range(tamanhoElite)
            permutations = tf.reshape(permutations, [tamanhoElite//2,2])
            keys = best_conv.keys()
            
            for key in best_conv: 
                    shape_module = tf.shape(best_conv[key])[0]
                    finish_conv[key] = tf.map_fn(lambda x: mutation_by_node(best_conv[key][x%shape_module],mutationRate,mutationPercent),tf.range( tamanhoMutacoes), dtype=tf.float32)

            for key in best_bias: 
                    shape_module = tf.shape(best_bias[key])[0]
                    finish_bias[key] = tf.map_fn(lambda x: mutation_by_node(best_bias[key][x%shape_module],mutationRate,mutationPercent),tf.range( tamanhoMutacoes), dtype=tf.float32)

        return finish_conv, finish_bias

def mutation_unbiased_operator(best_conv,best_bias,tamanhoElite,mutationRate,mutationPercent,tamanhoMutacoes):

        with tf.name_scope('Crossover'):
                
            finish = []
            finish_conv = {}
            finish_bias = {}
            tamanhoCrossover = tamanhoElite
            permutations = tf.range(tamanhoElite)
            permutations = tf.reshape(permutations, [tamanhoElite//2,2])
            keys = best_conv.keys()

            for key in best_conv:
                    shape_module = tf.shape(best_conv[key])[0]
                    finish_conv[key] = tf.map_fn(lambda x: mutation_unbiased(best_conv[key][x%shape_module],mutationRate,mutationPercent),tf.range( tamanhoMutacoes), dtype=tf.float32)

            for key in best_bias: 
                    shape_module = tf.shape(best_bias[key])[0]
                    finish_bias[key] = tf.map_fn(lambda x: mutation_unbiased(best_bias[key][x%shape_module],mutationRate,mutationPercent),tf.range( tamanhoMutacoes), dtype=tf.float32)

        return finish_conv, finish_bias
