
import tensorflow as tf
import numpy as np

def function_map(xInput,mutationRate):
    return tf.map_fn(lambda x: tf.cond( x < mutationRate , lambda: 1.0 , lambda: 0.0 ), xInput, dtype=tf.float32)

def mutation(tensor, mutationRate):
    # depois fazer matrix mascara (a.k.a recomendacao do gabriel)
    with tf.name_scope('Mutation'):


        shapeSize = tf.shape(tensor)
        random_array_binary =  tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=shapeSize)

        random_array_binary =  tf.map_fn(lambda x: function_map(x,mutationRate), random_array_binary, dtype=tf.float32)

        random_array_values =  tf.random_uniform(dtype=tf.float32, minval=-1, maxval=1, shape=shapeSize)

        random_mutation = tf.multiply(random_array_binary,random_array_values)
        
        mutated = tensor + random_mutation
        return mutated;