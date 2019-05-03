import tensorflow as tf
import numpy as np 
from mutation import mutation
from genetic_operators import generate_child_by_all
pai = tf.get_variable('w1', shape= (3,3,1,32), initializer=tf.random_normal_initializer(seed=1))
mae = tf.get_variable('w2', shape= (3,3,1,32), initializer=tf.random_normal_initializer(seed=2))

crossover = generate_child_by_all(pai,mae)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saida_pai,saida_mae = sess.run([pai,mae])

print("pai")
print(saida_pai)
print("mãe")
print(saida_mae)

filho = sess.run(crossover)
print (filho)

print(np.mean(saida_pai))
print(np.mean(saida_mae))
print(np.mean(filho))