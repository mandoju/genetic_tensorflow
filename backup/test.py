import numpy as numpy
import tensorflow as tf

def test(x):
    if(x < 0.05):
        return int(1)
    else:
        return int(0)


random_array_binary =  tf.random_uniform(dtype=tf.float32, minval=0, maxval=1, shape=[10])
random_array_binary =  tf.map_fn(lambda x: tf.cond( x < 0.05 , lambda: 1.0 , lambda: 0.0 ), random_array_binary, dtype=tf.float32)

sess = tf.Session()
print(sess.run(random_array_binary) )

