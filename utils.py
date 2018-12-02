import tensorflow as tf
def inicializar_tensor_array(tensors, session):
    for neural in tensors:
        for tensor in neural:
            session.run(tensor.initializer)

def pad_up_to(t, max_in_dims, constant_values):
    s = tf.shape(t)
    paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
    return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)