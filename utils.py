def inicializar_tensor_array(tensors, session):
    for neural in tensors:
        for tensor in neural:
            session.run(tensor.initializer)
