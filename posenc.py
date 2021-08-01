import numpy as np
import tensorflow as tf

def positional_embedding(pos, model_size):
    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return PE

def get_standard_posenc(length,dim):
    if type(length) == list:
        pencs = [get_standard_posenc(l,dim//len(length)) for l in length]
        
        ranges = [tf.range(0,l,dtype=tf.int64) for l in length]
        pr = zip(pencs,ranges)
        results = [tf.gather(data[0],data[1]) for data in pr]
        return results
    pes = []
    for i in range(length):
        pes.append(positional_embedding(float(i), dim))

    pes = np.concatenate(pes, axis=0)
    pes = tf.constant(pes, dtype=tf.float32)
    return pes

def get_zero_to_one_posenc(length,dim):
    if type(length) == list:
        pencs = [get_zero_to_one_posenc(l,dim//len(length)) for l in length]
        
        ranges = [tf.range(0,l,dtype=tf.int64) for l in length]
        pr = zip(pencs,ranges)
        results = [tf.gather(data[0],data[1]) for data in pr]
        return results
    indices = np.linspace(start=-50.0,stop=50.0, num=length, endpoint=True)
    pes = []
    for i in range(length):
        pes.append(positional_embedding(indices[i], dim))

    pes = np.concatenate(pes, axis=0)
    pes = tf.constant(pes, dtype=tf.float32)
    return pes
