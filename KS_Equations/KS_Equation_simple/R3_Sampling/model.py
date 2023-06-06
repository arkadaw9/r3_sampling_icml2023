import jax.numpy as np
from jax import random, grad, vmap, jit, jacfwd, jacrev
from jax.nn import relu
from jax import lax

# Define MLP
def MLP(layers, L=1.0, M=1, activation=relu):
  # Define input encoding function
    def input_encoding(t, x):
        w = 2.0 * np.pi / L
        k = np.arange(1, M + 1)
        out = np.hstack([t, 1, 
                         np.cos(k * w * x), np.sin(k * w * x)])
        return out
   
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
    def apply(params, inputs):
        t = inputs[0]
        x = inputs[1]
        H = input_encoding(t, x)
        for W, b in params[:-1]:
            outputs = np.dot(H, W) + b
            H = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(H, W) + b
        return outputs
    return init, apply


# Define modified MLP
def modified_MLP(layers, L=1.0, M=1, activation=relu, init_type="xavier_init"):
    def xavier_init(key, d_in, d_out):
        glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b
    
    def pytorch_init(key, d_in, d_out):
        k = np.sqrt(1. / d_in)
        W = random.uniform(key, (d_in, d_out), minval=-k, maxval=k)
        b = random.uniform(key, (d_out), minval=-k, maxval=k)
        return W, b

    # Define input encoding function
    def input_encoding(t, x):
        w = 2 * np.pi / L
        k = np.arange(1, M + 1)
        out = np.hstack([t, 1, 
                         np.cos(k * w * x), np.sin(k * w * x)])
        return out


    def init(rng_key):
        if init_type=="xavier_init":
            init_fn = xavier_init
        elif init_type=="pytorch_init":
            init_fn = pytorch_init
        
        U1, b1 =  init_fn(random.PRNGKey(12345), layers[0], layers[1])
        U2, b2 =  init_fn(random.PRNGKey(54321), layers[0], layers[1])
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = init_fn(k1, d_in, d_out)
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return (params, U1, b1, U2, b2) 

    def apply(params, inputs):
        params, U1, b1, U2, b2 = params

        t = inputs[0]
        x = inputs[1]
        inputs = input_encoding(t, x)  
        U = activation(np.dot(inputs, U1) + b1)
        V = activation(np.dot(inputs, U2) + b2)
        for W, b in params[:-1]:
            outputs = activation(np.dot(inputs, W) + b)
            inputs = np.multiply(outputs, U) + np.multiply(1 - outputs, V) 
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs
    return init, apply     