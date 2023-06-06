import numpy as onp
import jax.numpy as np
from jax import random, grad, vmap, jit, jacfwd, jacrev
from jax.nn import relu
from jax import lax
import jax
from torch.utils.data import Dataset
from jax.flatten_util import ravel_pytree
import itertools
from functools import partial
import jax.debug as jdb


class DataGenerator(Dataset):
    def __init__(self, t0, t1, 
                 N_r=1000, 
                 alpha=10.,
                 beta=1.,
                 beta_lr=1e-4,
                 gate_type='relu_tanh',
                 tol=20.,
                 grad_clip=1e-1,
                 rng_key=random.PRNGKey(1234)):
        'Initialization'
        
        self.t0 = t0
        self.t1 = t1
        self.N_r = N_r
        self.alpha = alpha
        self.beta = beta
        self.gate_type = gate_type
        self.beta_lr = beta_lr
        self.tol = tol
        self.grad_clip = grad_clip
        self.key = rng_key
        
        self.key, subkey = random.split(self.key)
        
        subkeys = random.split(subkey, 2)
        self.t_r = random.uniform(subkeys[0], shape=(self.N_r,), minval=self.t0, maxval=self.t1)
        self.x_r = random.uniform(subkeys[1], shape=(self.N_r,), minval=0.0, maxval=2.0*np.pi)

    def __getitem__(self, index):
        'Generate one batch of data'
        batch = (self.t_r, self.x_r)
        beta = self.beta
        return batch, beta
    
    def set_batch(self, batch):
        self.t_r, self.x_r = batch
    
    def set_beta(self, beta):
        self.beta = beta
    
    @partial(jit, static_argnums=(0,))
    def update(self, x_r, t_r, loss, beta, key):
        if len(loss) != len(x_r) or len(loss) != len(t_r):
            raise RuntimeError("Input loss mismatches dimension with x, or t.")
        
        updated_batch = self.get_old_new(x_r=x_r, t_r=t_r, loss=loss, beta=beta, key=key)
        updated_beta = self.update_beta(t_r=t_r, loss=loss, beta=beta)
        return updated_batch, updated_beta
    
    
    @partial(jit, static_argnums=(0,))
    def get_old_new(self, x_r, t_r, loss, beta, key):
        fitness = loss * self.causal_gate(t_r=t_r, beta=beta)
        mask = (fitness > fitness.mean()).astype(int)
        
        #generate key
        subkeys = random.split(key, 2)
        x_new = random.uniform(subkeys[1], shape=(self.N_r,), minval=0.0, maxval=2.0*np.pi)
        t_new = random.uniform(subkeys[0], shape=(self.N_r,), minval=self.t0, maxval=self.t1)
        x_new = mask * x_r + (1-mask) * x_new
        t_new = mask * t_r + (1-mask) * t_new
        return (t_new, x_new)
    
    @partial(jit, static_argnums=(0,))
    def causal_gate(self, t_r, beta):
        t_norm = (t_r - self.t0) / (self.t1-self.t0)
        if self.gate_type == 'sigmoid':
            return 1 - sigmoid(self.alpha * (t_norm - beta))
        elif self.gate_type == 'tanh':
            return (1 - np.tanh(self.alpha * (t_norm - beta)))/2
        elif self.gate_type == 'relu_tanh':
            return relu( -np.tanh(self.alpha * (t_norm - beta)) )
    
    @partial(jit, static_argnums=(0,))
    def update_beta(self, t_r, loss, beta):
        fitness = (loss * self.causal_gate(t_r=t_r, beta=beta))**2
        gradient = np.exp(-self.tol * fitness.mean())
        gradient = np.minimum(gradient, self.grad_clip)
        updated_beta = beta + self.beta_lr * gradient
        return updated_beta