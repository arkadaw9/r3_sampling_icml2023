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
                 rng_key=random.PRNGKey(1234)):
        'Initialization'
        
        self.t0 = t0
        self.t1 = t1
        self.N_r = N_r
        self.key = rng_key
        
        self.key, subkey = random.split(self.key)
        
        subkeys = random.split(subkey, 2)
        self.t_r = random.uniform(subkeys[0], shape=(self.N_r,), minval=self.t0, maxval=self.t1)
        self.x_r = random.uniform(subkeys[1], shape=(self.N_r,), minval=0.0, maxval=2.0*np.pi)

    def __getitem__(self, index):
        'Generate one batch of data'
        batch = (self.t_r, self.x_r)
        return batch
    
    def set_batch(self, batch):
        self.t_r, self.x_r = batch
    
    @partial(jit, static_argnums=(0,))
    def update(self, x_r, t_r, loss, key):
        if len(loss) != len(x_r) or len(loss) != len(t_r):
            raise RuntimeError("Input loss mismatches dimension with x, or t.")
        
        updated_batch = self.get_old_new(x_r=x_r, t_r=t_r, loss=loss, key=key)
        return updated_batch
    
    
    @partial(jit, static_argnums=(0,))
    def get_old_new(self, x_r, t_r, loss, key):
        fitness = loss
        mask = (fitness > fitness.mean()).astype(int)
        
        #generate key
        subkeys = random.split(key, 2)
        x_new = random.uniform(subkeys[1], shape=(self.N_r,), minval=0.0, maxval=2.0*np.pi)
        t_new = random.uniform(subkeys[0], shape=(self.N_r,), minval=self.t0, maxval=self.t1)
        x_new = mask * x_r + (1-mask) * x_new
        t_new = mask * t_r + (1-mask) * t_new
        return (t_new, x_new)