import numpy as onp
import jax.numpy as np
import jax
# from jax import debug
from jax import random, grad, vmap, jit, jacfwd, jacrev
# from jax.experimental import optimizers #perhaps was there in older version
from jax.example_libraries import optimizers
from jax.experimental.ode import odeint
from jax.experimental.jet import jet
from jax.nn import relu
from jax.config import config
from jax import lax
from jax.flatten_util import ravel_pytree
import itertools
from functools import partial
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from model import *
import matplotlib.pyplot as plt
import os

def StepLR_Scheduler(init_lr, decay_steps, decay_rate):
    def schedule(i):
        return init_lr * decay_rate ** (i // decay_steps)
    return schedule

# Define the model
class PINN:
    def __init__(self, key, arch, layers, M_t, M_x, state0, t0, t1, dataset, x_star, args_dict): 
        
        # IC
        t_ic = np.zeros((x_star.shape[0], 1))
        x_ic = x_star.reshape(-1, 1)
        self.X_ic = np.hstack([t_ic, x_ic])
        self.Y_ic = state0
        
        self.dataset = dataset
        self.args_dict = args_dict
        self.M_x = M_x
        self.M_t = M_t
        d0 = 2 * M_x + M_t + 1
        layers = [d0] + layers
        self.init, self.apply = modified_MLP(layers, 
                                             L=2.0*np.pi, 
                                             M_t=self.M_t, 
                                             M_x=self.M_x, 
                                             activation=np.tanh,
                                             init_type=self.args_dict['init_type']
                                            )
        params = self.init(rng_key = key)
        
        # Use optimizers to set optimizer initialization and update functions
        if self.args_dict['scheduler_type'] == "exponential":
            lr = optimizers.exponential_decay(self.args_dict['lr'], 
                                              decay_steps=self.args_dict['decay_steps'], 
                                              decay_rate=self.args_dict['decay_rate'])
        elif self.args_dict['scheduler_type'] == "step_lr":
            lr = StepLR_Scheduler(self.args_dict['lr'], 
                                  decay_steps=self.args_dict['decay_steps'], 
                                  decay_rate=self.args_dict['decay_rate'])
            
        self.opt_init,  self.opt_update, self.get_params = optimizers.adam(lr)
        self.opt_state = self.opt_init(params) 
        _, self.unravel = ravel_pytree(params)
        
        # Evaluate functions over a grid
        self.u_pred_fn = vmap(vmap(self.neural_net, (None, 0, None)), (None, None, 0))  # consistent with the dataset
        self.r_pred_fn = vmap(vmap(self.residual_net, (None, 0, None)), (None, None, 0))

        # Logger
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_res_log = []
        self.l2_error_log = []
        
        self.itercount = itertools.count()
        
    def neural_net(self, params, t, x):
        z = np.stack([t, x])
        outputs = self.apply(params, z)
        return outputs[0]

    def residual_net(self, params, t, x): 
        u = self.neural_net(params, t, x)
        u_t = grad(self.neural_net, argnums=1)(params, t, x)

        u_fn = lambda x: self.neural_net(params, t, x)
        _, (u_x, u_xx, u_xxx, u_xxxx) = jet(u_fn, (x, ), [[1.0, 0.0, 0.0, 0.0]])

        return u_t + 100.0 / 16.0 * u * u_x + 100.0 / 16.0**2 * u_xx + 100.0 / 16.0**4 * u_xxxx

    # Initial condition loss
    @partial(jit, static_argnums=(0,))
    def loss_ics(self, params):
        # Compute forward pass
        u_pred = vmap(self.neural_net, (None, 0, 0))(params, self.X_ic[:,0], self.X_ic[:,1])
        # Compute loss
        loss_ics = np.mean((self.Y_ic.flatten() - u_pred.flatten())**2)
        return loss_ics

    # Residual loss
    @partial(jit, static_argnums=(0,))
    def loss_res(self, params, batch, key):
        t_r, x_r = batch
        # Compute forward pass        
        r_pred = vmap(self.residual_net, (None, 0, 0))(params, t_r, x_r)
        # Compute loss
        loss_r = np.mean(r_pred**2)
        return loss_r 
    
    
    @partial(jit, static_argnums=(0,))
    def update_batch(self, params, batch, key):
        t_r, x_r = batch
        # Compute forward pass        
        updated_batch = self.dataset.update(x_r=x_r, t_r=t_r, key=key)
        return updated_batch
        
    # Total loss
    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch, key):
        L_0 = self.args_dict['lambda_ic'] * self.loss_ics(params)
        L_r = self.args_dict['lambda_f'] * self.loss_res(params, batch, key)
        # Compute loss
        loss = L_r + L_0
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch, key):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batch, key)
        updated_batch = self.update_batch(params, batch, key)
        return self.opt_update(i, g, opt_state), updated_batch
    
    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params):
        u_pred = self.u_pred_fn(params, self.args_dict['t_star'], self.args_dict['x_star'])
        l2_error = np.linalg.norm(u_pred - self.args_dict['usol']) / np.linalg.norm(self.args_dict['usol']) 
        return l2_error
    
    def plot_results(self, u_pred, r_pred, usol, t_star, x_star, save_path, it):
        TT, XX = np.meshgrid(t_star, x_star) #global variables
        error = np.linalg.norm(u_pred - usol) / np.linalg.norm(usol) 
        
        fig = plt.figure(figsize=(24, 5))
        plt.subplot(1, 4, 1)
        plt.pcolor(TT, XX, usol, cmap='jet')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title(r'Exact $u(x)$')
        plt.tight_layout()

        plt.subplot(1, 4, 2)
        plt.pcolor(TT, XX, u_pred, cmap='jet')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title(r'Predicted $u(x)$')
        plt.tight_layout()

        plt.subplot(1, 4, 3)
        plt.pcolor(TT, XX, np.abs(usol - u_pred), cmap='jet')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title('Absolute error')
        plt.tight_layout()
        
        plt.subplot(1, 4, 4)
        plt.pcolor(TT, XX, np.absolute(r_pred), cmap='hot')
        plt.colorbar()
        plt.xlabel('$t$')
        plt.ylabel('$x$')
        plt.title('PDE Residuals')
        plt.tight_layout()
        
        plt.suptitle(f"Error: {error}")
        plt.savefig(os.path.join(save_path, f"solution_{it}.pdf"), dpi=150)
        plt.close()
    
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict_u(self, params, X_star):
        u_pred = vmap(self.u_net, (None, 0, 0))(params, X_star[:,0], X_star[:,1])
        return u_pred
    
    # Optimize parameters in a loop
    def train(self, nIter):
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Get batch
            updated_key, subkey = random.split(self.dataset.key)
            
            batch = self.dataset.__getitem__(0)
            self.current_count = next(self.itercount)
            self.opt_state, updated_batch = self.step(self.current_count, 
                                                      self.opt_state, 
                                                      batch=batch, 
                                                      key=subkey
                                                    )
            
            self.dataset.set_batch(updated_batch)
            self.dataset.key = updated_key
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)
                
                l2_error_value = self.compute_l2_error(params)
                loss_value = self.loss(params, batch, subkey)
                loss_ics_value = self.loss_ics(params)
                loss_res_value  = self.loss_res(params, batch, subkey)
                updated_batch_values = self.update_batch(params, batch, subkey)
                
                self.l2_error_log.append(l2_error_value)
                self.loss_log.append(loss_value)
                self.loss_ics_log.append(loss_ics_value)
                self.loss_res_log.append(loss_res_value)
                pbar.set_postfix({'Loss': loss_value, 
                                  'loss_ics' : loss_ics_value, 
                                  'loss_res':  loss_res_value,
                                  'l2 error': l2_error_value,
                                  })
                
                #plot every 10000 epochs
                if it % 10000 == 0:
                    t_plot, x_plot = updated_batch_values
                    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
                    ax = axes
                    ax.scatter(x=t_plot, y=x_plot, s=2.)
                    ax.set_title("Evolved Points")
                    plt.savefig(os.path.join(self.args_dict['save_path'], f"plot_{it}.pdf"))

                    u_pred = self.u_pred_fn(params, self.args_dict['t_star'], self.args_dict['x_star'])
                    r_pred = self.r_pred_fn(params, self.args_dict['t_star'], self.args_dict['x_star'])
                    
                    self.plot_results(u_pred=u_pred, 
                                      r_pred=r_pred,
                                     usol=self.args_dict['usol'], 
                                     t_star=self.args_dict['t_star'], 
                                     x_star=self.args_dict['x_star'], 
                                     save_path=self.args_dict['save_path'], 
                                     it=it)
                
                
         
