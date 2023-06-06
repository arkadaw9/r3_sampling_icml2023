import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class UniformSampler(nn.Module):
    def __init__(self, x_lim: tuple, t_lim: tuple, N=1000, device=None):
        super(UniformSampler, self).__init__()
        
        self.N = N
        self.x_lim = x_lim
        self.t_lim = t_lim
        self.device = device if device is not None else torch.device('cpu')
        self.update()
    
    def update(self):
        self.x = torch.zeros(self.N, 1, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        self.t = torch.zeros(self.N, 1, dtype=torch.float32, device=self.device).uniform_(*self.t_lim)
        return self.x, self.t
    
class CausalPINNSampler(nn.Module):
    def __init__(self, x_lim: tuple, t_lim: tuple, n_x=100, n_t=100, device=None):
        super(CausalPINNSampler, self).__init__()
        
        self.n_x = n_x
        self.n_t = n_t
        self.x_lim = x_lim
        self.t_lim = t_lim
        self.device = device if device is not None else torch.device('cpu')
        self.update()
    
    def update(self):
        x_grid = torch.zeros(self.n_x, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        t_grid = torch.sort(torch.zeros(self.n_t, dtype=torch.float32, device=self.device).uniform_(*self.t_lim), dim=0)[0]
        XX, TT = torch.meshgrid(x_grid, t_grid)
        self.x = XX.reshape(-1, 1) 
        self.t = TT.reshape(-1, 1)
        return self.x, self.t
           
            
class R3Sampler(nn.Module):
    def __init__(self, x_lim: tuple, t_lim: tuple, N=1000, device=None):
        super(R3Sampler, self).__init__()
        
        self.N = N
        self.x_lim = x_lim
        self.t_lim = t_lim
        self.device = device if device is not None else torch.device('cpu')
        self.x = torch.zeros(N, 1, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        self.t = torch.zeros(N, 1, dtype=torch.float32, device=self.device).uniform_(*self.t_lim)
    
    def update(self, loss: torch.Tensor):
        if len(loss) != len(self.x) or len(loss) != len(self.t):
            raise RuntimeError("Input loss mismatches dimension with x, or t.")
        
        x_old, t_old, x_new, t_new = self.get_old_new(loss)
        
        self.x = torch.cat((x_old, x_new), dim=0)
        self.t = torch.cat((t_old, t_new), dim=0)
        return self.x, self.t
    
    def get_old_new(self, loss: torch.Tensor):
        fitness = self.get_fitness(loss)
        mask = fitness > fitness.mean()
        x_old = self.x[mask].unsqueeze(1)
        t_old = self.t[mask].unsqueeze(1)
        x_new = torch.zeros(self.N - torch.sum(mask), 1, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        t_new = torch.zeros(self.N - torch.sum(mask), 1, dtype=torch.float32, device=self.device).uniform_(*self.t_lim)
        return x_old, t_old, x_new, t_new
    
    def get_fitness(self, loss):
        return loss
       
class CausalR3Sampler(R3Sampler):
    def __init__(self, 
                 x_lim: tuple, 
                 t_lim: tuple, 
                 N=1000, 
                 device=None, 
                 beta:float=1.0, 
                 alpha:float=10.0, 
                 gate_type:str='relu_tanh',
                 beta_lr:float=1e-4,
                 tol:float=20.0,
                 grad_clip:float=1e-1,
                 **kwargs
                ):
        super(CausalR3Sampler, self).__init__(x_lim=x_lim, t_lim=t_lim, N=N, device=device, **kwargs)
          
        #Gate Parameters
        self.beta = torch.ones(1, device=self.device)*beta
        self.alpha = torch.ones(1, device=self.device)*alpha
        self.gate_type = gate_type
        self.beta_lr = beta_lr
        self.tol = tol
        
        #for gradient clipping
        self.grad_clip = torch.tensor(grad_clip, device=self.device).float()

    def get_fitness(self, loss):
        return loss * self.causal_gate(self.t)
    
    def update(self, loss: torch.Tensor):
        if len(loss) != len(self.x) or len(loss) != len(self.t):
            raise RuntimeError("Input loss mismatches dimension with x, or t.")
        
        x_old, t_old, x_new, t_new = self.get_old_new(loss)
        
        self.update_beta(loss)
        
        self.x = torch.cat((x_old, x_new), dim=0)
        self.t = torch.cat((t_old, t_new), dim=0)
        return self.x, self.t
    
    def causal_gate(self, t):
        t_norm = (t - self.t_lim[0]) / (self.t_lim[1]-self.t_lim[0])
        if self.gate_type == 'sigmoid':
            return 1 - torch.sigmoid(self.alpha * (t_norm - self.beta))
        elif self.gate_type == 'tanh':
            return (1 - torch.tanh(self.alpha * (t_norm - self.beta)))/2
        elif self.gate_type == 'relu_tanh':
            return F.relu( -torch.tanh(self.alpha * (t_norm - self.beta)) )
    
    def update_beta(self, loss):
        fitness = (loss * self.causal_gate(self.t))**2
        gradient = torch.exp(-self.tol * fitness.mean())
        gradient = gradient if gradient <= self.grad_clip else self.grad_clip
        self.beta += self.beta_lr * gradient

class HoldoutSampler(nn.Module):
    def __init__(self, x_lim: tuple, t_lim: tuple, N=1000, n=1000, device=None):
        super(HoldoutSampler, self).__init__()
        
        self.N = N
        self.n = n
        
        self.x_lim = x_lim
        self.t_lim = t_lim
        self.device = device if device is not None else torch.device('cpu')
        self.x, self.y = self.get_uniform()
    
    def get_uniform(self):
        self.x = torch.zeros(self.N, 1, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        self.t = torch.zeros(self.N, 1, dtype=torch.float32, device=self.device).uniform_(*self.t_lim)
        return self.x, self.t
    
    def update(self):
        idx = np.random.choice(self.N, self.n, replace=False)
        return self.x[idx], self.t[idx]
