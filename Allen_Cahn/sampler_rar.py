import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
# from utils import *
# set_seed(1234)


            
class RAR_G_Sampler(nn.Module):
    def __init__(self, 
                 x_lim: tuple, 
                 t_lim: tuple, 
                 N=1000, #Number of collocation points
                 N_s=10000, #Number of points in the (dense) sample set
                 m=10, #Number of points to add to the original collocation points
                 device=None):
        super(RAR_G_Sampler, self).__init__()
        print("Using RAR-G Sampler.")
        self.N = N
        self.N_s = N_s
        self.m = m
        self.x_lim = x_lim
        self.t_lim = t_lim
        self.device = device if device is not None else torch.device('cpu')
        
        #Initializing the collocation points
        self.x_f = torch.zeros(N, 1, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        self.t_f = torch.zeros(N, 1, dtype=torch.float32, device=self.device).uniform_(*self.t_lim)
        
        #Initializing the dense sample points
        self.x_s = torch.zeros(N_s, 1, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        self.t_s = torch.zeros(N_s, 1, dtype=torch.float32, device=self.device).uniform_(*self.t_lim)
        self.x_new, self.t_new = None, None
        self.x_list, self.t_list = [], []
        
    def update(self, loss: torch.Tensor):
        if len(loss) != len(self.x_s) or len(loss) != len(self.t_s):
            raise RuntimeError("Input loss mismatches dimension with x, or t.")
            
        #picking the top m collocation points with the highest residuals.
        idx = torch.topk(loss.flatten(), self.m).indices
        self.x_new, self.t_new = self.x_s[idx], self.t_s[idx]
        
        self.x_list.append(self.x_new)
        self.t_list.append(self.t_new)
        
        self.x_f = torch.cat((self.x_f, self.x_new), dim=0)
        self.t_f = torch.cat((self.t_f, self.t_new), dim=0)
        return self.x_f, self.t_f
    
    def get_adaptive_points(self):
        x_adaptive, t_adaptive = None, None
        if len(self.x_list) > 0: 
            x_adaptive = torch.concat(self.x_list, dim=0)
            t_adaptive = torch.concat(self.t_list, dim=0)
        self.x_list, self.t_list = [], []
        return x_adaptive, t_adaptive

class RAD_Sampler(nn.Module):
    def __init__(self, 
                 x_lim: tuple, 
                 t_lim: tuple, 
                 N=1000, #Number of collocation points
                 N_s=10000, #Number of points in the (dense) sample set
                 device=None):
        super(RAD_Sampler, self).__init__()
        
        assert N_s >= N, "Size of Dense Sample set should be greater than the number of collocation points."
        print("Using RAD Sampler.")
        self.N = N
        self.N_s = N_s
        self.x_lim = x_lim
        self.t_lim = t_lim
        self.device = device if device is not None else torch.device('cpu')
        
        #Initializing the collocation points
        self.x_f = torch.zeros(N, 1, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        self.t_f = torch.zeros(N, 1, dtype=torch.float32, device=self.device).uniform_(*self.t_lim)
        
        #Initializing the dense sample points
        self.x_s = torch.zeros(N_s, 1, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        self.t_s = torch.zeros(N_s, 1, dtype=torch.float32, device=self.device).uniform_(*self.t_lim)
    
    def update(self, loss: torch.Tensor):
        if len(loss) != len(self.x_s) or len(loss) != len(self.t_s):
            raise RuntimeError("Input loss mismatches dimension with x, or t.")
        
        #resampling the collocation points according to the multinomial distribution
        idx = torch.multinomial(loss.flatten(), self.N)
        self.x_f, self.t_f = self.x_s[idx], self.t_s[idx]
        
        return self.x_f, self.t_f
    
    
class RAR_D_Sampler(nn.Module):
    def __init__(self, 
                 x_lim: tuple, 
                 t_lim: tuple, 
                 N=1000, #Number of collocation points
                 N_s=10000, #Number of points in the (dense) sample set
                 m=10, #Number of points to add to the original collocation points
                 device=None):
        super(RAR_D_Sampler, self).__init__()
        
        assert N_s >= N, "Size of Dense Sample set should be greater than the number of collocation points."
        print("Using RAR-D Sampler.")
        self.N = N
        self.N_s = N_s
        self.m = m
        self.x_lim = x_lim
        self.t_lim = t_lim
        self.device = device if device is not None else torch.device('cpu')
        
        #Initializing the collocation points
        self.x_f = torch.zeros(N, 1, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        self.t_f = torch.zeros(N, 1, dtype=torch.float32, device=self.device).uniform_(*self.t_lim)
        
        #Initializing the dense sample points
        self.x_s = torch.zeros(N_s, 1, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        self.t_s = torch.zeros(N_s, 1, dtype=torch.float32, device=self.device).uniform_(*self.t_lim)
        self.x_new, self.t_new = None, None
        self.x_list, self.t_list = [], []
        
    def update(self, loss: torch.Tensor):
        if len(loss) != len(self.x_s) or len(loss) != len(self.t_s):
            raise RuntimeError("Input loss mismatches dimension with x, or t.")
        
        #sampling the m new collocation points according to the multinomial distribution
        idx = torch.multinomial(loss.flatten(), self.m)
        self.x_new, self.t_new = self.x_s[idx], self.t_s[idx]
        
        self.x_list.append(self.x_new)
        self.t_list.append(self.t_new)
        
        self.x_f = torch.cat((self.x_f, self.x_new), dim=0)
        self.t_f = torch.cat((self.t_f, self.t_new), dim=0)
        return self.x_f, self.t_f
    
    def get_adaptive_points(self):
        x_adaptive, t_adaptive = None, None
        if len(self.x_list) > 0: 
            x_adaptive = torch.concat(self.x_list, dim=0)
            t_adaptive = torch.concat(self.t_list, dim=0)
        self.x_list, self.t_list = [], []
        return x_adaptive, t_adaptive
    
class Linf_Sampler(nn.Module):
    def __init__(self, 
                 x_lim: tuple, 
                 t_lim: tuple, 
                 N=1000, #Number of collocation points
                 N_s=100000, #Number of points in the (dense) sample set
                 device=None):
        super(Linf_Sampler, self).__init__()
        
        assert N_s >= N, "Size of Dense Sample set should be greater than the number of collocation points."
        print("Using RAD Sampler.")
        self.N = N
        self.N_s = N_s
        self.x_lim = x_lim
        self.t_lim = t_lim
        self.device = device if device is not None else torch.device('cpu')
        
        #Initializing the collocation points
        self.x_f = torch.zeros(N, 1, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        self.t_f = torch.zeros(N, 1, dtype=torch.float32, device=self.device).uniform_(*self.t_lim)
        
        #Initializing the dense sample points
        self.x_s = torch.zeros(N_s, 1, dtype=torch.float32, device=self.device).uniform_(*self.x_lim)
        self.t_s = torch.zeros(N_s, 1, dtype=torch.float32, device=self.device).uniform_(*self.t_lim)
    
    def update(self, loss: torch.Tensor):
        if len(loss) != len(self.x_s) or len(loss) != len(self.t_s):
            raise RuntimeError("Input loss mismatches dimension with x, or t.")
        
        idx = torch.topk(loss.flatten(), self.N).indices
        self.x_f, self.t_f = self.x_s[idx], self.t_s[idx]
        
        return self.x_f, self.t_f