import torch
import numpy as np

class DatasetGenerator(torch.utils.data.Dataset):
    def __init__(self, x_star, t_star, usol, N=10):
        super(DatasetGenerator, self).__init__()
        self.x_star = x_star
        self.t_star = t_star
        self.usol = usol
        self.N = N
        self.x_lim = (x_star.min(), x_star.max())
    
    def __get_item__(self, idx):
        t_idx = len(self.t_star)//self.N
        if idx == self.N:
            usol = np.copy(self.usol[:,(idx-1)*t_idx:])
            t_star_ = np.copy(self.t_star[(idx-1)*t_idx:])
            t_star_ -= t_star_[0] # timeshift
        else:
            usol = np.copy(self.usol[: , (idx-1)*t_idx : idx*t_idx+1])
            t_star_ = np.copy(self.t_star[(idx-1)*t_idx : idx*t_idx+1])
            t_star_ -= t_star_[0] # timeshift
        
        t0, t1 = 0, (t_star_[-1,0]-t_star_[0,0]) # time limit (shifted)
        state0 = np.copy(usol[:, 0:1]) # first point of the cropeed usol
        t_lim = (t0, t1)
        return state0, t_star_, usol, t_lim, self.x_lim