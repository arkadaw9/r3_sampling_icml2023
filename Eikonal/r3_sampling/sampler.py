import torch


class R3Sampler(object):
    def __init__(self, x_lim: tuple, y_lim: tuple, uniform=True, N=1000, device=None, beta:float=1.0):
        self.N = N
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.device = device if device is not None else torch.device('cpu')
        self.x = torch.zeros(N, 1, dtype=torch.float32, device=self.device).uniform_(*x_lim)
        self.y = torch.zeros(N, 1, dtype=torch.float32, device=self.device).uniform_(*y_lim)
        self.beta = beta
        
    def update(self, loss: torch.Tensor):
        if len(loss) != len(self.x) or len(loss) != len(self.y):
            raise RuntimeError("Input loss mismatches dimension with x, or y.")
        mask = loss > self.beta * loss.mean()
        self.x_old = self.x[mask].unsqueeze(1)
        self.y_old = self.y[mask].unsqueeze(1)
        self.x_new = torch.zeros(
            self.N - len(self.x_old), 1, dtype=torch.float32, device=self.device
        ).uniform_(*self.x_lim)
        self.y_new = torch.zeros(
            self.N - len(self.y_old), 1, dtype=torch.float32, device=self.device
        ).uniform_(*self.y_lim)
        self.x = torch.cat((self.x_old, self.x_new), dim=0)
        self.y = torch.cat((self.y_old, self.y_new), dim=0)
        return self.x, self.y
    
    def update_beta(self, beta):
        self.beta = beta
        
        

class UniformSampler(object):
    def __init__(self, x_lim: tuple, y_lim: tuple, uniform=True, N=1000, device=None):
        self.N = N
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.device = device if device is not None else torch.device('cpu')
        self.update()
    
    def update(self):
        self.x = torch.zeros(
            self.N, 1, dtype=torch.float32, device=self.device
        ).uniform_(*self.x_lim)
        self.y = torch.zeros(
            self.N, 1, dtype=torch.float32, device=self.device
        ).uniform_(*self.y_lim)
        return self.x, self.y