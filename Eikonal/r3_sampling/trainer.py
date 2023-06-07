
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import trange

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

from r3_sampling.models import modified_MLP
from r3_sampling.sampler import R3Sampler, UniformSampler



# Define the model
class BaseTrainer(object):
    def __init__(
        self, 
        data, 
        out_dir="./", 
        N_f=1000, 
        device=torch.device('cpu'),
        layers=[2, 128, 128, 128, 128, 1],
        x_lim=(-1,1),
        y_lim=(-1,1)
    ):  
        self.device = device
        self.data = data
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.out_dir = out_dir
        self.N_f = N_f 
        
        # Network architecture
        self.layers = layers

        self.usol = data['sdf']
        X_ic = data['zero_contours'][:, ::-1].copy()
        grid_size = data['grid_size']
        mask = data['img']
        
        # Regular Grid for visualization
        x_star = np.linspace(-1, 1, grid_size[0])
        y_star = np.linspace(-1, 1, grid_size[1])
        XX, YY = np.meshgrid(x_star, y_star)
        self.X_star, self.Y_star = XX.flatten(), YY.flatten()
        self.grid_size = grid_size
        
        # IC
        self.init_ic(X_ic)
        
        # BC
        self.init_bc()
        
        # Collocation Points
        self.init_xf()

        # Initalize the network
        self.dnn = modified_MLP(layers, activation=nn.Tanh).to(self.device)
        
        # Initalize optimizer and scheduler
        self.init_optimizer()
        
        # logs
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_res_log = []
    
    def init_ic(self, ic):
        self.X_ic = torch.tensor(ic, device=self.device).float()
        return None
        
    def init_bc(self):
        x_bc = np.linspace(*self.x_lim, 10)
        y_bc = np.linspace(*self.y_lim, 10)
        self.X_bc, self.Y_bc = np.meshgrid(x_bc, y_bc)
        self.X_bc = torch.tensor(self.X_bc, device=self.device)
        self.X_bc = torch.cat([self.X_bc[[0, -1], :].reshape(-1, 1), self.X_bc[:, [0, -1]].reshape(-1, 1)])
        self.X_bc = self.X_bc.float()
        self.Y_bc = torch.tensor(self.Y_bc, device=self.device)
        self.Y_bc = torch.cat([self.Y_bc[[0, -1], :].reshape(-1, 1), self.Y_bc[:, [0, -1]].reshape(-1, 1)])
        self.Y_bc = self.Y_bc.float()
        return None
    
    def init_xf(self):
        self.f_sampler = UniformSampler(
            x_lim=self.x_lim, 
            y_lim=self.y_lim, 
            uniform=True, 
            N=self.N_f, 
            device=self.device
        )
        self.x_f = self.f_sampler.x
        self.y_f = self.f_sampler.y
        self.x_f.requires_grad=True
        self.y_f.requires_grad=True
        return self.x_f, self.y_f
    
    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.dnn.parameters())
        self.decay_rate = 0.9
        self.decay_steps = 10000
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.decay_steps, 
            gamma=self.decay_rate
        )
        return None
    
    def neural_net(self, x, y):
        u = self.dnn(torch.cat([x, y], dim=1))
        return u

    def residual_net(self, x, y):
        u = self.neural_net(x=x, y=y)
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        return u_x**2 + u_y**2 - 1.0 

    def loss_ics(self):
        u_pred = self.neural_net(x=self.X_ic[:,0:1], y=self.X_ic[:,1:2])
        loss_ics = (u_pred**2).mean() # SDF = 0 on surface
        return loss_ics
    
    def loss_bcs(self):
        u_pred = self.neural_net(x=self.X_bc, y=self.Y_bc)
        loss_bcs = (F.relu(-u_pred)).mean()
        return loss_bcs
    
    def loss_res(self): 
        r_pred = self.residual_net(x=self.x_f, y=self.y_f)
        loss_r = torch.mean(r_pred**2)
        return loss_r, r_pred  
 
    def loss(self):
        L_0 = self.loss_ics()
        L_f, self.r_pred = self.loss_res()
        L_bc = self.loss_bcs()
        loss = L_f + 500. * L_0 + 10. * L_bc
        return loss, L_0, L_f

    def update_sample(self):
        return None
    
    def train(self, nIter=10000):
        self.dnn.train()
        pbar = trange(nIter)
        for it in pbar:
            self.optimizer.zero_grad()
            loss, L_0, L_f = self.loss()
            
            self.update_sample()
                
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            
            if it % 100 == 0:
                self.loss_log.append(loss.item())
                self.loss_ics_log.append(L_0.item())
                self.loss_res_log.append(L_f.item())

                pbar.set_postfix({'Loss': loss.item(), 
                                  'loss_ics' : L_0.item(), 
                                  'loss_res':  L_f.item()})
                
            if it % 5000 == 0:
                u_pred = self.predict(x=self.X_star, y=self.Y_star)
                u_pred = (u_pred.reshape(self.grid_size[1], self.grid_size[0])) 
                f_pred = np.absolute(self.predict_f(x=self.X_star, y=self.Y_star))
                f_pred = (f_pred.reshape(self.grid_size[1], self.grid_size[0])).T
                self.visualize(it, self.usol, u_pred, f_pred)
                self.save(iter=it)
        return None
    
    def predict(self, x, y):
        self.dnn.eval()
        x = torch.tensor(x, requires_grad=True).float().to(self.device).unsqueeze(1)
        y = torch.tensor(y, requires_grad=True).float().to(self.device).unsqueeze(1)

        u = self.neural_net(x=x, y=y)
        u = u.detach().cpu().numpy()
        return u
    
    def predict_f(self, x, y):
        self.dnn.eval()
        x = torch.tensor(x, requires_grad=True).float().to(self.device).unsqueeze(1)
        y = torch.tensor(y, requires_grad=True).float().to(self.device).unsqueeze(1)

        f = self.residual_net(x=x, y=y)
        f = f.detach().cpu().numpy()
        return f
        
    def visualize(self, iter, usol, u_pred, f_pred):
        fig, axes = plt.subplots(1, 5, figsize=(25, 4.4), dpi=100)
        
        ax = axes[0]
        h = ax.imshow(usol, cmap='jet', aspect="auto")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        ax.set_xlabel('$y$')
        ax.set_ylabel('$x$')
        ax.set_title(r'Exact $u(x)$')

        ax = axes[1]
        h = ax.imshow(u_pred, cmap='jet', aspect="auto")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        ax.set_xlabel('$y$')
        ax.set_ylabel('$x$')
        ax.set_title(r'Predicted $u(x)$')

        ax = axes[2]
        h = ax.imshow(np.abs(usol - u_pred), cmap='jet', aspect="auto")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)
        ax.set_xlabel('$y$')
        ax.set_ylabel('$x$')
        ax.set_title('Absolute error')
        
        ax = axes[3]
        ax.scatter(
            self.y_f.detach().cpu().numpy(), 
            self.x_f.detach().cpu().numpy(), 
            s=2., alpha=0.5, color="#b87333"
        )
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        
        ax = axes[4]
        h = ax.imshow(f_pred, interpolation='nearest', cmap='copper',
                    extent=[self.y_lim[0], self.y_lim[1], self.x_lim[0], self.x_lim[1]],
                    origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = fig.colorbar(h, cax=cax)
        cbar.ax.tick_params(labelsize=15)

        fig.tight_layout()
        fn = os.path.join(self.out_dir, "%d.jpg" % iter)
        fig.savefig(fn)
        return None

    def save_loss(self):
        df = pd.DataFrame({
            "loss_ic": self.loss_ics_log, 
            "loss_res": self.loss_res_log, 
        })
        df.to_csv(os.path.join(self.out_dir, "loss.csv"))
        return None
        
    def save_checkpoint(self, iter):
        torch.save({
            "state_dict": self.dnn.state_dict(),
            "network": self.dnn.__repr__(),
        }, f=os.path.join(self.out_dir, "epoch_%d.pt" % iter))
        return None
    
    def save(self, iter):
        self.save_loss()
        self.save_checkpoint(iter)
        return None
      
        
        
        
class R3Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(R3Trainer, self).__init__(*args, **kwargs)
        
    def init_xf(self):
        self.beta = 1.0
        self.f_sampler = R3Sampler(
            x_lim=self.x_lim, 
            y_lim=self.y_lim, 
            uniform=True, 
            N=self.N_f, 
            device=self.device, 
            beta=self.beta
        )
        self.x_f = self.f_sampler.x
        self.y_f = self.f_sampler.y
        self.x_f.requires_grad=True
        self.y_f.requires_grad=True
        return self.x_f, self.y_f
    
    def update_sample(self):
        with torch.no_grad():
            x_f, y_f = self.f_sampler.update(torch.abs(self.r_pred).detach())
            self.x_f, self.y_f = x_f, y_f
            self.x_f.requires_grad = True
            self.y_f.requires_grad = True
        return x_f, y_f
    
    def save_checkpoint(self, iter):
        torch.save({
            "sampler": self.f_sampler,
            "state_dict": self.dnn.state_dict(),
            "network": self.dnn.__repr__(),
        }, f=os.path.join(self.out_dir, "epoch_%d.pt" % iter))
        return None
    
    
        
class PINNTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(PINNTrainer, self).__init__(*args, **kwargs)
        return None
    
    def init_xf(self):
        n = int(np.ceil(np.sqrt(self.N_f)))
        x_f = torch.linspace(-1, 1, n).to(self.device)
        y_f = torch.linspace(-1, 1, n).to(self.device)
        self.x_f, self.y_f = torch.meshgrid(x_f, y_f)
        self.x_f = self.x_f.reshape(-1, 1)
        self.y_f = self.y_f.reshape(-1, 1)
        self.x_f.requires_grad = True
        self.y_f.requires_grad = True
        return x_f, y_f
    
    
    
class uPINNTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(uPINNTrainer, self).__init__(*args, **kwargs)
        return None
    
    def update_sample(self):
        with torch.no_grad():
            x_f, y_f = self.f_sampler.update()
            self.x_f, self.y_f = x_f, y_f
            self.x_f.requires_grad = True
            self.y_f.requires_grad = True
        return x_f, y_f

    
        
class IMPTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(IMPTrainer, self).__init__(*args, **kwargs)
        return None
    
    def init_xf(self):
        n = int(np.ceil(np.sqrt(self.N_f)))
        x_f = torch.linspace(-1, 1, n).to(self.device)
        y_f = torch.linspace(-1, 1, n).to(self.device)
        self.x_f, self.y_f = torch.meshgrid(x_f, y_f)
        self.x_f = self.x_f.reshape(-1, 1)
        self.y_f = self.y_f.reshape(-1, 1)
        self.x_f.requires_grad = True
        self.y_f.requires_grad = True
        return x_f, y_f 
        
    def loss_res(self): 
        r_pred = self.residual_net(x=self.x_f, y=self.y_f)
        loss_r = torch.mean(r_pred * r_pred.detach())
        return loss_r, r_pred 
    
    
    
        
class uIMPTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(uIMPTrainer, self).__init__(*args, **kwargs)
        return None
    
    def update_sample(self):
        with torch.no_grad():
            x_f, y_f = self.f_sampler.update()
            self.x_f, self.y_f = x_f, y_f
            self.x_f.requires_grad = True
            self.y_f.requires_grad = True
        return x_f, y_f 
        
    def loss_res(self): 
        r_pred = self.residual_net(x=self.x_f, y=self.y_f)
        loss_r = torch.mean(r_pred * r_pred.detach())
        return loss_r, r_pred 