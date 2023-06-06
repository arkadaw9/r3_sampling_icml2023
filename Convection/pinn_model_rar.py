import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import trange
import numpy as np

from model import *
from sampler_rar import *
from utils import *
from scipy.io import savemat
import scipy

set_seed(1234)

# Define the model
class PINN:
    def __init__(self, params):
        
        self.method = params['method']
        self.device = params['device']

        self.x_lim = (0, 2*np.pi)
        self.t_lim = (0, 1)
        
        #Visualization settings
        self.update_logs_every = params['viz_params']['update_logs_every']
        self.viz_every = params['viz_params']['viz_every']
        self.show_viz = params['viz_params']['show_viz']
        self.viz_dir = params['viz_params']['viz_dir']
        
        #Regular Grid for visualization
        x_star = params['viz_params']['x_star']
        t_star = params['viz_params']['t_star']
        self.XX, self.TT = np.meshgrid(x_star, t_star) # all the X grid points T times, all the T grid points X times
        self.X_star, self.T_star = self.XX.flatten(), self.TT.flatten()
        self.usol = params['viz_params']['usol']
        self.n_x = x_star.shape[0]
        self.n_t = t_star.shape[0]

        # IC
        t_ic = torch.zeros((x_star.shape[0], 1), device=self.device).float()
        x_ic = torch.tensor(x_star.reshape(-1, 1), device=self.device).float()
        self.X_ic = torch.cat([t_ic, x_ic], dim=1)
        self.Y_ic = torch.tensor(params['pde_params']['state0'], device=self.device).float()
        
        #Boundary data
        t_lb = torch.tensor(t_star.reshape(-1,1), device=self.device).float()
        x_lb = torch.ones_like(t_lb, device=self.device).float() * 0. ## x_lim is zero
        self.X_lb = torch.cat([t_lb, x_lb], dim=1)
        self.X_lb.requires_grad = True

        t_ub = torch.tensor(t_star.reshape(-1,1), device=self.device).float()
        x_ub = torch.ones_like(t_ub, device=self.device).float() * (2*np.pi) ## x_lim is zero
        self.X_ub = torch.cat([t_ub, x_ub], dim=1)
        self.X_ub.requires_grad = True
        
        #PDE Coeffts
        self.system = params['pde_params']['system']
        self.pde_nu = params['pde_params']['nu']
        self.pde_beta = params['pde_params']['beta']
        self.pde_rho = params['pde_params']['rho']
        
        
        # Initalize the network
        if params['model_params']['dnn_type'] == 'mlp':
            model_fn = DNN
        elif params['model_params']['dnn_type'] == 'modified_mlp':
            model_fn = modified_MLP
            
        self.dnn = model_fn(layers=params['model_params']['layers'], 
                            activation=params['model_params']['activation'],
                            init=params['model_params']['init']
                          ).to(self.device)
        
        
        # collocation points
        if self.method == "rar_g":
            self.sampler = RAR_G_Sampler(x_lim=self.x_lim, 
                                         t_lim=self.t_lim, 
                                         N=params['sampler_params']['N_f'], 
                                         device=self.device, 
                                         N_s=params['sampler_params']['N_s'],
                                         m=params['sampler_params']['m'],
                                        )
       
        elif self.method == "rad":
            self.sampler = RAD_Sampler(x_lim=self.x_lim, 
                                         t_lim=self.t_lim, 
                                         N=params['sampler_params']['N_f'], 
                                         device=self.device, 
                                         N_s=params['sampler_params']['N_s'],
                                        )
        elif self.method == "rar_d":
            self.sampler = RAR_D_Sampler(x_lim=self.x_lim, 
                                         t_lim=self.t_lim, 
                                         N=params['sampler_params']['N_f'], 
                                         device=self.device, 
                                         N_s=params['sampler_params']['N_s'],
                                         m=params['sampler_params']['m'],
                                        )
        elif self.method == "l_inf":
            self.sampler = Linf_Sampler(x_lim=self.x_lim, 
                                        t_lim=self.t_lim, 
                                        N=params['sampler_params']['N_f'], 
                                        device=self.device, 
                                        N_s=params['sampler_params']['N_s'],
                                       )
            
        #initializing collocation points from the sampler
        self.x_f = self.sampler.x_f
        self.t_f = self.sampler.t_f
        self.x_f.requires_grad = True
        self.t_f.requires_grad = True
        
        #initializing the dense sample set from the sampler
        self.x_s = self.sampler.x_s
        self.t_s = self.sampler.t_s
        self.x_s.requires_grad = True
        self.t_s.requires_grad = True
            
        self.temp_dict = {
                          "r_pred":None
                         }
        
        # Resampling period
        self.k = params['sampler_params']['k']    
        
        # Use optimizers to set optimizer initialization and update functions
        self.lr = params['train_params']['lr']
        self.optimizer_fn = params['train_params']['optimizer']
        self.optimizer = self.optimizer_fn(self.dnn.parameters(), self.lr)
        
        self.use_lr_scheduler = params['train_params']['use_lr_scheduler']

        if self.use_lr_scheduler:
            print("Initializing LR Scheduler")
            self.decay_rate = params['train_params']['decay_rate']
            self.decay_steps = params['train_params']['decay_steps']
            self.lr_scheduler_fn = params['train_params']['lr_scheduler']
            self.lr_scheduler = self.lr_scheduler_fn(
                                                    self.optimizer, 
                                                    step_size=self.decay_steps, 
                                                    gamma=self.decay_rate
                                                )
            self.use_dynamic_lrsch = params['dynamic_scheduler_params']['use_dynamic']
            if self.use_dynamic_lrsch:
                self.lr_chg_beta = params['dynamic_scheduler_params']['lr_chg_beta']
                self.new_decay_steps = params['dynamic_scheduler_params']['new_decay_steps']
                
            
        self.lambda_ic = params['train_params']['lambda_ic']
        self.lambda_bc = params['train_params']['lambda_bc']
        self.lambda_f = params['train_params']['lambda_f']
        
        
        
        
        # Creating logs
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []
        
        self.abs_err_log = []
        self.rel_l2_err_log = []
        self.skew_f_log = []
        self.kurtosis_f_log = []
        self.skew_grid_log = []
        self.kurtosis_grid_log = []
        
        self.beta_log = []
                
    def neural_net(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u
    
    def residual_net(self, x, t):
        """ Autograd for calculating the residual for different systems."""
        u = self.neural_net(x=x, t=t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
            )[0]

        if 'convection' in self.system or 'diffusion' in self.system:
            f = u_t - self.pde_nu*u_xx + self.pde_beta*u_x
        elif 'rd' in self.system:
            f = u_t - self.pde_nu*u_xx - self.pde_rho*u + self.pde_rho*u**2
        elif 'reaction' in self.system:
            f = u_t - self.pde_rho*u + self.pde_rho*u**2
        return f  

    def net_b_derivatives(self, u_lb, u_ub, x_bc_lb, x_bc_ub):
        """For taking BC derivatives."""

        u_lb_x = torch.autograd.grad(
            u_lb, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
            )[0]

        u_ub_x = torch.autograd.grad(
            u_ub, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
            )[0]
        return u_lb_x, u_ub_x
    
    def loss_bcs(self):
        t_lb, x_lb = self.X_lb[:,0:1], self.X_lb[:,1:2]
        t_ub, x_ub = self.X_ub[:,0:1], self.X_ub[:,1:2]
        u_pred_lb = self.neural_net(t=t_lb, x=x_lb)
        u_pred_ub = self.neural_net(t=t_ub, x=x_ub)
        loss_bc = torch.mean((u_pred_lb - u_pred_ub) ** 2)
        if self.pde_nu != 0:
            u_pred_lb_x, u_pred_ub_x = self.net_b_derivatives(u_pred_lb, u_pred_ub, x_lb, x_ub)
            loss_bc += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)
        return loss_bc
        
    def loss_ics(self):
        # Evaluate the network over IC
        u_pred = self.neural_net(t=self.X_ic[:,0:1], x=self.X_ic[:,1:2])
        # Compute the initial loss
        loss_ics = torch.mean((self.Y_ic.flatten() - u_pred.flatten())**2)
        return loss_ics
    
    def loss_res_rar(self):
        self.r_pred = self.residual_net(t=self.t_f, x=self.x_f)
        loss_r = torch.mean(self.r_pred**2)
        self.temp_dict['r_pred'] = self.r_pred
        return loss_r
        
    def loss_res(self):
        if self.method in ["rar_g", "rad", "rar_d","l_inf"]:
            return self.loss_res_rar()
 
    def loss(self, lamda_ic=1.0, lamba_f=1.0, lamda_bc=1.0):
        L_0 = self.loss_ics()
        L_bc = self.loss_bcs()
        L_f = self.loss_res()
        
        # Compute loss
        loss = lamba_f * L_f + lamda_ic * L_0 + lamda_bc * L_bc
                   
        return loss, L_0, L_f, L_bc
    
    # Optimize parameters in a loop
    def train(self, nIter = 10000):
        
        self.dnn.train()
        
        pbar = trange(nIter+1)
        # Main training loop
        for it in pbar:
            
            #update the sampler every k epochs
            if it % self.k == 0 and it >0:
                loss_s = torch.abs(self.residual_net(x=self.x_s, t=self.t_s))
                with torch.no_grad():
                    x_f, t_f = self.sampler.update(loss=loss_s.detach())
                    self.x_f, self.t_f = x_f, t_f
                    self.x_f.requires_grad = True
                    self.t_f.requires_grad = True
#                     print(f"Collocation Points updated. Length: {self.x_f.shape}")
                
            self.optimizer.zero_grad()

            loss, L_0, L_f, L_bc = self.loss(lamda_ic=self.lambda_ic, lamba_f=self.lambda_f, lamda_bc=self.lambda_bc)
            loss.backward()
            self.optimizer.step()
            if self.use_lr_scheduler:
                self.lr_scheduler.step()
            
            if it % self.update_logs_every == 0:
                u_pred = self.predict(t=self.T_star, x=self.X_star)
                u_pred = (u_pred.reshape(self.n_t, self.n_x)).T
                
                l2_error = np.linalg.norm(u_pred - self.usol) / np.linalg.norm(self.usol) 
                abs_error = np.absolute(u_pred-self.usol).mean()
                
                r_pred = self.r_pred.detach().cpu().numpy()
                f_pred = np.absolute(self.predict_f(t=self.T_star, x=self.X_star))
                
                skew_f = scipy.stats.skew(r_pred.flatten())
                skew_grid = scipy.stats.skew(f_pred.flatten())
                kurtosis_f = scipy.stats.kurtosis(r_pred.flatten(), fisher=True)
                kurtosis_grid = scipy.stats.kurtosis(f_pred.flatten(), fisher=True)
                
                self.abs_err_log.append(abs_error)
                self.rel_l2_err_log.append(l2_error)
                self.loss_log.append(loss.item())
                self.loss_bcs_log.append(L_bc.item())
                self.loss_ics_log.append(L_0.item())
                self.loss_res_log.append(L_f.item())
                
                self.skew_f_log.append(skew_f)
                self.kurtosis_f_log.append(kurtosis_f)
                self.skew_grid_log.append(skew_grid)
                self.kurtosis_grid_log.append(kurtosis_grid)
                
                dict_ = {'Loss': loss.item(), 
                         'loss_ics' : L_0.item(),
                         'loss_bcs' : L_bc.item(),
                         'loss_res':  L_f.item(),
                         'l2_error':l2_error,
                         'abs_error':abs_error,
                        }
                        
                if self.use_lr_scheduler:
                    dict_['curr_lr'] = self.lr_scheduler.get_last_lr()
                pbar.set_postfix(dict_)
                
            if it % self.viz_every == 0:
                self.intermediate_visualization(epoch=it)
            
                
    def predict(self, t, x):
        self.dnn.eval()
        x = torch.tensor(x, requires_grad=True).float().to(self.device).unsqueeze(1)
        t = torch.tensor(t, requires_grad=True).float().to(self.device).unsqueeze(1)
        
        u = self.neural_net(t=t, x=x)
        u = u.detach().cpu().numpy()
        return u
    
    def predict_f(self, t, x):
        self.dnn.eval()
        x = torch.tensor(x, requires_grad=True).float().to(self.device).unsqueeze(1)
        t = torch.tensor(t, requires_grad=True).float().to(self.device).unsqueeze(1)

        f = self.residual_net(t=t, x=x)
        f = f.detach().cpu().numpy()
        return f
    
    def intermediate_visualization(self, epoch):
        if self.viz_dir is not None:
            viz_path_scatter = os.path.join(self.viz_dir, f"EvoSample_viz_{str(epoch).zfill(7)}.pdf")
            viz_path_pred = os.path.join(self.viz_dir, f"Predictions_viz_{str(epoch).zfill(7)}.pdf")
            viz_path_ltw = os.path.join(self.viz_dir, f"Lt_w_viz_{str(epoch).zfill(7)}.pdf")
        else:
            viz_path_scatter = None
            viz_path_pred = None
            viz_path_ltw = None
            
        u_pred = self.predict(t=self.T_star, x=self.X_star)
        u_pred = (u_pred.reshape(self.n_t, self.n_x)).T
        
        f_pred = np.absolute(self.predict_f(t=self.T_star, x=self.X_star))
        f_pred_viz = (f_pred.reshape(self.n_t, self.n_x)).T
        
        #plotting the output predictions
        visualize(usol=self.usol, u_pred=u_pred, XX=self.XX, TT=self.TT, 
                  show_viz=self.show_viz, viz_path=viz_path_pred)
        
        if self.viz_dir is not None:
            viz_dict = {
                        "usol":self.usol,
                        "u_pred":u_pred,
                        "f_pred":f_pred_viz, #the residuals on the viz grid
                        "r_pred":self.r_pred.detach().cpu().numpy(), #the residuals on the collocation points
                        "XX":self.XX,
                        "TT":self.TT
                        }
            savemat(os.path.join(self.viz_dir, f"viz_dict_epoch_{str(epoch).zfill(7)}.mat"), viz_dict)
        
        if self.method in ["rar_g", "rad", "rar_d","l_inf"]:
            x_plot, t_plot = self.x_f, self.t_f
            if self.method in ["rar_g", "rar_d"]:
#                 x_new, t_new = self.sampler.x_new, self.sampler.t_new
                x_new, t_new = self.sampler.get_adaptive_points()
            else:
                x_new, t_new = None, None
            f_pred = np.absolute(self.predict_f(t=self.T_star, x=self.X_star))
            f_pred = (f_pred.reshape(self.n_t, self.n_x)).T
            visualize_scatter_rar(x=x_plot, 
                                  t=t_plot, 
                                  x_new=x_new,
                                  t_new=t_new,
                                  f_pred=f_pred, 
                                  x_lim=self.x_lim,
                                  t_lim=self.t_lim,
                                  show_viz=self.show_viz,
                                  viz_path=viz_path_scatter
                                 )
            if self.viz_dir is not None:
                if x_new is not None:
                    x_new, t_new = x_new.detach().cpu().numpy(), t_new.detach().cpu().numpy()
                else:
                    x_new, t_new = [], []
                scatter_plot_dict = {
                                     "x_new":x_new, "t_new":t_new, 
                                     "x_old":x_plot.detach().cpu().numpy(), "t_old":t_plot.detach().cpu().numpy(), 
                                     "f_pred":f_pred,
                                     "x_lim":self.x_lim, "t_lim":self.t_lim,
                                    }
                savemat(os.path.join(self.viz_dir, f"scatter_dict_epoch_{str(epoch).zfill(7)}.mat"), scatter_plot_dict)

def visualize_scatter_rar(x, t, f_pred, x_lim, t_lim, x_new=None, t_new=None, show_viz=True, viz_path=None): 
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(t.detach().cpu().numpy(), x.detach().cpu().numpy(), color='b', s=2., alpha=0.5, label="Collocation points")
    if x_new is not None:
        ax[0].scatter(t_new.detach().cpu().numpy(), x_new.detach().cpu().numpy(), color='r', s=2., alpha=0.5, label="Adaptive Samples")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("x")
    ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
            mode="expand", borderaxespad=0, ncol=2, fontsize=8)

    h = ax[1].imshow(f_pred, interpolation='nearest', cmap='gray',
                extent=[t_lim[0], t_lim[1], x_lim[0],  x_lim[1]],
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    ax[1].set_title("PDE Residuals", fontsize=10)

    if show_viz:
        plt.show()
    if viz_path is not None:
        plt.savefig(viz_path)
    plt.close()