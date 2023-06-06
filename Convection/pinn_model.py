import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import trange
import numpy as np

from model import *
from sampler import *
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
        if self.method in ["causal_r3", "r3_sampling"]:
            if self.method == "causal_r3":
                self.sampler = CausalR3Sampler(
                                                x_lim=self.x_lim, 
                                                t_lim=self.t_lim, 
                                                N=params['sampler_params']['N_f'], 
                                                device=self.device, 
                                                beta=params['causal_gate_params']['beta'], 
                                                alpha=params['causal_gate_params']['alpha'], 
                                                gate_type=params['causal_gate_params']['gate_type'],
                                                beta_lr=params['causal_gate_params']['beta_lr'],
                                                tol=params['causal_gate_params']['tol'],
                                                grad_clip=params['causal_gate_params']['grad_clip'],
                                             )
            elif self.method in ["r3_sampling"]:
                self.sampler = R3Sampler(
                                            x_lim=self.x_lim, 
                                            t_lim=self.t_lim, 
                                            N=params['sampler_params']['N_f'], 
                                            device=self.device, 
                                        )

            self.x_f = self.sampler.x
            self.t_f = self.sampler.t
            self.x_f.requires_grad = True
            self.t_f.requires_grad = True
            
            if self.method == "causal_r3":
                self.plot_causal_gate()#Plotting Causal Gate at initialization
            
            self.temp_dict = {
                                "r_pred":None
                             }
        
        elif self.method in ["causal_pinn", "causal_pinn_uniform"]:
            # Causality Parameters
            self.c_nt = params['causal_pinn_params']['n_t'] #causal t-grid locations
            self.c_nx = params['causal_pinn_params']['n_x'] #causal x-grid locations
            self.tol_list = [torch.tensor(i).float() for i in params['causal_pinn_params']['tol_list']] #tolerance or causality parameter
            self.M = torch.tensor(np.triu(np.ones((self.c_nt, self.c_nt)), k=1).T, device=self.device).float()
            self.tol_pt = 0
            self.tol_iter = 0 #the number of iterations the current tolerance value has been used
            
            self.sampler = CausalPINNSampler(x_lim=self.x_lim, 
                                             t_lim=self.t_lim, 
                                             n_x=self.c_nx, 
                                             n_t=self.c_nt, 
                                             device=self.device, 
                                            )
            self.x_f = self.sampler.x
            self.t_f = self.sampler.t
            self.x_f.requires_grad = True
            self.t_f.requires_grad = True
            
            self.temp_dict = {
                                "L_t":None,
                                "W":None,
                             }

        elif self.method in ["pinn", "pinn_uniform", "curriculum_reg"]:
            self.sampler = UniformSampler(
                                          x_lim=self.x_lim, 
                                          t_lim=self.t_lim, 
                                          N=params['sampler_params']['N_f'], 
                                          device=self.device, 
                                         )
            self.x_f = self.sampler.x
            self.t_f = self.sampler.t
            self.x_f.requires_grad = True
            self.t_f.requires_grad = True
            
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
        
    
    def plot_causal_gate(self, epoch=0.):
        # Plotting the causal gate
        t_ = torch.linspace(self.t_lim[0], self.t_lim[1], 1000, device=self.device)
        gate = self.sampler.causal_gate(t_)
        if self.viz_dir is not None:
            viz_path = os.path.join(self.viz_dir, f"Causal_Gate_epoch_{str(epoch).zfill(7)}.pdf")
            gate_dict = {
                          "t":t_.detach().cpu().numpy(), "gate":gate.detach().cpu().numpy(), 
                        }
            savemat(os.path.join(self.viz_dir, f"causal_gate_epoch_{str(epoch).zfill(7)}.mat"), gate_dict)
        else:
            viz_path = None
        plot_gate(t=t_.detach().cpu().numpy(), gate=gate.detach().cpu().numpy(), show_viz=self.show_viz, viz_path=viz_path)
            
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
    
    def get_causal_weights_and_residuals(self):
        self.r_pred = self.residual_net(t=self.t_f, x=self.x_f)
        f_pred = self.r_pred.view(self.c_nt, self.c_nx)
        L_t = torch.mean(f_pred**2, dim=1)
        tol = self.tol_list[self.tol_pt]
        W = torch.exp(-tol * (self.M @ L_t)).detach()
        return L_t, W
    
    def loss_res_causal_pinn(self):
        L_t, W = self.get_causal_weights_and_residuals()
        # Compute loss
        loss_r = torch.mean(W * L_t)
        self.temp_dict['L_t'] = L_t
        self.temp_dict['W'] = W
        return loss_r
        
    def loss_res_evo_sample(self): 
        self.r_pred = self.residual_net(t=self.t_f, x=self.x_f)
        if self.method == "causal_r3":
            # Compute loss
            gate = self.sampler.causal_gate(self.t_f)
            loss_r = torch.mean((self.r_pred * gate)**2)
        
        elif self.method == "r3_sampling":
            loss_r = torch.mean(self.r_pred**2)
            
        self.temp_dict['r_pred'] = self.r_pred
        return loss_r
    
    def loss_res_pinn(self):
        self.r_pred = self.residual_net(t=self.t_f, x=self.x_f)
        loss_r = torch.mean(self.r_pred**2)
        return loss_r
        
    def loss_res(self):
        if self.method in ["causal_r3", "r3_sampling"]:
            return self.loss_res_evo_sample()
        elif self.method in ["causal_pinn", "causal_pinn_uniform"]:
            return self.loss_res_causal_pinn()
        elif self.method in ["pinn", "pinn_uniform", "curriculum_reg"]:
            return self.loss_res_pinn()

    def loss(self, lamda_ic=1.0, lamba_f=1.0, lamda_bc=1.0):
        L_0 = self.loss_ics()
        L_bc = self.loss_bcs()
        L_f = self.loss_res()
        
        # Compute loss
        loss = lamba_f * L_f + lamda_ic * L_0 + lamda_bc * L_bc
        
        if self.method in ["causal_r3", "r3_sampling"]:
            with torch.no_grad():
                r_pred = self.temp_dict['r_pred']
                x_f, t_f = self.sampler.update(torch.abs(r_pred).detach())
                self.x_f, self.t_f = x_f, t_f
                self.x_f.requires_grad = True
                self.t_f.requires_grad = True
        
        elif self.method in ["causal_pinn_uniform", "pinn_uniform"]:
            with torch.no_grad():
                self.x_f, self.t_f = self.sampler.update()
                self.x_f.requires_grad = True
                self.t_f.requires_grad = True        
        return loss, L_0, L_f, L_bc
    
    def set_curriculum_schedule(self, maxIter):
        if self.system == "convection":
            self.pde_param_schedule = torch.linspace(0, self.pde_beta, maxIter+1)
        elif self.system == "rd":
            self.pde_param_schedule = torch.linspace(0, self.pde_rho, maxIter+1)
        #to be implemented for other pdes
    
    def update_pde_param(self, it):
        if self.system == "convection":
            self.pde_beta = self.pde_param_schedule[it]
        elif self.system == "rd":
            self.pde_rho = self.pde_param_schedule[it]
        # to be implemented for other pdes.
    
    # Optimize parameters in a loop
    def train(self, nIter = 10000):
        if self.method == "curriculum_reg": 
            self.set_curriculum_schedule(maxIter=nIter)
        elif self.method in ["causal_pinn", "causal_pinn_uniform"]:
            self.max_allowed_iter = nIter//len(self.tol_list)
            print("Max allowed Iter", self.max_allowed_iter)
        
        self.dnn.train()
        
        pbar = trange(nIter+1)
        # Main training loop
        for it in pbar:
            
            if self.method == "curriculum_reg":
                self.update_pde_param(it=it)
                
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
                
                if self.method == "causal_r3":
                    dict_['beta'] = self.sampler.beta.item()
                elif self.method == "curriculum_reg":
                    if self.system == "convection":
                        dict_['pde_beta'] = self.pde_beta.item()
                    elif self.system == "rd":
                        dict_['pde_rho'] = self.pde_rho.item()
                elif self.method in ["causal_pinn", "causal_pinn_uniform"]:
                    W = self.temp_dict['W'].detach().cpu().numpy()
                    dict_['tol'] = self.tol_list[self.tol_pt].item()
                    dict_['min_W'] = np.min(W)
                        
                if self.use_lr_scheduler:
                    dict_['curr_lr'] = self.lr_scheduler.get_last_lr()
                pbar.set_postfix(dict_)
                
            if it % self.viz_every == 0:
                self.intermediate_visualization(epoch=it)
            
            #updating the tolerance for Causal PINNs
            if self.method in ["causal_pinn", "causal_pinn_uniform"]:
                self.tol_iter += 1
                W = self.temp_dict['W'].detach().cpu().numpy()
                if self.tol_iter >= self.max_allowed_iter or np.min(W)>= 0.99:
                    print(self.tol_iter, np.min(W))
                    self.tol_pt += 1
                    self.tol_iter = 0
                    print("Tolerance Value Updated.")
                    if self.tol_pt == len(self.tol_list):
                        if np.min(W)>= 0.99:
                            print("Stopping Criterion Enforced.")
                            break
                        else:
                            self.tol_pt -= 1 #letting it run till all the iterations is complete.
                
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
            viz_path_scatter = os.path.join(self.viz_dir, f"R3Sample_viz_{str(epoch).zfill(7)}.pdf")
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
        
        if self.method in ["causal_r3", "r3_sampling"]:
            with torch.no_grad():
                r_pred = self.temp_dict['r_pred']
                x_old, t_old, x_new, t_new = self.sampler.get_old_new(torch.abs(r_pred).detach())

            f_pred = np.absolute(self.predict_f(t=self.T_star, x=self.X_star))
            if self.method == "causal_r3":
                t_star = torch.tensor(self.T_star.reshape(-1,1), device=self.device)
                causal_gate = self.sampler.causal_gate(t_star).detach().cpu().numpy()
                fitness = f_pred * causal_gate

            #reshape
            f_pred = (f_pred.reshape(self.n_t, self.n_x)).T
            
            if self.method == "causal_r3":
                fitness = (fitness.reshape(self.n_t, self.n_x)).T
                causal_gate = (causal_gate.reshape(self.n_t, self.n_x)).T

                visualize_scatter(
                                  x_old=x_old, 
                                  t_old=t_old, 
                                  x_new=x_new, 
                                  t_new=t_new, 
                                  f_pred=f_pred, 
                                  fitness=fitness, 
                                  causal_gate=causal_gate,
                                  x_lim=self.x_lim,
                                  t_lim=self.t_lim,
                                  show_viz=self.show_viz,
                                  viz_path=viz_path_scatter
                                 )
                
                if self.viz_dir is not None:
                    scatter_plot_dict = {
                                         "x_new":x_new.detach().cpu().numpy(), "t_new":t_new.detach().cpu().numpy(), 
                                         "x_old":x_old.detach().cpu().numpy(), "t_old":t_old.detach().cpu().numpy(), 
                                         "f_pred":f_pred, "fitness":fitness, "causal_gate":causal_gate,
                                         "x_lim":self.x_lim, "t_lim":self.t_lim,
                                        }
                    savemat(os.path.join(self.viz_dir, f"scatter_dict_epoch_{str(epoch).zfill(7)}.mat"), scatter_plot_dict)

                #plotting the causal gate
                self.plot_causal_gate(epoch=epoch)
            
            elif self.method in ["r3_sampling"]:
                visualize_scatter_r3_sampling(
                                              x_old=x_old, 
                                              t_old=t_old, 
                                              x_new=x_new, 
                                              t_new=t_new, 
                                              f_pred=f_pred, 
                                              x_lim=self.x_lim,
                                              t_lim=self.t_lim,
                                              show_viz=self.show_viz,
                                              viz_path=viz_path_scatter
                                             )
                if self.viz_dir is not None:
                    scatter_plot_dict = {
                                         "x_new":x_new.detach().cpu().numpy(), "t_new":t_new.detach().cpu().numpy(), 
                                         "x_old":x_old.detach().cpu().numpy(), "t_old":t_old.detach().cpu().numpy(), 
                                         "f_pred":f_pred,
                                         "x_lim":self.x_lim, "t_lim":self.t_lim,
                                        }
                    savemat(os.path.join(self.viz_dir, f"scatter_dict_epoch_{str(epoch).zfill(7)}.mat"), scatter_plot_dict)
                
        
        elif self.method in ["causal_pinn", "causal_pinn_uniform"]:
            L_t = self.temp_dict['L_t'].detach().cpu().numpy()
            W = self.temp_dict['W'].detach().cpu().numpy()
            visualize_ltw(L_t, W, show_viz=self.show_viz, viz_path=viz_path_ltw)
            if self.viz_dir is not None:
                Ltw_dict = {
                            "L_t":L_t, "W":W, 
                            }
                savemat(os.path.join(self.viz_dir, f"ltw_epoch_{str(epoch).zfill(7)}.mat"), Ltw_dict)
    
    def update_tolerance(self, curr_Iter, max_Iter):
        update_step = max_Iter // len(self.tol_list)
        if curr_Iter % update_step == 0 and curr_Iter!=0:
            curr_idx = curr_Iter//update_step
            self.tol = self.tol_list[curr_idx]
            print(f"Causality Parameter Updated: {self.tol}")

        
    

    
        
    