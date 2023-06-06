import argparse
import numpy as np
import random

import torch
from torch.utils import data

import scipy.io
from scipy.interpolate import griddata
from utils import *
from pinn_model import PINN
from dataset import DatasetGenerator
from parameters import get_params
import os
from scipy.io import savemat

################
# Arguments
################
parser = argparse.ArgumentParser(description='R3 Sampling for PINNs')

parser.add_argument('--seed', type=int, default=1234, help='Random initialization.')
parser.add_argument('--method', type=str, default="r3_sampling", help='Type of method: r3_sampling, causal_r3, pinn, pinn_uniform, causal_pinn, causal_pinn_uniform, curriculum_reg')
parser.add_argument('--dnn_type', type=str, default="mlp", help='Type of method: mlp, modified_mlp')

#Model Params
parser.add_argument('--layers', type=str, default="128, 128, 128, 128, 1", help='Layer List')
parser.add_argument('--M', type=int, default=10, help='For Fourier Mapping')
parser.add_argument('--L', type=float, default=2., help='For Fourier Mapping')
parser.add_argument('--activation', type=str, default="tanh", help='Activation Function')

#train params
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--use_lr_scheduler', type=str, default="True", help='Use a learning rate scheduler')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay Rate of Learning Rate Scheduler')
parser.add_argument('--decay_steps', type=float, default=5000, help='Decay Steps of Learning Rate Scheduler')
parser.add_argument('--lambda_ic', type=float, default=1., help='Lambda for initial condition')
parser.add_argument('--lambda_f', type=float, default=1., help='Lambda for residual loss')
parser.add_argument('--max_iter', type=int, default=200000, help='Max iterations')

#visualization params
parser.add_argument('--update_logs_every', type=int, default=50, help='Update logs')
parser.add_argument('--viz_every', type=int, default=10000, help='Visualization every')
parser.add_argument('--show_viz', type=str, default="False", help='Visualize the solution.')

#sampler_params
parser.add_argument('--N_f', type=int, default=1000, help='Number of collocation points to sample.')
parser.add_argument('--N', type=int, default=1, help='Number of time windows.')

#causal_gate_params
parser.add_argument('--beta', type=float, default=-0.5, help='Shift parameter for gate')
parser.add_argument('--alpha', type=float, default=5.0, help='Steepness parameter for gate')
parser.add_argument('--gate_type', type=str, default="tanh", help='Gate Function: tanh, relu_tanh')
parser.add_argument('--beta_lr', type=float, default=1e-3, help='Beta learning rate')
parser.add_argument('--tol', type=float, default=20.0, help='tolerance value of gate')
parser.add_argument('--grad_clip', type=float, default=1e-1, help='gradient clipping')

#causal pinns params
parser.add_argument('--tol_list', type=str, default="1e-2,1e-1,1,1e1,1e2", help='List of tolerance values for Causal PINNs')
parser.add_argument('--n_x', type=int, default=64, help='Number of points in the x domain')
parser.add_argument('--n_t', type=int, default=32, help='Number of points in the t domain')

parser.add_argument('--results_dir', type=str, default="./results/", help='Gate Function')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')

#dynamic lr change scheduler
parser.add_argument('--use_dynamic', type=str, default="False", help='Dynamically Changing beta')
parser.add_argument('--lr_chg_beta', type=float, default=1.4, help='When the new lr scheduler would take place')
parser.add_argument('--new_decay_steps', type=float, default=5000, help='Decay Steps of new Learning Rate Scheduler')
args = parser.parse_args()

args.use_dynamic = True if args.use_dynamic=="True" else False
args.use_lr_scheduler = True if args.use_lr_scheduler=="True" else False
args.show_viz = True if args.show_viz=="True" else False

set_seed(args.seed)

# CUDA support
if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.gpu_id}')
else:
    device = torch.device('cpu')

print("Device Initialized.")


#Results dir
sub_dir = f"Seed_{args.seed}_{args.dnn_type}_layers_{args.layers}_activation_{args.activation}_L_{args.L}_M_{args.M}/lr_{args.lr}_uselrsch_{args.use_lr_scheduler}_decayrate_{args.decay_rate}_decaysteps_{args.decay_steps}/maxiter_{args.max_iter}_lambda_ic_{args.lambda_ic}_lambda_f_{args.lambda_f}/{args.method}_timewindow_{args.N}/"
if args.method == "causal_r3":
    subsub_dir = f"Nf_{args.N_f}_beta_{args.beta}_alpha_{args.alpha}_gate_type_{args.gate_type}_beta_lr_{args.beta_lr}_tol_{args.tol}_grad_clip_{args.grad_clip}/"
elif args.method in ["causal_pinn", "causal_pinn_uniform"]:
    subsub_dir = f"tol_list_{args.tol_list}_nx_{args.n_x}_nt_{args.n_t}/"
elif args.method in ["r3_sampling", "curriculum_reg", "pinn_uniform", "pinn"]:
    subsub_dir = f"N_f_{args.N_f}"

args.layers = [int(item) for item in args.layers.split(',')]
args.tol_list = [float(item) for item in args.tol_list.split(',')]

base_dir = os.path.join(args.results_dir,sub_dir,subsub_dir)
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

#Processing the data
data = scipy.io.loadmat('../data/AC.mat')
usol = data['uu']

# Grid
t_star = data['tt'][0]
x_star = data['x'][0]
TT, XX = np.meshgrid(t_star, x_star)

# Reference solution
plt.figure(figsize=(5, 4), dpi=150)
plt.pcolor(TT, XX, usol, cmap='jet')
plt.colorbar()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.savefig(os.path.join(base_dir,"reference_solution.pdf"))
plt.close()

print("Data Loading Complete.")

params = get_params(args, x_star, t_star, usol, device)
print(params['model_params'])
print(params['train_params'])
print(params['causal_gate_params'])
print(params['sampler_params'])

def evaluate_model(x_star, t_star, model):
    # Get trained network parameters
    X, T = np.meshgrid(x_star, t_star) # all the X grid points T times, all the T grid points X times
    n_x = x_star.shape[0]
    n_t = t_star.shape[0]

    u_pred = model.predict(T.flatten(), X.flatten())
    u_pred = u_pred.reshape(n_t, n_x)
    u_pred = u_pred.T

    error = np.linalg.norm(u_pred - usol) / np.linalg.norm(usol) 
    print('Relative l2 error: {:.3e}'.format(error))
    return error, u_pred

dataset = DatasetGenerator(x_star, t_star, usol, args.N)
error_log = []
u_pred_list = []

for idx in range(1, args.N + 1):
    print(f"Training on Time Domain: {idx}")
    time_window_dr = os.path.join(base_dir, f"Time_window_{idx}")
    if not os.path.exists(time_window_dr):
        os.makedirs(time_window_dr)
        
    state0, t_star, usol, t_lim, x_lim = dataset.__get_item__(idx)
    params['pde_params']["state0"] = state0
    params['pde_params']["x_lim"] = x_lim
    params['pde_params']["t_lim"] = t_lim
    
    params['viz_params']['t_star'] = t_star 
    params['viz_params']['usol'] = usol
    params['viz_params']['viz_dir'] = time_window_dr
    
    if idx > 1:
        params['pde_params']['state0'] = new_state0
    
    model = PINN(params)
    # Train
    model.train(nIter=params['train_params']['max_iterations'])
    
    #saving logs
    log_dict = {
        "loss":np.array(model.loss_log),
        "ics_loss":np.array(model.loss_ics_log),
        "res_loss":np.array(model.loss_res_log),
        "abs_err":np.array(model.abs_err_log),
        "l2_err":np.array(model.rel_l2_err_log),
    }
    
    savemat(os.path.join(time_window_dr,"loss_log.mat"), log_dict)
    
    torch.save(model.dnn, os.path.join(time_window_dr, f"model_time_domain_{idx}.pth"))
    error, u_pred = evaluate_model(x_star, t_star, model)
    savemat(os.path.join(time_window_dr,"model_pred.mat"), {"error":error, "u_pred":u_pred})
    new_state0 = u_pred[:,-1].reshape(-1,1)
    
    error_log.append(error)
    u_pred_list.append(u_pred_list)

print(f"Error: {error_log}")
np.save(os.path.join(base_dir, "error_list.npy"), error_log)
np.save(os.path.join(base_dir, "u_pred_list.npy"), u_pred_list)


