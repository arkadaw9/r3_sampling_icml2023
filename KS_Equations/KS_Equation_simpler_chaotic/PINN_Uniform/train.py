import sys
import numpy as onp
import jax.numpy as np
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
from tqdm import trange

import scipy.io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from model import *
from dataset import *
from pinn import *
from utils import plot_results

import argparse
import os

parser = argparse.ArgumentParser(description='PINN with Uniform Sampling')
parser.add_argument('--M_t', type=int, default=6, help='')
parser.add_argument('--M_x', type=int, default=5, help='')
parser.add_argument('--t0', type=float, default=0.0, help='')
parser.add_argument('--t1', type=float, default=0.5, help='')
parser.add_argument('--N', type=int, default=5, help='time marching windows')
parser.add_argument('--N_r', type=int, default=8192, help='Number of collocation points')

parser.add_argument('--nIter', type=int, default=300000, help='number of iterations')
parser.add_argument('--experiment_name', type=str, default="experiment_1", help='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--decay_steps', type=int, default=5000, help='')
parser.add_argument('--decay_rate', type=float, default=0.9, help='')
parser.add_argument('--scheduler_type', type=str, default="exponential", help='exponential/step_lr')
parser.add_argument('--init_type', type=str, default="xavier_init", help='init_type: xavier_init/pytorch_init')

parser.add_argument('--lambda_ic', type=float, default=1e4, help='')
parser.add_argument('--lambda_f', type=float, default=1, help='')
parser.add_argument('--layers', type=str, default="128, 128, 128, 128, 128, 128, 128, 128, 1", help='Layer List')
args = parser.parse_args()

print(args)

args_dict = {
        "lr":args.lr,
        "decay_steps":args.decay_steps,
        "decay_rate":args.decay_rate,
        "scheduler_type":args.scheduler_type,
        "init_type":args.init_type,
        "lambda_ic":args.lambda_ic,
        "lambda_f":args.lambda_f,
        }



save_path = f"./results/{args.experiment_name}/TimeWindows_{args.N}/N_r_{args.N_r}_nIter_{args.nIter}_lr_{args.lr}_decay_steps_{args.decay_steps}_rate_{args.decay_rate}_type_{args.scheduler_type}_lambda_ic_{args.lambda_ic}_lambda_f_{args.lambda_f}_layers_{args.layers}_init_type_{args.init_type}_M_x_{args.M_x}_M_t_{args.M_t}/"

print("Save Path:", save_path)
print(os.path.exists(save_path))

if not os.path.exists(save_path):
    os.makedirs(save_path)

print("Training optimized script.")
# Load data
data = scipy.io.loadmat('../ks_simpler_chaotic.mat')
# Test data
usol = data['usol']


# Hpyer-parameters
key = random.PRNGKey(1234)

t1_sub = args.t1/args.N
layers = [int(item) for item in args.layers.split(',')]

# Initial state
state0 = usol[:, 0:1]
dt = 1 / 250
idx = int(t1_sub / dt)
t_star = data['t'][0][:idx]
x_star = data['x'][0]

args_dict['x_star'] = x_star
args_dict['t_star'] = t_star

t_star_full = data['t'][0]

arch = 'modified_MLP'
print('arch:', arch)


u_pred_list = []
params_list = []
losses_list = []


# Time marching
for k in range(args.N):
    # Initialize model
    print('Final Time: {}'.format((k + 1) * t1_sub))
    #reinitialize dataset every_time
    dataset = DataGenerator(args.t0, t1_sub, N_r=args.N_r)
    
    #for visualization
    usol_window = usol[:,k*idx:(k+1)*idx]
    args_dict['usol'] = usol_window
    
    args_dict['save_path'] = os.path.join(save_path, f"Window_ID_{k}")
    if not os.path.exists(args_dict['save_path']):
        os.makedirs(args_dict['save_path'])
        
    model = PINN(key=key, 
                 arch=arch, 
                 layers=layers, 
                 M_t=args.M_t, 
                 M_x=args.M_x,
                 state0=state0, 
                 t0=args.t0, 
                 t1=t1_sub, 
                 dataset=dataset, 
                 x_star=x_star, 
                 args_dict=args_dict)
    # Train
    model.train(nIter=args.nIter)
        
    # Store
    params = model.get_params(model.opt_state) 
    u_pred = model.u_pred_fn(params, t_star, x_star)
    u_pred_list.append(u_pred)
    flat_params, _  = ravel_pytree(params)
    params_list.append(flat_params)
    losses_list.append([model.loss_log, model.loss_ics_log, model.loss_res_log])
    

    np.save(os.path.join(save_path, 'Evo_u_pred_list.npy'), u_pred_list)
    np.save(os.path.join(save_path, 'Evo_params_list.npy'), params_list)
    np.save(os.path.join(save_path, 'Evo_losses_list.npy'), losses_list)
    
    # error 
    u_preds = np.hstack(u_pred_list)
    error = np.linalg.norm(u_preds - usol[:, :(k+1) * idx]) / np.linalg.norm(usol[:, :(k+1) * idx]) 
    print('Relative l2 error: {:.3e}'.format(error))
    
    params = model.get_params(model.opt_state)
    u0_pred = vmap(model.neural_net, (None, None, 0))(params, t1_sub, x_star)
    state0 = u0_pred
    
    N = (k+1) * idx
    plot_results(u_pred_list, usol, t_star_full, x_star, N, save_path)
