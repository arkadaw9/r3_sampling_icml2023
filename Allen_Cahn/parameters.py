import numpy as np
import os
import sys
import torch

def get_params(args, x_star, t_star, usol, device):
    d0 = args.M * 2 + 2 # for fourier mapping
    if 'n' not in vars(args):
        args.n = args.N_f
    params = {
        "method":args.method,
        "model_params":{
                        "dnn_type":args.dnn_type,
                        "layers":[d0] + args.layers,
                        "M":args.M,
                        "activation":args.activation,
                        "L":args.L,
                        },
        "train_params":{
                        "optimizer":torch.optim.Adam,
                        "lr":args.lr,
                        "use_lr_scheduler":args.use_lr_scheduler, 
                        "lr_scheduler":torch.optim.lr_scheduler.StepLR,
                        "decay_rate":args.decay_rate,
                        "decay_steps":args.decay_steps,
                        "lambda_ic":args.lambda_ic,
                        "lambda_f":args.lambda_f,
                        "max_iterations":args.max_iter
                        },
        "viz_params":{
                        "x_star":x_star,
                        "t_star":t_star,
                        "usol":usol,
                        "update_logs_every":args.update_logs_every,
                        "viz_every":args.viz_every,
                        "show_viz":args.show_viz,
                        "viz_dir":None
                    },
        "pde_params":{
                        "state0":None,
                        "x_lim":None,
                        "t_lim":None,
                    },
        "sampler_params":{
                         "N_f":args.N_f,
                         "n":args.n,
                        },
        "causal_gate_params":{
                            "beta":args.beta,
                            "alpha":args.alpha,
                            "gate_type":args.gate_type,
                            "beta_lr":args.beta_lr,
                            "tol":args.tol,
                            "grad_clip":args.grad_clip,
                            "use_grad_clip":True #not implemented
                            },
        "dynamic_scheduler_params":{
                                "use_dynamic":args.use_dynamic,
                                "lr_chg_beta":args.lr_chg_beta,
                                "new_decay_steps":args.new_decay_steps,
                                },
        "causal_pinn_params":{
                            "tol_list":args.tol_list, 
                            "n_x":args.n_x,
                            "n_t":args.n_t
                        },
        "io_params":{
                    "results_dir":args.results_dir
                    },
        "device":device
    }
    return params
