import numpy as np
import os
import sys
import torch

def get_params(args, x_star, t_star, usol, device):   
    params = {
        "method":args.method,
        "model_params":{
                        "dnn_type":args.dnn_type,
                        "layers": args.layers,
                        "activation":args.activation,
                        "init":args.init,
                        },
        "train_params":{
                        "optimizer":torch.optim.Adamax,
                        "lr":args.lr,
                        "use_lr_scheduler":args.use_lr_scheduler, 
                        "lr_scheduler":torch.optim.lr_scheduler.StepLR,
                        "decay_rate":args.decay_rate,
                        "decay_steps":args.decay_steps,
                        "lambda_ic":args.lambda_ic,
                        "lambda_bc":args.lambda_bc,
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
                        "system":args.system,
                        "nu":args.pde_nu,
                        "beta":args.pde_beta,
                        "rho":args.pde_rho,
                        "state0":None,
                        "x_lim":None,
                        "t_lim":None,
                    },
        "sampler_params":{
                         "N_f":args.N_f,
                         "N_s":args.N_s,
                         "m":args.m,
                         "k":args.k,
                        },
        "dynamic_scheduler_params":{
                                "use_dynamic":args.use_dynamic,
                                "lr_chg_beta":args.lr_chg_beta,
                                "new_decay_steps":args.new_decay_steps,
                                },
        "io_params":{
                    "results_dir":args.results_dir
                    },
        "device":device
    }
    return params
