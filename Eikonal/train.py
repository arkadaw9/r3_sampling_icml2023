import os
import torch
import argparse

from r3_sampling.trainer import *
from r3_sampling.utils import set_seed
set_seed(1234)


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, 
                    help='Name for the experiment (same for data loading and output results).')
parser.add_argument('--model', type=str, choices=['pinn', 'r3_sampling', 'imp', "upinn", "uimp"],
                    help="Model name, can be 'pinn', 'r3_sampling' or 'imp'.")
parser.add_argument('--epochs', type=int, default=200000,
                    help='Number of Training Epochs.')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU device id.')
parser.add_argument('--N', type=int, default=100,
                    help='Number of collocation points.')
parser.add_argument('--out_root_dir', type=str, default="results/",
                    help='Output root directory.')
parser.add_argument('--data_root_dir', type=str, default="data/",
                    help='Data root directory.')
args = parser.parse_args()
print("[INFO] %s" % args)


# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda:%d' % args.gpu)
else:
    device = torch.device('cpu')
    
out_dir = os.path.join(args.out_root_dir, args.name + "_" + args.model + "_" + str(args.N))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
# load the data
data = torch.load(os.path.join(args.data_root_dir, args.name + ".pt"))


# initialize trainer
if args.model == 'r3_sampling':
    trainer = R3Trainer
elif args.model == 'pinn':
    trainer = PINNTrainer
elif args.model == 'imp':
    trainer = IMPTrainer
elif args.model == 'upinn':
    trainer = uPINNTrainer
elif args.model == 'uimp':
    trainer = uIMPTrainer
else:
    raise RuntimeError("Model has to be ['pinn', 'evo', 'imp'].")
    
    
# Train    
model = trainer(
    data, 
    out_dir=out_dir, 
    N_f=args.N, 
    device=device,
    layers=[2, 128, 128, 128, 128, 1],
    x_lim=(-1, 1),
    y_lim=(-1, 1)
)
model.train(nIter=args.epochs)

