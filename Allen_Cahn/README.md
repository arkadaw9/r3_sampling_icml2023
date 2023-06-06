An example input command for each of the proposed r3 sampling and baselines for the Allen Cahn Equation have been provided below:

### R3-Sampling 
python train.py --method "r3_sampling" --lambda_ic 100. --lambda_f 1. --N_f 10000

### Causal R3-Sampling 
python train.py --method "causal_r3" --lambda_ic 100. --lambda_f 1. --N_f 10000

### Causal PINN
python train.py --method "causal_pinn" --lambda_ic 100. --lambda_f 1. --tol_list "1e-2,1e-1,1,1e1,1e2" --n_x 256 --n_t 100
python train.py --method "causal_pinn_uniform" --lambda_ic 100. --lambda_f 1. --tol_list "1e-2,1e-1,1,1e1,1e2" --n_x 256 --n_t 100

### PINN
python train.py --method "pinn" --lambda_ic 100. --lambda_f 1. --N_f 10000
python train.py --method "pinn_uniform" --lambda_ic 100. --lambda_f 1. --N_f 10000

### RAR/RAR-D/RAD/L-inf
python train_rar.py --method "rar_g" --lambda_ic 100. --lambda_f 1. --N_f 10000 --N 1 --N_s 100000 --k 100 --m 1