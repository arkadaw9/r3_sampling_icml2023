An example input command for each of the proposed r3 sampling and baselines for the Convection Equation have been provided below:

### R3-Sampling 
python train.py --method "r3_sampling" --lambda_ic 100. --lambda_f 1. --lambda_bc 100. --N_f 1000 --u0_str "sin(x)"

### Causal R3-Sampling 
python train.py --method "causal_r3" --lambda_ic 100. --lambda_f 1. --lambda_bc 100. --N_f 1000 --u0_str "sin(x)"

### PINN (fixed)
python train.py --method "pinn" --lambda_ic 100. --lambda_f 1. --lambda_bc 100. --N_f 1000 --u0_str "sin(x)" 

### PINN (uniform)
python train.py --method "pinn_uniform" --lambda_ic 100. --lambda_f 1. --lambda_bc 100. --N_f 1000 --u0_str "sin(x)" 

### Curriculum Regularization
python train.py --method "curriculum_reg" --lambda_ic 1. --lambda_f 1. --lambda_bc 1. --N_f 1000 --u0_str "sin(x)" 

### Causal PINNs
python train.py --method "causal_pinn" --lambda_ic 100. --lambda_f 1. --lambda_bc 100. --u0_str "sin(x)" --tol_list "1e-2,1e-1,1,1e1,1e2" --n_x 32 --n_t 32

### RAR-G/RAD/RAR-D/L-inf
python train_rar.py --method "rar_g" --lambda_ic 100. --lambda_f 1. --lambda_bc 100. --N_f 1000 --N_s 100000 --k 100 --m 10 --u0_str "sin(x)"