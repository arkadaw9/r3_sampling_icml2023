# Mitigating Propagation Failures in Physics-informed Neural Networks using Retain-Resample-Release (R3) Sampling

This is the official Repository of [[paper]](https://arxiv.org/abs/2207.02338) (published at ICML 2023)

To cite this work, please use the following:
```bibtex
@misc{daw2023mitigating,
      title={Mitigating Propagation Failures in Physics-informed Neural Networks using Retain-Resample-Release (R3) Sampling}, 
      author={Arka Daw and Jie Bu and Sifan Wang and Paris Perdikaris and Anuj Karpatne},
      year={2023},
      eprint={2207.02338},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Training Instructions

The implementation for the Allen Cahn Equation, Convection Equation and Eikonal Equations are in PyTorch, while the Kuramoto-Sivashinsky Equations (for regular and chaotic regimes) are in JAX. Example training commands for each one of them can be found in the README.md files in the respective folders.

### Allen-Cahn Equation
Run the training script in PyTorch:
```
python ./Allen_Cahn/train.py --method [method name] --lambda_ic [lambda ic] --lambda_f [lambda pde] --N_f [number of collocations] --gpu_id [GPU device id]
```
The visualizations, model checkpoints and log for the experiments will be stored in `./Allen_Cahn/results/...`.

### Convection Equation
Run the training script in PyTorch:
```
python ./Convection/train.py --method [method name] --lambda_ic [lambda ic] --lambda_f [lambda pde] --lambda_bc [lambda bc] --N_f [number of collocations] --u0_str "sin(x)" --gpu_id [GPU device id]
```
Note: The initial condition for the Convection Equation is chosen to be sin(x).

The visualizations, model checkpoints and log for the experiments will be stored in `./Convection/results/...`.

### Eikonal Equation
Run the training script in PyTorch:
```
python ./Eikonal/train.py --method [method name] --name [name of data] ---N [number of collocations] -gpu [GPU device id]
```
E.g., name = "gear" will use the gear.pt file from the data.
The visualizations, model checkpoints and log for the experiments will be stored in `./Eikonal/results/...`.

### Kuramoto-Sivasinsky Equation
Run the training script in JAX:
```
python ./KS_Equations/KS_Equation_[regime]/[method]/train.py --N [number of time windows] --N_r [number of collocation] --experiment_name [experiment name]
```
The visualizations, model checkpoints and log for the experiments will be stored in `./KS_Equations/KS_Equation_[regime]/[method]/results/...`.

### Optimization behavior of R3-Sampling
The empirical validation of the Retain Property of R3-sampling was shown using different optimization functions. The notebook in the folder `./R3_Test_Optimization/` can be run to reproduce the results.



## Acknowledgements
The implementation for the Convection Equation is borrowed from the Github repository: https://github.com/a1k12/characterizing-pinns-failure-modes.
```bibtex
@article{krishnapriyan2021characterizing,
  title={Characterizing possible failure modes in physics-informed neural networks},
  author={Krishnapriyan, Aditi and Gholami, Amir and Zhe, Shandian and Kirby, Robert and Mahoney, Michael W},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={26548--26560},
  year={2021}
}
```
The implementation for the Causal PINNs is borrowed from the Github repository: https://github.com/PredictiveIntelligenceLab/CausalPINNs.
```bibtex
@article{wang2022respecting,
  title={Respecting causality is all you need for training physics-informed neural networks},
  author={Wang, Sifan and Sankaran, Shyam and Perdikaris, Paris},
  journal={arXiv preprint arXiv:2203.07404},
  year={2022}
}
```
