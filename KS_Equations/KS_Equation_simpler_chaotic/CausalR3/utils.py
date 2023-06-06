import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np
import os

def plot_results(u_pred_list, usol, t_star_full, x_star, N, save_path=None):
    u_pred = np.concatenate(u_pred_list, axis=1)
    TT, XX = np.meshgrid(t_star_full, x_star) #global variables
    
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(TT[:,:N], XX[:,:N], usol[:,:N], cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(r'Exact $u(x)$')
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(TT[:,:N], XX[:,:N], u_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(r'Predicted $u(x)$')
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT[:,:N], XX[:,:N], np.abs(usol[:,:N] - u_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Absolute error')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"plot_after_{N}.pdf"), dpi=150)
    else:
        plt.show()