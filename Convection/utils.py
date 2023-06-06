import numpy as np
import random
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        

def plot_gate(t, gate, show_viz=True, viz_path=None):
    plt.figure(figsize=(4,2), dpi=150)
    plt.plot(t, gate, lw=2.)
    plt.ylim([0., 1.1])
    plt.grid("on", alpha=0.2)
    plt.title("Causal Gate")
    if show_viz:
        plt.show()
    if viz_path is not None:
        plt.savefig(viz_path)
    plt.close()

def visualize_scatter(x_old, t_old, x_new, t_new, f_pred, fitness, causal_gate, x_lim, t_lim, show_viz=True, viz_path=None): 
#     fig, ax = plt.subplots(1, 4, figsize=(20, 4), dpi=150)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    ax[0].scatter(t_new.detach().cpu().numpy(), x_new.detach().cpu().numpy(), color='b', s=2., alpha=0.5, label="Re-sampled points")
    ax[0].scatter(t_old.detach().cpu().numpy(), x_old.detach().cpu().numpy(), color='r', s=2., alpha=0.5, label="Retained points")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("x")
    ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
            mode="expand", borderaxespad=0, ncol=2, fontsize=8)

    h = ax[1].imshow(f_pred, interpolation='nearest', cmap='gray',
                extent=[t_lim[0], t_lim[1], x_lim[0],  x_lim[1]],
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    ax[1].set_title("PDE Residuals", fontsize=10)

    h = ax[2].imshow(fitness, interpolation='nearest', cmap='gray',
                extent=[t_lim[0], t_lim[1], x_lim[0], x_lim[1]],
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    ax[2].set_title("Fitness", fontsize=10)

    h = ax[3].imshow(causal_gate, interpolation='nearest', cmap='gray',
                extent=[t_lim[0], t_lim[1], x_lim[0], x_lim[1]],
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    ax[3].set_title("Causal Gate", fontsize=10)
    if show_viz:
        plt.show()
    if viz_path is not None:
        plt.savefig(viz_path)
    plt.close()

def visualize_scatter_r3_sampling(x_old, t_old, x_new, t_new, f_pred, x_lim, t_lim, show_viz=True, viz_path=None): 
#     fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(t_new.detach().cpu().numpy(), x_new.detach().cpu().numpy(), color='b', s=2., alpha=0.5, label="Re-sampled points")
    ax[0].scatter(t_old.detach().cpu().numpy(), x_old.detach().cpu().numpy(), color='r', s=2., alpha=0.5, label="Retained points")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("x")
    ax[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
            mode="expand", borderaxespad=0, ncol=2, fontsize=8)

    h = ax[1].imshow(f_pred, interpolation='nearest', cmap='gray',
                extent=[t_lim[0], t_lim[1], x_lim[0],  x_lim[1]],
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    ax[1].set_title("PDE Residuals", fontsize=10)

    if show_viz:
        plt.show()
    if viz_path is not None:
        plt.savefig(viz_path)
    plt.close()

def visualize(usol, u_pred, XX, TT, show_viz=True, viz_path=None):
#     fig = plt.figure(figsize=(18, 5), dpi=150)
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.pcolor(TT.T, XX.T, usol, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(r'Exact $u(x)$')
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(TT.T, XX.T, u_pred, cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title(r'Predicted $u(x)$')
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT.T, XX.T, np.abs(usol - u_pred), cmap='jet')
    plt.colorbar()
    plt.xlabel('$t$')
    plt.ylabel('$x$')
    plt.title('Absolute error')
    plt.tight_layout()
    if show_viz:
        plt.show()
    if viz_path is not None:
        plt.savefig(viz_path)
    plt.close()
    
def visualize_ltw(L_t, W, show_viz, viz_path):
#     fig, axes = plt.subplots(1,2,figsize=(8, 3), dpi=150)
    fig, axes = plt.subplots(1,2,figsize=(8, 3))
    ax = axes[0]
    ax.plot(L_t, lw=2.)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$L_t$')
    
    ax = axes[1]
    ax.plot(W, lw=2.)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$W$')
    
    if show_viz:
        plt.show()
    if viz_path is not None:
        plt.savefig(viz_path)
    plt.close()