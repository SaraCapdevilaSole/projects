import os
import numpy as np
import jax.numpy as jnp

from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import Delaunay


def plot_fig_tgt(t_star, u_pred, u_ref, u_error, save_dir, p=100, test=False):
    fig = plt.figure(figsize=(7, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t_star, u_pred[:, p], label="predicted", color="#FF5733")
    plt.plot(t_star, u_ref[:, p], label="reference", color="#1893BB")
    plt.ylabel('Solution')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.title(r'Real and Predicted $\varphi_e$')
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plt.plot(t_star, u_error[:, p])
    plt.title('Relative error of $u$')
    plt.ylabel('Solution')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

    fname = 'u_error'
    if test:
        fname += '_test'
    save_path = os.path.join(save_dir, fname + '.pdf')
    fig.savefig(save_path, bbox_inches="tight", dpi=300)

def plot_fig_separate(t_star, u_pred, u_ref, u_error, save_dir, p=10, test=False):
    fig = plt.figure(figsize=(18, 12))
    plt.subplot(3, 1, 1)
    plt.plot(t_star, u_ref[:, :p])
    plt.title(r'Real $\varphi_e$')
    plt.ylabel('Solution')
    plt.xlabel('Time (s)')
    plt.tight_layout()

    plt.subplot(3, 1, 2)
    plt.plot(t_star, u_pred[:, :p])
    plt.title(r'Predicted $\varphi_e$')
    plt.ylabel('Solution')
    plt.xlabel('Time (s)')
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.plot(t_star, jnp.mean(u_error, axis=1))
    plt.title(r'Absolute error of $\varphi_e$')
    plt.ylabel('Solution')
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

    fname = f"first_{p}_parcells_u_error"
    if test:
        fname += '_test'
    save_path = os.path.join(save_dir, fname + '.pdf')
    fig.savefig(save_path, bbox_inches="tight", dpi=300)

def plot_time_space_cross_section(t_star, coords, u_ref, u_pred, u_error, save_dir, test=False):
    TT, XX = np.meshgrid(t_star, coords[:, 0], indexing="ij")

    fig = plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.pcolor(TT, XX, u_ref, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.pcolor(TT, XX, u_pred, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.pcolor(TT, XX, u_error, cmap="jet")
    plt.colorbar()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("Absolute error")
    plt.tight_layout()

    fname = "time_space_cross_section"
    if test:
        fname += '_test'
    fig_path = os.path.join(save_dir, fname + '.pdf')
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)

def create_faces(points):
    delaunay = Delaunay(points)
    tri = delaunay.simplices # triangulation
    return tri

def plot_2d_parcels(dt, coords, u_ref, u_pred, u_error, t_array, save_dir, test=False):
    x = coords[:, 0]
    y = coords[:, 1]

    files_frames = [0, -1]
    time_frames = [t_array[fl] for fl in files_frames]

    data = [
        [(u_ref[files_frames[0], :], 'Reference', time_frames[0]),
         (u_pred[files_frames[0], :], 'Predicted', time_frames[0]),
         (u_error[files_frames[0], :], 'Absolute Error', time_frames[0])],
        [(u_ref[files_frames[1], :], 'Reference', time_frames[1]),
         (u_pred[files_frames[1], :], 'Predicted', time_frames[1]),
         (u_error[files_frames[1], :], 'Absolute Error', time_frames[1])],
        [(np.mean(u_ref, axis=0), 'Reference', None),
         (np.mean(u_pred, axis=0), 'Predicted', None),
         (np.mean(u_error, axis=0), 'Absolute Error', None)]
    ]

    all_data = np.concatenate([u_ref, u_pred], axis=0)
    vmin, vmax = all_data.min(), all_data.max()

    vmin_e, vmax_e = u_error.min(), u_error.max()

    title_fontsize = 14

    points = np.vstack((x, y)).T
    faces = create_faces(points)

    for n in range(3):
        fig3 = plt.figure(figsize=(15, 5))
        data_fig = data[n]
        
        for i, (u_data, title, time_frame) in enumerate(data_fig, start=1):
            ax = fig3.add_subplot(1, 3, i)
            triang = mtri.Triangulation(x, y, faces)
            if i==3:
                vmin_p, vmax_p = vmin_e, vmax_e
            else:
                vmin_p, vmax_p = vmin, vmax

            contour = ax.tricontourf(triang, u_data, cmap='coolwarm', vmin=vmin_p, vmax=vmax_p)
            
            if time_frame is not None:
                ax.set_title(f'{title} at t={time_frame:.2f}s', fontsize=title_fontsize)
            else:
                ax.set_title(f'{title} Average', fontsize=title_fontsize)
            
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)  

            cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)  
            cbar.ax.yaxis.set_major_formatter(formatter)
        
        plt.tight_layout()
        
        fname = f"2d_plot_{n}"
        if test:
            fname += '_test'
        fig_path = os.path.join(save_dir, fname + '.pdf')
        fig3.savefig(fig_path, bbox_inches="tight", dpi=300)


def plot_2d_parcels_backup(dt, coords, u_ref, u_pred, u_error, save_dir, test=False):
    x = coords[:, 0]
    y = coords[:, 1]
    # z = coords[:, 2]

    time_frames = [2, 20]
    files_frames = [int(tr/dt) for tr in time_frames]

    data = [
        [(u_ref[files_frames[0], :], 'Reference', time_frames[0]),
         (u_pred[files_frames[0], :], 'Predicted', time_frames[0]),
         (u_error[files_frames[0], :], 'Absolute Error', time_frames[0])],
        [(u_ref[files_frames[1], :], 'Reference', time_frames[1]),
         (u_pred[files_frames[1], :], 'Predicted', time_frames[1]),
         (u_error[files_frames[1], :], 'Absolute Error', time_frames[1])],
        [(np.mean(u_ref, axis=0), 'Reference', None),
         (np.mean(u_pred, axis=0), 'Predicted', None),
         (np.mean(u_error, axis=0), 'Absolute Error', None)]
    ]

    title_fontsize = 14

    for n in range(3):
        fig3 = plt.figure(figsize=(15, 5))
        data_fig = data[n]
        
        for i, (u_data, title, time_frame) in enumerate(data_fig, start=1):
            ax = fig3.add_subplot(1, 3, i)
            scatter = ax.scatter(x, y, c=u_data, cmap='coolwarm')
            
            if time_frame is not None:
                ax.set_title(f'{title} at t={time_frame:.3f}s', fontsize=title_fontsize)
            else:
                ax.set_title(f'{title} Average', fontsize=title_fontsize)
            
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)  
            
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)  
            cbar.ax.yaxis.set_major_formatter(formatter)
        
        plt.tight_layout()
        
        fname = f"3d_plot_{n}"
        if test:
            fname += '_test'
        fig_path = os.path.join(save_dir, fname + '.pdf')
        fig3.savefig(fig_path, bbox_inches="tight", dpi=300)