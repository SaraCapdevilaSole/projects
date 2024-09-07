import os
import meshio
import numpy as np
import jax.numpy as jnp

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
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

def plot_3d_parcels(dt, coords, u_ref, u_pred, u_error, save_dir, test=False):
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

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
        fig3 = plt.figure(figsize=(10, 23))
        data_fig = data[n]
        
        # Loop to create subplots
        for i, (u_data, title, time_frame) in enumerate(data_fig, start=1):
            ax = fig3.add_subplot(1, 3, i, projection='3d')
            scatter = ax.scatter(x, y, z, c=u_data, cmap='coolwarm')
            
            if time_frame is not None:
                ax.set_title(f'{title}', fontsize=title_fontsize, y=1.15)
            else:
                ax.set_title(f'{title} Average', fontsize=title_fontsize, y=1.15)
            
            plt.grid(False)
            plt.axis('off')
            
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)  # Force scientific notation
            cbar.ax.yaxis.set_major_formatter(formatter)
        
        plt.tight_layout()
        
        fname = f"3d_plot_{n}"
        if test:
            fname += '_test'
        fig_path = os.path.join(save_dir, fname + '.pdf')
        fig3.savefig(fig_path, bbox_inches="tight", dpi=300)

def create_faces(points):
    delaunay = Delaunay(points)
    tri = delaunay.simplices # triangulation
    return tri

def plot_3d_snapshot(mesh_file, mesh_coords, firing_rate_data, firing_rate_pred, time_data, path, cd, name):
    indices = [0, firing_rate_data.shape[0]//2, -1]
    
    if mesh_file is not None:
        mesh = meshio.read(mesh_file)
        pos = mesh.points
    elif mesh_coords is not None:
        valid_indices = mesh_coords[:, 1] < 20
        pos = mesh_coords[valid_indices, :]
        firing_rate_data = firing_rate_data[:, valid_indices]
        firing_rate_pred = firing_rate_pred[:, valid_indices]

    faces = create_faces(pos[:, :2])
    firing_rate_error = np.abs(firing_rate_data - firing_rate_pred)

    for idx in indices:
        fig, axes = plt.subplots(1, 3, figsize=(25, 7))
        plt.subplots_adjust(wspace=-0.1)

        step = idx % firing_rate_data.shape[0]

        rate_actual = firing_rate_data[step]
        rate_pred = firing_rate_pred[step]
        abs_error = firing_rate_error[step]

        for count, ax in enumerate(axes):
            ax.cla()
            ax.set_aspect('equal')
            ax.set_axis_off()

            if count == 0:
                phi = rate_actual
                title = 'Reference:'
                vmin = rate_actual.min()
                vmax = rate_actual.max()
            elif count == 1:
                phi = rate_pred
                title = 'Prediction:'
                vmin = rate_pred.min()
                vmax = rate_pred.max()
            else:
                phi = abs_error
                title = 'Error:'
                vmin = abs_error.min()
                vmax = abs_error.max()

            triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
            mesh_plot = ax.tripcolor(triang, phi, vmin=vmin, vmax=vmax, shading='flat', cmap='coolwarm') 
            # mesh_plot = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=phi, vmin=vmin, vmax=vmax, cmap='coolwarm') 
            ax.set_title('{} t={:.2f}s'.format(title, time_data[step]), fontsize='25', pad=20)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.3)
            clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
            clb.ax.tick_params(labelsize=20)
            clb.ax.set_title('', fontdict={'fontsize': 15}, pad=20)

            tick_positions = np.linspace(vmin, vmax, 3)
            clb.set_ticks(tick_positions)

            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 3))
            clb.ax.yaxis.set_major_formatter(formatter)
            clb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            clb.ax.yaxis.get_offset_text().set_size(15)

        fig_path = os.path.join(path, '{}_snapshot_{}.png'.format(name, step))
        plt.savefig(fig_path)
        plt.clf()
        
    plt.close()