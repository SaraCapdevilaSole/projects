import os
import meshio
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay

# TODO: change to Axes3D and use view_init

def create_faces(points):
    delaunay = Delaunay(points)
    tri = delaunay.simplices # triangulation
    return tri

def make_2d_animation(mesh_file, mesh_coords, firing_rate_data, firing_rate_pred, time_data, path, cd, name, skip=1, save_anim=True):
    fig, axes = plt.subplots(3, 1, figsize=(20, 20))
    num_steps = firing_rate_data.shape[0]
    num_frames = num_steps // skip

    if mesh_file is not None:
        mesh = meshio.read(mesh_file)
        pos = mesh.points
    elif mesh_coords is not None:
        pos = mesh_coords
    #faces = mesh.cells[0].data#[:int(cd.parcellations_to_use*2):cd.use_every_voxel] / pos.shape[0]
    # pos = pos[:cd.parcellations_to_use:cd.use_every_voxel]
    faces = create_faces(pos)

    firing_rate_error = np.abs(firing_rate_data - firing_rate_pred)

    def animate(num):
        step = (num*skip) % num_steps

        rate_actual = firing_rate_data[step]
        rate_pred = firing_rate_pred[step]
        abs_error = firing_rate_error[step]

        count = 0

        for ax in axes:
            ax.cla()
            ax.set_aspect('equal')
            ax.set_axis_off()

            if (count == 0):
                # ground truth
                phi = rate_actual
                title = 'Reference:'
                vmin = firing_rate_data.min()
                vmax = firing_rate_data.max()
            elif (count == 1):
                phi = rate_pred
                title = 'Prediction:'
                vmin = firing_rate_data.min()
                vmax = firing_rate_data.max()
            else:
                phi = abs_error
                title = 'Error:'
                vmin = firing_rate_error.min()
                vmax = firing_rate_error.max()

            triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
            mesh_plot = ax.tripcolor(triang, phi, vmin=vmin, vmax=vmax, shading='flat', cmap='coolwarm') 
            # ax.triplot(triang, 'ko-', ms=0, lw=0.01, alpha=0.01)
            # ax.view_init(elev=30, azim=1*step)

            ax.set_title('{} t={:.2f}s'.format(title, time_data[step]), fontsize='25')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
            clb.ax.tick_params(labelsize=20)

            clb.ax.set_title('', fontdict={'fontsize': 15}, pad=20)
            count += 1
        return fig,

    # Save animation for visualization
    if not os.path.exists(path):
        os.makedirs(path)

    if (save_anim):
        gs_anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
        writergif = animation.PillowWriter(fps=1)
        anim_path = os.path.join(path, '{}_s_anim.gif'.format(name))
        gs_anim.save(anim_path, writer=writergif)
        plt.show(block=True)
    else:
        pass

def make_3d_animation_backup(mesh_file, firing_rate_data, firing_rate_pred, time_data, path, name, skip=1, save_anim=True):
    mesh = meshio.read(mesh_file)
    points = mesh.points
    cells = mesh.cells[0].data 

    fig = plt.figure(figsize=(10, 24))
    ax1 = fig.add_subplot(311, projection='3d')
    ax2 = fig.add_subplot(312, projection='3d')
    ax3 = fig.add_subplot(313, projection='3d')
    
    num_steps = firing_rate_data.shape[0]
    num_frames = num_steps // skip

    norm = plt.Normalize(vmin=firing_rate_data.min(), vmax=firing_rate_data.max())
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])  

    norm_abs_error = plt.Normalize(vmin=0, vmax=np.abs(firing_rate_data - firing_rate_pred).max())
    sm_abs_error = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm_abs_error)
    sm_abs_error.set_array([])

    def animate(num):
        for ax in [ax1, ax2, ax3]:
            ax.cla()
            ax.set_aspect('auto')
            ax.set_axis_off()

        step = (num * skip) % num_steps
        rate_actual = firing_rate_data[step]
        rate_pred = firing_rate_pred[step]
        abs_error = np.abs(rate_actual - rate_pred)

        triang = mtri.Triangulation(points[:, 0], points[:, 1], cells)

        ax1.plot_trisurf(triang, points[:, 2], cmap='gray', edgecolor=None, alpha=0.7)
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=rate_actual, cmap='coolwarm')
        ax1.set_title(f'Reference Rate\nt = {time_data[step]:.3f}s', fontsize=15)

        ax2.plot_trisurf(triang, points[:, 2], cmap='gray', edgecolor=None, alpha=0.7)
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=rate_pred, cmap='coolwarm')
        ax2.set_title(f'Predicted Rate\nt = {time_data[step]:.3f}s', fontsize=15)

        ax3.plot_trisurf(triang, points[:, 2], cmap='gray', edgecolor=None, alpha=0.7)
        ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c=abs_error, cmap='coolwarm')
        ax3.set_title(f'Absolute Error\nt = {time_data[step]:.3f}s', fontsize=15)

        ax1.view_init(elev=30, azim=1*step)
        ax2.view_init(elev=30, azim=1*step)
        ax3.view_init(elev=30, azim=1*step)

        return fig,

    cbar1 = fig.colorbar(sm, ax=[ax1], fraction=0.05, pad=0.04, aspect=10)
    cbar1.set_label('', fontsize=15, rotation=0, labelpad=15)

    cbar2 = fig.colorbar(sm, ax=[ax2], fraction=0.05, pad=0.04, aspect=10)
    cbar2.set_label('', fontsize=15, rotation=0, labelpad=15)

    cbar_error = fig.colorbar(sm_abs_error, ax=[ax3], fraction=0.05, pad=0.04, aspect=10)
    cbar_error.set_label('', fontsize=15, rotation=0, labelpad=15)

    if not os.path.exists(path):
        os.makedirs(path)

    if save_anim:
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=10)
        writergif = animation.PillowWriter(fps=200)
        anim_path = os.path.join(path, f'{name}_anim.gif')
        anim.save(anim_path, writer=writergif)
        plt.show(block=True)
    else:
        plt.show()