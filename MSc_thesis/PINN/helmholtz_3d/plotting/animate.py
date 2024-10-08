import os
import meshio
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Delaunay


def create_faces(points):
    delaunay = Delaunay(points)
    tri = delaunay.simplices # triangulation
    return tri

def make_3d_animation(mesh_file, mesh_coords, firing_rate_data, firing_rate_pred, time_data, path, cd, name, skip=1, save_anim=True):
    fig, axes = plt.subplots(3, 1, figsize=(20, 20))
    num_steps = firing_rate_data.shape[0]
    num_frames = num_steps // skip

    if mesh_file is not None:
        mesh = meshio.read(mesh_file)
        pos = mesh.points
    elif mesh_coords is not None:
        pos = mesh_coords
    faces = create_faces(pos[:, :2])

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

            ax.set_title('{} t={:.2f}s'.format(title, time_data[step]), fontsize='25')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            clb = fig.colorbar(mesh_plot, cax=cax, orientation='vertical')
            clb.ax.tick_params(labelsize=20)

            clb.ax.set_title('', fontdict={'fontsize': 15}, pad=20)
            count += 1
        return fig,

    if not os.path.exists(path):
        os.makedirs(path)

    if (save_anim):
        gs_anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
        writergif = animation.PillowWriter(fps=int(1e-1/cd.tr))
        anim_path = os.path.join(path, '{}_s_anim.gif'.format(name))
        gs_anim.save(anim_path, writer=writergif)
        plt.show(block=True)
    else:
        pass
