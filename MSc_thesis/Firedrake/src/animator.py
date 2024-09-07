import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.animation as animation
import os
import meshio

from config import EamonConfig

c = EamonConfig()

def make_3d_animation(mesh_file, firing_rate_file, path, name, skip=2, save_anim=True):
    mesh = meshio.read(mesh_file)
    points = mesh.points
    cells = mesh.cells[0].data 

    data = np.load(firing_rate_file, allow_pickle=True).item()
    firing_rate_data = data['phi_e']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    num_steps = firing_rate_data.shape[0]
    num_frames = num_steps // skip

    norm = plt.Normalize(vmin=firing_rate_data.min(), vmax=firing_rate_data.max())
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])  

    def animate(num):
        ax.cla()
        ax.set_aspect('auto')
        ax.set_axis_off()

        step = (num * skip) % num_steps
        rate = firing_rate_data[step]

        triang = mtri.Triangulation(points[:, 0], points[:, 1], cells)
        ax.plot_trisurf(triang, points[:, 2], cmap='gray', edgecolor=None, alpha=0.7)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=rate, cmap='coolwarm')

        ax.set_title(f'Time Step: {step}s', fontsize=20)
        ax.view_init(elev=30, azim=1*step)

        return fig,

    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.04, aspect=10)
    cbar.set_label(r'$\varphi_e$', fontsize=15, rotation=0, labelpad=15)

    if not os.path.exists(path):
        os.makedirs(path)

    if save_anim:
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=100)
        writergif = animation.PillowWriter(fps=5)
        anim_path = os.path.join(path, f'{name}_anim.gif')
        anim.save(anim_path, writer=writergif)
        plt.show(block=True)
    else:
        plt.show()

if __name__ == "__main__":
    make_3d_animation(
        mesh_file=c.path_msh, 
        firing_rate_file=c.pinns_input_file, 
        path='simulations', 
        name=f'firing_rate_visualisation_{c.NAME}'
    )




