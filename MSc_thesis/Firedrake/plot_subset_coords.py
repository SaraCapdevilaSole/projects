import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import meshio
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from concurrent.futures import ThreadPoolExecutor
import matplotlib.tri as mtri
from scipy.spatial import Delaunay
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from src.utils.utils_helmholtz_linear import real_mesh_patch, find_mesh_patch
from src.utils.utils import get_config
from src.config.params import T, sn, dt, size, XN, TN, noise, max_size, noise_type, _extra_str, use_noise, shape, save_derivatives, config_kwargs
from src.config.config import SphereConfig, SquareConfig

WRITE_COORDS_TO_FILE = False
WRITE_N_CURV_TO_FILE = True
PLOT = False
PLOT_KNN = False
max_size =  164

c = get_config(T, dt, sn, noise, size, XN, TN, _extra_str, shape, **config_kwargs)
save_path = f'/vol/bitbucket/sc3719/JAXPI/jaxpi/examples/helmholtz_3d_inverse/real_data/mesh_{max_size}k_patches'
os.makedirs(save_path, exist_ok=True)

if shape == 'square':
    mesh = UnitSquareMesh(max_size, max_size)
elif shape == 'sphere':
    mesh = IcosahedralSphereMesh(radius=c.radius, refinement_level=max_size)
elif shape == 'real_mesh':
    try:
        mesh = Mesh(c.path_func(max_size), dim=3)
    except FileNotFoundError as e:
        print("File not found: ", e)

def create_faces(points):
    delaunay = Delaunay(points)
    tri = delaunay.simplices # triangulation
    return tri

def triang_func(pos):
    faces = create_faces(pos[:, :2])
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    return triang

mesh_coordinates = np.array(mesh.coordinates.dat.data)

ft = 0
output_path = c.output_phi_e
pinns_input_file = c.pinns_input_file
data = np.load(pinns_input_file, allow_pickle=True).item()
u_data = data['phi_e']
mesh_subset = data['mesh_coordinates']

n_subjects = 0
mesh_subsets = []
subset_datas = []
# subjects = ['bp', 'zt', 'jp', 'wc', 'jc', 'hl', 'jm']
subjects = ['jm']

n_curv_general_path = f'data/input_data/inflated_mesh_curvature_normals/norm_curv_{max_size}k'
curv_path = n_curv_general_path + f'/curv_{max_size}.txt'
n_path = n_curv_general_path + f'/normals_{max_size}.txt'
n_curv_path = [n_path, curv_path]
for s, subject in enumerate(subjects):
    # rg_interest, mesh_subset, subset_data, raw_coords = real_mesh_patch(mesh, subject=subject, return_data=True)
    rg_interest, n_patch, curv_patch = real_mesh_patch(mesh, subject=subject, n_curv_path=n_curv_path)
    if rg_interest.shape[0] == 0:
        subjects.pop(s)
        print(s)
        continue
    else:
        if WRITE_COORDS_TO_FILE:
            s_path = os.path.join(save_path, f'{subject}_{max_size}.npy')
            mesh_dict = {'nn_coords': mesh_subset, 'region': rg_interest}
            np.save(s_path, mesh_dict)
        if WRITE_N_CURV_TO_FILE:
            s_path = n_curv_general_path + f'/norm_curv_{max_size}_{subject}.npy'
            ncurv_dict = {'norm': n_patch, 'curv': curv_patch}
            np.save(s_path, ncurv_dict)
        n_subjects += 1

if PLOT_KNN:
    mesh_subset, subset_data, raw_coords, rg_interest = real_mesh_patch(mesh, subject='bp', return_data=True)
    fig = plt.figure(figsize=(10, 8), dpi=150)  # Adjusted figure size

    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(mesh_coordinates[:, 0], mesh_coordinates[:, 1], mesh_coordinates[:, 2], color='gray', s=10, zorder=1, alpha=0.01)
    sc2 = ax1.scatter(raw_coords[:, 0], raw_coords[:, 1], raw_coords[:, 2], color='red', s=20, label='data', zorder=2)
    ax1.grid(False)
    ax1.set_axis_off()
    ax1.view_init(elev=20, azim=0)

    ax2 = fig.add_subplot(122, projection='3d')
    sc3 = ax2.scatter(mesh_coordinates[:, 0], mesh_coordinates[:, 1], mesh_coordinates[:, 2], color='gray', s=10, zorder=1, alpha=0.01)
    sc5 = ax2.scatter(rg_interest[:, 0], rg_interest[:, 1], rg_interest[:, 2], color='yellow', s=10, zorder=1, alpha=0.1)
    sc4 = ax2.scatter(mesh_subset[:, 0], mesh_subset[:, 1], mesh_subset[:, 2], color='black', s=20, label='NN data', zorder=2)
    ax2.grid(False)
    ax2.set_axis_off()
    ax2.view_init(elev=20, azim=0)

    sc5 = ax2.scatter(rg_interest[:1, 0], rg_interest[:1, 1], rg_interest[:1, 2], color='#EDE200', s=20, zorder=1, label='mesh region')

    handles, labels = [], []
    for ax in [ax1, ax2]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    ax2.legend(handles, labels, loc='upper right', fontsize=12)

    plt.subplots_adjust(wspace=-.1) 

    plt.savefig('test.pdf')



if PLOT:
    x = mesh_coordinates[:, 0]
    y = mesh_coordinates[:, 1]    
    z = mesh_coordinates[:, 2]

    from random import randrange

    def plot_comparison(ax, raw_data, sf=1000, t_cut=2.01, t_avoid=1, sigma=50):
        t_cut_files = int(t_cut * sf)
        t_avoid_files = int(t_avoid * sf)
        time_array = np.linspace(0, raw_data.shape[0]/sf, raw_data.shape[0])
        for _ in range(10):
            vertex = randrange(raw_data.shape[1])
            ax.plot(time_array[t_avoid_files:t_cut_files], raw_data[t_avoid_files:t_cut_files,vertex], linewidth=0.1)
        ax.grid(False)
        min_y = np.min(raw_data[t_avoid_files:t_cut_files, :])
        max_y = np.max(raw_data[t_avoid_files:t_cut_files, :])
        ax.set_yticks([]) 
        ax.set_xticks([int(np.min(time_array[t_avoid_files:t_cut_files])), int(np.max(time_array[t_avoid_files:t_cut_files]))])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([min_y * 1.1, max_y * 1.1])

    fig, axes = plt.subplots(1, n_subjects, figsize=(12, 4), subplot_kw={'projection': '3d'}, dpi=150)

    colors = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'black', 'gray', 'pink']

    for i, ax in enumerate(axes):
        ax.scatter(mesh_coordinates[:, 0], mesh_coordinates[:, 1], mesh_coordinates[:, 2], color='gray', s=15, zorder=1, alpha=0.005)
        mesh_subset = mesh_subsets[i]
        ax.scatter(mesh_subset[:, 0], mesh_subset[:, 1], mesh_subset[:, 2], color=colors[i], s=10, zorder=2, alpha=1.0, label='{subjects[i]}')
        
        pos = ax.get_position()

        inset_ax_width = 0.15
        inset_ax_height = 0.15
        inset_ax_x = pos.x0 - 0.02# Adjust these values as needed
        inset_ax_y = pos.y0 - inset_ax_height - 0.02  # Adjust the vertical offset as needed

        inset_ax = fig.add_axes([inset_ax_x, inset_ax_y, inset_ax_width, inset_ax_height])

        plot_comparison(inset_ax, subset_datas[i])
        if i == 0:
            inset_ax.set_ylabel('ECoG [a.u.]')
        if i == n_subjects//2:
            inset_ax.set_xlabel('Time (s)')

        ax.view_init(elev=20, azim=0)
        ax.set_title(subjects[i])
        ax.grid(False)
        ax.set_axis_off()

    save_path = 'simulations/real_mesh'
    os.makedirs(save_path, exist_ok=True)
    fig_path = os.path.join(save_path, f'{noise_type}_{size}_{dt}_{_extra_str}.pdf')
    fig.savefig(fig_path, bbox_inches="tight", dpi=100)

    plt.show()
