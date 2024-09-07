import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
import meshio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.utils.utils_helmholtz_linear import real_mesh_patch, find_mesh_patch
from src.utils.utils import get_config
from src.config.params import T, sn, dt, size, XN, TN, noise, max_size, noise_type, _extra_str, use_noise, shape, save_derivatives, config_kwargs
from src.config.config import SphereConfig, SquareConfig

c = get_config(T, dt, sn, noise, size, XN, TN, _extra_str, shape, **config_kwargs)

if shape == 'square':
    mesh = UnitSquareMesh(max_size, max_size)
elif shape == 'sphere':
    mesh = IcosahedralSphereMesh(radius=c.radius, refinement_level=max_size)
elif shape == 'real_mesh':
    try:
        mesh = Mesh(c.path_func(max_size), dim=3)
    except FileNotFoundError as e:
        print("File not found: ", e)

output_path = c.output_phi_e
pinns_input_file = c.pinns_input_file

num_files = int((T / dt) / c.freq)

phi_e_combined_array = [None] * num_files  
Qs_combined_array = np.load(f'{c.output_phi_e}/noise.npy') if noise_type == "load_from_file" else [None] * num_files
p_t_combined_array = [None] * num_files
p_tt_combined_array = [None] * num_files

def load_file_with_index(index, file_path):
    try:
        data = np.load(file_path)
        return index, data
    except FileNotFoundError:
        return index, None

last_index = num_files
e_count = 0

# Parallelise to speed up
with ThreadPoolExecutor() as executor:
    futures = []
    for i, n in enumerate(range(0, int(T/dt), c.freq)):
        file_path = f"{output_path}/phi_e_{n}.npy"
        futures.append(executor.submit(load_file_with_index, i, file_path))
    
    for future in tqdm(futures):
        index, data = future.result()
        if data is not None:
            phi_e_combined_array[index] = data
        else:
            last_index = min(last_index, index)
            e_count += 1
            print(f"File not found for index: {len(phi_e_combined_array)}")
    
    if use_noise and noise_type != "load_from_file":
        futures = []
        for i, n in enumerate(range(0, int(T/dt), c.freq)):
            file_path = f"{output_path}/Qs_{n}.npy"
            futures.append(executor.submit(load_file_with_index, i, file_path))

        for future in tqdm(futures):
            index, data = future.result()
            if data is not None:
                Qs_combined_array[index] = data

    if save_derivatives:
        futures_t = []
        futures_tt = []
        for i, n in enumerate(range(0, int(T/dt), c.freq)):
            p_t_path = f"{output_path}/phi_t_e_{n}.npy"
            p_tt_path = f"{output_path}/phi_tt_e_{n}.npy"
            futures_t.append(executor.submit(load_file_with_index, i, p_t_path))
            futures_tt.append(executor.submit(load_file_with_index, i, p_tt_path))

        for future in tqdm(futures_t):
            index, data = future.result()
            if data is not None:
                p_t_combined_array[index] = data

        for future in tqdm(futures_tt):
            index, data = future.result()
            if data is not None:
                p_tt_combined_array[index] = data

if e_count != 0:
    print(f"Total missing files: {e_count} / {int((T/dt)/c.freq)}")

array_cleanup = lambda array_data: np.array(array_data[:last_index])

phi_e_T_data = array_cleanup(phi_e_combined_array)
Qs_T_data = array_cleanup(Qs_combined_array)
if save_derivatives:
    p_t_T_data = array_cleanup(p_t_combined_array)
    p_tt_T_data = array_cleanup(p_tt_combined_array)

# mesh = meshio.read(c.path_msh)
# mesh_coordinates = mesh.points

mesh_coordinates = np.array(mesh.coordinates.dat.data)
if isinstance(c.prop_to_keep, str):
    del mesh_coordinates
    mesh_coordinates = real_mesh_patch(mesh, subject=c.prop_to_keep)
    print(f'\nmesh coordinates after: \n{mesh_coordinates}')
elif isinstance(c.prop_to_keep, list):
    if c.prop_to_keep[0] < 1:
        del mesh_coordinates
        mesh_coordinates = find_mesh_patch(mesh, c.prop_to_keep, None)

t_array = np.linspace(0, T, int(T/sn)) 
t_array = t_array[:phi_e_T_data.shape[0]]

data_dict = {
    'phi_e': phi_e_T_data, # shape: (T/dt_files, X)
    'Qs': Qs_T_data,
    'mesh_coordinates': mesh_coordinates, # shape: (X, D=3) # x,y,z
    't_star': t_array # shape: (T/dt_files - avoid_files,)
}

if save_derivatives:
    data_dict['p_t'] = p_t_T_data
    data_dict['p_tt'] = p_tt_T_data
    print(np.shape(p_tt_T_data))

# Save the data 
np.save(pinns_input_file, data_dict)

# print(phi_e_T_data[1,:])
print(np.shape(mesh_coordinates))
print(np.shape(phi_e_T_data))
print(np.shape(Qs_T_data))
print(np.shape(t_array))
