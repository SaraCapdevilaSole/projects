from src.config.config_PINNs import BOLD_PATH, OUTPUT_PATH, end_files, avoid_files, T, T_avoid, dt_files, data_path_fd
from src.utils.utils import load_from_file, plot_sample_curves

import meshio
import numpy as np

def main(files=end_files):
    # load mesh
    mesh = meshio.read(data_path_fd + f'_{0}.vtu')
    mesh_coordinates = mesh.points 

    # load data from all files
    phi_e_T_data = load_from_file(
        data_path_in=data_path_fd,
        file_numbers=files, 
        start_file=avoid_files 
    ) 

    t_array = np.arange(0, T - T_avoid, dt_files)

    # save to file
    data_dict = {
        'phi_e': phi_e_T_data, # shape: (T/dt_files - avoid_files, X=5124)
        'mesh_coordinates': mesh_coordinates, # shape: (X=5124, D=3) # x,y,z
        't_star': t_array # shape: (T/dt_files - avoid_files,)
    }

    np.save(f'{OUTPUT_PATH}/nft_firedrake_{files - avoid_files}.npy', data_dict)

    plot_sample_curves(data_array=phi_e_T_data, time_to_plot=[0, len(t_array)], time_array=t_array)

if __name__ == "__main__":
    main()
