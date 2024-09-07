from src.config.config_PINNs import end_files, avoid_files, T, T_avoid, dt_files, data_path_fd
from src.utils.utils import load_from_file, plot_sample_curves

import meshio
import numpy as np
import h5py
import logging 

def main(files=end_files):
    t_array = np.arange(0, T - T_avoid, dt_files)
    # load mesh
    mesh = meshio.read(data_path_fd + f'_{0}.vtu')
    mesh_pos = mesh.points
    assert len(mesh.cells) == 1
    cells = mesh.cells[0].data # triangles: 10240 cells
    node_type = [[0]]*len(mesh_pos)

    # load data from all files
    phi_e_T_data = load_from_file(
        data_path_in=data_path_fd,
        file_numbers=files, 
        start_file=avoid_files
    ) 

    # save to file
    data_dict = {
        'phi_e': phi_e_T_data, # shape: (Time, Vertices)
        'mesh_pos': mesh_pos, # shape: (Time, Vertices, Dimensions)
        'cells': cells,
        'node_type': node_type, # shape: (Time, Vertices)
        't_star': t_array # shape: (Time,)
    }

    try:
        with h5py.File(f'../DeepMind/meshgraphnets_nft/datasets_nft/data_r.h5', 'w') as hf:
            for key, value in data_dict.items():
                hf.create_dataset(key, data=value)
        logging.info("Data saved successfully")
    except Exception as e:
        logging.error(f"An error occurred while saving data to HDF5: {e}")

    # np.save(f'../DeepMind/meshgraphnets_nft/datasets_nft/data.npy', data_dict)
    # plot_sample_curves(data_array=phi_e_T_data, time_to_plot=[0, len_t], time_array=t_array)

if __name__ == "__main__":
    main()
