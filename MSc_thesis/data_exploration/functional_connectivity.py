from src.utils.utils import load_from_file, plot_sample_curves
from src.utils.utils_fc import compute_FC_matrix, compute_cdist_matrix, compute_FC_average, compute_static_FC, normalise_bold_data
from src.config.config_fc import (
    TR, 
    KIND, 
    OUTPUT_PATH,
    T,
    FC_MAP_PATH,
    dt_files,
    TR_inc,
    start_files, 
    end_files,
    compute_fc_mean, 
    plot_mean, 
    plot_fcs, 
    plot_bold_curves, 
    compute_cdist, 
    plot_cdist, 
    compute_static, 
    sliding_window_fc,
    plot_static,
    plot_samples,
    total_samples,
    normalise_solution
)

import numpy as np

def main():
    """
    Script to find the Functional Connectivity matrix of the BOLD signal.
    """
    t_array = np.arange(0, T, dt_files)
    assert end_files <= len(t_array) 

    solution = load_from_file(
        data_path_in=OUTPUT_PATH,
        start_file=start_files,
        file_numbers=end_files
    )

    # Apply parcellation map: 5124 voxels -> 100 parcels
    if normalise_solution:
        solution = normalise_bold_data(data=solution, filename=FC_MAP_PATH)

    if plot_bold_curves:
        plot_sample_curves(data_array=solution, 
                           time_to_plot=[start_files,end_files], 
                           time_array=t_array, 
                           n_samples_to_plot=plot_samples, 
                           total_samples=total_samples)
        
    if compute_static:
        compute_static_FC(solution, kind=KIND, plot_it=plot_static)
    
    if sliding_window_fc:
        FC_matrices = compute_FC_matrix(solution, window_size=TR, increment=TR_inc, kind=KIND, plot_it=plot_fcs)

        if compute_fc_mean:
            average_FC_matrix = compute_FC_average(FC_matrices, plot_it=plot_mean)

        if compute_cdist:
            cdist_matrix = compute_cdist_matrix(solution, plot_it=plot_cdist)

if __name__ == "__main__":
    main()
