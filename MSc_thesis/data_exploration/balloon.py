from src.ode_solvers import EulerSolver, RK4Solver, ODEIntSolver, HeunsSolver
from src.utils.utils import write_to_file, plot_sample_curves, load_from_file
from src.utils.utils_balloon import Balloon
from src.config.config_balloon import (
    DATA_PATH, 
    OUTPUT_PATH, 
    WHITE_PATH,
    T, 
    white_noise_run,
    wave_model, 
    ODE_type,
    dt_files, 
    write, 
    plot,
    run,
    load
)

import numpy as np
import os

"""
Script to convert the NFT wave signal to a BOLD signal.
TODO: 
    - To concentrate on the frequency range where resting-state activity appears the most functionally relevant, 
        both empirical and simulated BOLD signals were band pass filtered between 0.1 and 0.01 Hz
"""

def main(write=write, white_noise_run=white_noise_run, plot=plot, dt_files=dt_files, T=T, data=None, run=run, t_array=None, load=None):
    # Initialise arrays
    if t_array is None:
        t_array = np.arange(0, T, dt_files)
    x0 = np.array([0,1,1,1,0])

    if white_noise_run or run:
        # Classes
        if white_noise_run:
            BalloonODE = Balloon(t_array, data_path=WHITE_PATH)
        else:
            BalloonODE = Balloon(t_array, data=data)
        n_parcellations: int = BalloonODE.get_n_parcellations()
        odes = BalloonODE.ode_system
        # Solvers 
        solver_type = globals()[ODE_type]
        ode_solver = solver_type(odes)
        solution = ode_solver.solve(x0, dt_files, T, n_parcellations)
        y = solution[:, -1, :]

    if write:
        write_to_file(
            data=y,
            data_path_in=os.path.join(DATA_PATH, wave_model), 
            data_path_out=OUTPUT_PATH, 
            file_numbers=len(t_array)
        )

    end_t_to_plot = 100 #len(t_array)
    time_to_plot = 40000
    start_t = end_t_to_plot - time_to_plot - 1

    if end_t_to_plot > len(t_array):
        end_t_to_plot = len(t_array)

    load = write is False and data is None if load is None else load
    if load:
        solution = load_from_file(
            data_path_in=OUTPUT_PATH,
            file_numbers=end_t_to_plot,
            start_file=start_t
        )
        y = solution # shape: (T_to_plot, n_parcellations)


    if plot:
        plot_sample_curves(data_array=y, time_to_plot=[start_t,end_t_to_plot], time_array=t_array)

    return y


if __name__ == "__main__":
    main()
