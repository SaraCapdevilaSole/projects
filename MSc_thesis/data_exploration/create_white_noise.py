from src.config.config_balloon import DATA_PATH, WHITE_PATH, wave_model, T, dt_files
from src.utils.utils_balloon import Balloon
from src.utils.utils import write_to_file, plot_solution, generate_random_signal

import numpy as np
import os

def main():
    """
    Script to create white noise signal
    """
    t_array = np.arange(0, T, dt_files)

    t_length = int(T//dt_files + 1)

    BalloonODE = Balloon(t_array)
    n_parcellations: int = BalloonODE.get_n_parcellations()

    signal = generate_random_signal(shape=(t_length, n_parcellations), std=0.5, mean=0)
    # noíse = generate_random_signal(shape=t_length, std=1e-12)
    # signal = np.zeros((t_length, n_parcellations))
    # signal[:,:] = noíse.reshape(1, t_length, 1) # same noise for all parcellations in each time frame

    write_to_file(
        data=signal,
        data_path_in=os.path.join(DATA_PATH, wave_model), 
        data_path_out=WHITE_PATH, 
        file_numbers=len(t_array)
    )

    plot_solution(np.linspace(0, T, len(signal[0])),signal[0])

if __name__ == "__main__":
    main()