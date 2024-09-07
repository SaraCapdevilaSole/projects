from src.config.config_balloon import dt_files, T

import os

# Data source
HEMI = 'both'
wave_model = f'robinson_{HEMI}'
DATA_PATH = f'../Firedrake_simulation/{wave_model}'
OUT_DIR = 'BOLD_data'
BOLD_PATH = os.path.join(OUT_DIR, f'balloon_data_{HEMI}/balloon_robinson_{HEMI}')
OUTPUT_PATH = 'PINNs_data'
data_path_fd = os.path.join(DATA_PATH, wave_model)

# define what to convert
# T = 10*60
end_files = int(T/dt_files)
T_avoid = 0#20 # avoid first 20s - transient state
avoid_files = int(T_avoid/dt_files)
