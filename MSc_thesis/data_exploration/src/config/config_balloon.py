import os
import numpy as np

np.random.seed(22)

# config run:
white_noise_run = False
no_ODE_y = True
VERBOSE = False

# Extra parameters:
write = False
plot = True
run = True#write
load = False

# ODE solvers
ODEs = ['EulerSolver', 'RK4Solver', 'HeunsSolver']
ODE_type = ODEs[0]

# Data source
HEMI = 'both'
wave_model = f'robinson_{HEMI}'
DATA_PATH = f'../ML-NFT/{wave_model}'
OUT_DIR = 'BOLD_data'
OUTPUT_PATH = os.path.join(OUT_DIR, f'balloon_data_{HEMI}/balloon_robinson_{HEMI}')

WHITE_PATH = os.path.join(OUT_DIR, 'white_noise_data/white_noise')

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Parameters (from Firedrake):
if HEMI == 'both':
    T = 10*60
    dt = 1e-4 
else:
    T = 0.1
    dt = 3e-5 

if white_noise_run:
    T = 1 
    dt = 1e-4 
    write, plot = False, True

freq = int(0.01/dt)
dt_files = freq * dt #soln recorded every freq * dt