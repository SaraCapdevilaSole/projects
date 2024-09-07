# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# import firedrake as f
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.config.config import SphereConfig, SquareConfig
from src.config.params import T, sn, dt, size, noise, TN, XN, config_kwargs
from src.utils.utils_waves import check_spatial_correlation, check_variance_and_mean, compare_res, spatial_source, analyse_and_plot, spherical_spatial_source, interpolate_Z
from src.config.waves_config import bands_mean

"""
span a broad range of frequencies that by convention is separated into specific 
delta (1–4 Hz), theta (5–8 Hz), alpha (9–14 Hz), beta (15–30 Hz), gamma low (30–50 Hz), 
gamma high (50–100 Hz), and gamma very high (100–150 Hz) bands
"""

all_bands = ["delta", "theta", "alpha", "beta", "gamma", "gamma high", "gamma very high"]

def main():
    # bands = [all_bands[:i] for i in range(1,len(all_bands) + 1)]
    # bands = [[all_bands[i]] for i in range(len(all_bands))]
    # plot_multiple_band_signals(t, bands, noise=noise)
    # compare_res(noise=0.3)
    # signal, t = spatial_source(noise=0.3, animate=False, voxels=[[2,3,-1,19], [2,3,-1,21]]) # bands_names=all_bands
    # analyse_and_plot(signal, t)
    # signal_flat = signal.flatten()
    # print(signal_flat)
    # check_time_correlation(signal_flat)
    # check_spatial_correlation(signal_flat)
    # check_variance_and_mean(signal_flat)
    radius = 100 
    Z_higher = spherical_spatial_source(noise, num_points=500, animate=True, radius=radius)
    # Z_lower = spherical_spatial_source(noise, num_points=100, animate=True, radius=radius)
    # interpolate_Z(*Z_higher, *Z_lower)




if __name__ == "__main__":
    main()



