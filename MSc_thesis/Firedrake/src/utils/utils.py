import numpy as np
import meshio
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from src.config.config import SquareConfig, SphereConfig, BrainConfig
from scipy.signal import detrend, butter, lfilter

def plot_solution(x,y,subplots=False):
    plt.plot(x, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal [a.u.]')
    if not subplots:
        plt.show()

def plot_sample_curves(data_array, time_array, sn, n_samples_to_plot=1000, total_samples=5124, savefig='test_bold.png'):
    logging.info(f'Plotting {n_samples_to_plot} curves')
    samples = set(np.arange(0,total_samples))
    plt.figure(figsize=(15,4), dpi=100)
    for _ in range(n_samples_to_plot):
        sample = np.random.choice(list(samples))
        plot_solution(x=time_array, y=data_array[:, sample], subplots=True)
        samples.remove(sample)
    # plot_one_more(time_array[time_to_plot[0]:time_to_plot[1]], np.max(data_array))
    # plt.xlim([int(t*sn) for t in time_to_plot])
    # plt.ylim([1.02e-5, 1.05e-5]) # firedrake
    # plt.ylim([-0.75e-7, 0.75e-7]) # bold signals

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(savefig, bbox_inches="tight", dpi=300)

def find_XN_TN(XN, TN, T, dt, size, r=100):
    if XN == -1:
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        import firedrake as f
        mesh = f.IcosahedralSphereMesh(radius=r, refinement_level=size)
        V = f.FunctionSpace(mesh, "CG", 1)
        XN = V.dim()

    if TN == -1:
        TN = int(T/dt)
    return XN, TN

def get_config(T, dt, sn, noise, size, XN, TN, _extra_str, shape, **kwargs):
    if shape == 'square':
        Config = SquareConfig  
    elif shape == 'sphere':
        Config = SphereConfig 
    elif shape == 'real_mesh':
        Config = BrainConfig
    else:
        raise NotImplementedError(f"Shape '{shape}' is not implemented.")

    config_params = {
        'T': T,
        'dt': dt,
        'sampling_nom': sn,
        'noise': noise,
        'size': size,
        'XN': XN,
        'TN': TN,
        'extra': _extra_str
    }
    config_params.update(kwargs)
        
    return Config(**config_params)

def bandpass_data(signal, lowcut, highcut, tr, k):
    fnq = 1 / (2 * tr)
    Wn = [lowcut / fnq, highcut / fnq] # 0 < Wn < 1
    bfilt, afilt = butter(k, Wn, btype='bandpass')
    filtered_data = lfilter(bfilt, afilt, signal)
    return filtered_data

def detrend_data(signal):
    """Remove linear trend along axis from data."""
    return detrend(signal, axis=0)
