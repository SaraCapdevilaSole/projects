from src.config.config_balloon import wave_model, dt_files

import os
from collections import OrderedDict
import numpy as np
import meshio
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from scipy.signal import detrend, butter, lfilter

logging.basicConfig(level=logging.INFO)

def bandpass_data(signal, lowcut, highcut, tr, k):
    fnq = 1 / (2 * tr)
    Wn = [lowcut / fnq, highcut / fnq] # 0 < Wn < 1
    bfilt, afilt = butter(k, Wn, btype='bandpass')
    filtered_data = lfilter(bfilt, afilt, signal)
    return filtered_data

def detrend_data(signal):
    """Remove linear trend along axis from data."""
    return detrend(signal, axis=0)

def save_list_to_txt(lst: list, filename: str) -> None:
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')

def open_list_from_txt(filename: str) -> list:
    try:
        with open(filename, 'r') as file:
            data_list = [line.strip() for line in file.readlines()]
        return data_list
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []

def merge_dictionaries(data_lh, data_rh):
    merged_dict = OrderedDict()
    for key in data_rh:
        try:
            merged_dict[key] = np.concatenate((data_rh[key], data_lh[key]))
        except ValueError:
            merged_dict[key] = data_rh[key]
    return merged_dict

def plot_solution(x,y,subplots=False):
    plt.plot(x, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal [a.u.]')
    if not subplots:
        plt.show()

def generate_random_signal(shape, mean=0, std=10):
    return np.random.normal(mean, std, size=shape)

def write_to_file(data_path_in: str, 
                  data_path_out: str, 
                  file_numbers: int, 
                  data: np.array, 
                  extension: callable=lambda f: f'_{f}.vtu'):
    logging.info(f'Writing files to path:\n{data_path_out}')
    
    for f in tqdm(range(0, file_numbers)):
        mesh = meshio.read(data_path_in + extension(f))
        mesh.point_data['phi_e'] = data[f, :] 
        mesh.write(data_path_out + extension(f), file_format = 'vtu')

def load_from_file(data_path_in: str, 
                   file_numbers: int, 
                   start_file: int = 0, 
                   extension: callable=lambda f: f'_{f}.vtu'):
    logging.info(f'Loading files from path: {data_path_in}')
    data = []
    for f in tqdm(range(start_file, file_numbers)):
        mesh = meshio.read(data_path_in + extension(f))
        data.append(mesh.point_data['phi_e'])
    return np.array(data)

def plot_sample_curves(data_array, time_to_plot, time_array, n_samples_to_plot=1000, total_samples=5124):
    logging.info(f'Plotting {n_samples_to_plot} curves')
    samples = set(np.arange(0,total_samples))
    plt.figure(figsize=(15,4), dpi=100)
    for _ in range(n_samples_to_plot):
        sample = np.random.choice(list(samples))
        plot_solution(x=time_array[time_to_plot[0]:time_to_plot[1]], y=data_array[:, sample], subplots=True)
        samples.remove(sample)
    # plot_one_more(time_array[time_to_plot[0]:time_to_plot[1]], np.max(data_array))
    plt.xlim([int(t*dt_files) for t in time_to_plot])
    # plt.ylim([1.02e-5, 1.05e-5]) # firedrake
    # plt.ylim([-0.75e-7, 0.75e-7]) # bold signals

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.show()

def plot_one_more(time_array, a):
    """for debugging Band Pass filters"""
    frequency_low = 0.01 # 1 wave every 100s
    frequency_high = 0.1 # 1 wave every 10s

    # Generate time array
    t = time_array

    # Generate sinusoidal signal
    sinusoidal_signal1 = a * np.sin(2 * np.pi * frequency_low * t)
    sinusoidal_signal2 = a * np.sin(2 * np.pi * frequency_high * t)

    # Plot the signal
    plt.plot(t, sinusoidal_signal1, linewidth=5)
    plt.plot(t, sinusoidal_signal2, linewidth=5)
