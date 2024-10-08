import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from random import randrange
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit

"""
A script that loads and analyses real-world ECoG Data
"""

"""
NAME: fixation_PAC
SOURCE: https://exhibits.stanford.edu/data/catalog/zk881ps0522
INFORMATION:
Patient	Age	Sex	Handedness	Array location
bp	1	18	F	R	        L Frontoparietal
cc	2	21	M	R	        R Frontotemporal 
zt	3	27	F	R	        L Fronto-temporal-parietal
jp	4	35	F	R	        L Fronto-Temporal
wc	7	32	M	R	        L Fronto-temporal-parietal
wm	8	19	F	R	        R Fronto-parietal
jc	9	18	F	R	        L Frontal 
hl	10	42	M	R	        L Frontotemporal 
jm	11	41	M	L	        L Frontotemporal 
ug	12	39	F	R	        R Frontal
"""

# Read Data
def read_ecog_data(name='jm', path='/Users/saracapdevilasole/Downloads/fixation_PAC/data'):
    file_path = os.path.join(path, name, f'{name}_base.mat')
    mat_data = loadmat(file_path)
    print(list(mat_data.keys()))
    return mat_data

data = read_ecog_data('zt')

# Time dependence:
def plot_time_dependence(data, sf=1000, t_cut=10):
    t_cut_files = t_cut * sf
    fig = plt.subplots(figsize=(10,6), dpi=100)
    time_array = np.linspace(0, data['data'].shape[0]/sf, data['data'].shape[0])
    for vertex in range(data['data'].shape[1]):
        plt.plot(time_array[:t_cut_files], data['data'][:t_cut_files,vertex])
    plt.xlabel('Time (s)')
    plt.ylabel('ECoG [a.u.]')
    plt.show()

def smooth_moving_average(data, window_size=5):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')

def smooth_gaussian(data, sigma=50):
    return gaussian_filter1d(data, sigma=sigma)

def smooth_data(data, method='moving_average', window_size=5, sigma=2):
    T, nodes = data.shape
    smoothed_data = np.zeros_like(data)
    
    for node in range(nodes):
        if method == 'moving_average':
            smoothed_data[:, node] = smooth_moving_average(data[:, node], window_size)
        if method == 'gaussian':
            smoothed_data[:, node] = smooth_gaussian(data[:, node], sigma)
    
    return smoothed_data

def plot_comparison(smoothed_data, raw_data, sf=1000, t_cut=10, t_avoid=1, sigma=50, plot_it=True):
    t_cut_files = int(t_cut * sf)
    t_avoid_files = int(t_avoid * sf)
    time_array = np.linspace(0, raw_data['data'].shape[0]/sf, raw_data['data'].shape[0])
    if plot_it:
        fig = plt.subplots(figsize=(10,6), dpi=100)
        for _ in range(60):
            vertex = randrange(raw_data['data'].shape[1])
            plt.plot(time_array[t_avoid_files:t_cut_files], raw_data['data'][t_avoid_files:t_cut_files,vertex], label='Raw data')
            plt.plot(time_array[t_avoid_files:t_cut_files], smoothed_data['data'][t_avoid_files:t_cut_files,vertex], label=rf'Smoothed data ($\sigma={sigma}$)')
        plt.xlabel('Time (s)')
        plt.ylabel('ECoG [a.u.]')
        # plt.legend()
        plt.show()
    return time_array[t_avoid_files:t_cut_files], raw_data['data'][t_avoid_files:t_cut_files,:]

plot_time_dependence(data)

sigma = 5
smoothed_data = data.copy()
smoothed_data['data'] = smooth_data(data['data'], method='gaussian', window_size=5, sigma=sigma)

plot_time_dependence(smoothed_data)
time, dat = plot_comparison(smoothed_data, data, sigma=sigma, t_cut=1, t_avoid=0.05, plot_it=False)


"""Extract Finite Difference Noise Function:"""

# Define parameters
Q0 = -0.35371 
gamma_s = 116
r_s = 30
nt, nx = dat.shape
dx = 1e-3
dt = 1e-3

phi = dat.copy() / np.max(dat) 
Q = np.zeros((nt, nx))

x = np.linspace(0, (nx-1)*dx, nx)

for t in range(0, nt):
    if t > 1:
        d2phi_dt2 = (phi[t] - 2 * phi[t-1] + phi[t-2]) / dt**2
    else:
        d2phi_dt2 = np.zeros_like(phi[0])
    
    dphi_dt = (phi[t] - phi[t-1]) / dt
    
    d2phi_dx2 = (np.roll(phi[t], -1, axis=0) - 2 * phi[t] + np.roll(phi[t], 1, axis=0)) / dx**2
    Q[t]  = (1/gamma_s**2) * d2phi_dt2 + (2/gamma_s) * dphi_dt + phi[t] - r_s**2 * d2phi_dx2

Q /= np.max(Q)

"""Analyse Extracted Noise"""

SAMPLE_RATE = 1000
N = len(Q)

alphas = []
alphas_errors = []

print(Q.shape[1])
for i in range(Q.shape[1]):
    q = Q[:,i]

    T = np.linspace(0, N/SAMPLE_RATE, N, endpoint=False)  # Time vector

    signal = q / q.max()

    # Compute the FFT
    yf = fft(signal)
    xf = fftfreq(N, 1 / SAMPLE_RATE)

    # Positive frequencies
    xf = xf[:N // 2]
    yf = yf[:N // 2]

    thresholds = [1]
    num_frequencies = []
    errors = []
    reconstructed_signals = []
    phases = []
    frequencies = []
    amplitudes = []

    for threshold in thresholds:
        mask = np.abs(yf) > threshold
        xf_filtered = xf[mask]
        yf_filtered = yf[mask]

        # Reconstruct the signal using filtered frequencies
        reconstructed_signal = np.zeros(N)
        for freq, coeff in zip(xf_filtered, yf_filtered):
            phase = np.angle(coeff)
            amplitude = np.abs(coeff)
            reconstructed_signal += amplitude * np.cos(2 * np.pi * freq * T + phase)
            phases.append(phase)
            frequencies.append(freq)
            amplitudes.append(amplitude)

        normalised_recon_signal = reconstructed_signal / reconstructed_signal.max()
        
        error = np.sqrt(np.sum((normalised_recon_signal - signal)**2)) / np.sqrt(np.sum(signal**2))
        errors.append(error)
        reconstructed_signals.append(normalised_recon_signal)
        num_frequencies.append(len(yf_filtered))

    model = lambda x, c1, c2: c1 * (x ** c2)

    amplitudes /= np.max(amplitudes) 
    popt, pcov = curve_fit(model, frequencies[2:], amplitudes[2:], p0=[1, 1]) 

    c1_opt, c2_opt = popt

    p_sigma = np.sqrt(np.diag(pcov))
    error_c2 = p_sigma[1]
    
    alpha = - c2_opt
    alphas.append(alpha)
    alphas_errors.append(p_sigma[1])
    print(f"{c2_opt:.2f} \pm {p_sigma[1]:.2f}")

    x_fit = np.linspace(min(frequencies[2:]), max(frequencies[2:]), 100)
    y_fit_p = model(x_fit, popt[0] + p_sigma[0], popt[1] + p_sigma[1])
    y_fit_m = model(x_fit, popt[0] - p_sigma[0], popt[1] - p_sigma[1])
    y_fit_mean = model(x_fit, *popt)


fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].scatter(frequencies[2:], phases[2:], color='black', alpha=0.7, s=5)
axes[0].set_xlabel("Frequency [Hz]", fontsize=12)
axes[0].set_ylabel("Phase [rad]", fontsize=12)
axes[0].set_yticks([-2,0,2])

axes[1].scatter(frequencies[2:], amplitudes[2:], color='black', alpha=0.7, s=5)
axes[1].plot(x_fit, y_fit_mean, label=f'$A(f)\propto f^{{{c2_opt:.2f} \pm {error_c2:.2f}}}$', color='red', alpha=0.7)
axes[1].fill_between(x_fit, y_fit_m, y_fit_p, color='red', alpha=0.7)
axes[1].legend(fontsize=12)
axes[1].set_xlabel("Frequency [Hz]", fontsize=12)
axes[1].set_ylabel("Amplitude [a.u.]", fontsize=12)
axes[1].set_yticks([0,0.5,1])
axes[2].scatter(phases[2:], amplitudes[2:], color='black', alpha=0.7, s=5)
axes[2].set_xlabel("Phase [rad]", fontsize=12)
axes[2].set_ylabel("Amplitude [a.u.]", fontsize=12)
axes[2].set_xticks([-2,0,2])
axes[2].set_yticks([0,0.5,1])
plt.tight_layout()
plt.show()

print(len(amplitudes))
print(np.mean(alphas))
print(np.std(alphas))

plt.figure(figsize=(8, 6))
plt.hist(alphas, bins=30, color='blue', edgecolor='black')

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Values')

plt.show()
