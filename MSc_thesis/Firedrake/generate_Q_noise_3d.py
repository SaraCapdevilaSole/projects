import os
os.environ["OMP_NUM_THREADS"] = "1"
import firedrake as f
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import welch

from src.config.config import SphereConfig, SquareConfig
from src.config.params import T, sn, dt, size, noise, TN, XN

print(f'Using XN={XN} and TN={TN}')
#######################################################
# Create Noise Vector of particular frequency in space and time domain

# on a mesh with SphereConfig size
# size = 60
# c = SquareConfig(T=0.01, dt=1e-5, sampling_nom=1e-3, noise=0.1, size=size)
# mesh = f.UnitSquareMesh(size, size)
mesh = f.IcosahedralSphereMesh(radius=100, refinement_level=size)
V = f.FunctionSpace(mesh, "CG", 1)
rng = np.random.default_rng(seed=22)

save_dir = f'data/input_data/Qs_3d_tx'
os.makedirs(save_dir, exist_ok=True)

mesh_size = V.dim()
print(f"Size of the mesh V: {mesh_size}")

# XN = mesh_size #100 # number of space samples
# TN = #50 # number of time samples

noise_matrix = rng.normal(0, noise, size=(TN, XN))
x_old = np.linspace(0, 1, XN)
t_old = np.linspace(0, 1, TN)

# print(noise_matrix)
np.save(save_dir + f'/Qs_X={XN}_T={TN}_n={noise}.npy', noise_matrix)

#######################################################
# Create Interpolated Noise

XN_ = mesh_size # number of space samples after interpolating
TN_ = int(np.ceil(T/dt)) # number of time samples after interpolating

x_new = np.linspace(0, 1, XN_)
t_new = np.linspace(0, 1, TN_)
interpolator_x = interp1d(x_old, noise_matrix, kind='quadratic', axis=1) # spatial interpolator
noise_matrix_int_x = interpolator_x(x_new)
interpolator_t = interp1d(t_old, noise_matrix_int_x, kind='quadratic', axis=0) # temporal interpolator
noise_matrix_int_tx = interpolator_t(t_new)

print(noise_matrix_int_tx)
print(noise_matrix_int_tx.shape)

# np.save(save_dir + f'/Qs_X={XN}_T={TN}_n={noise}_interpolated_Xn={XN_}_Tn={TN_}.npy', noise_matrix_int_tx)

mean_std = lambda matrix, ax: [np.mean(np.mean(matrix, axis=ax)), np.mean(np.std(matrix, axis=ax))]
mean, std = mean_std(noise_matrix, 0)
error_m = np.abs(mean)
error_std = np.abs(std-noise)
mean_int, std_int = mean_std(noise_matrix_int_tx, 0)
print(f"errors: mean {error_m}, std {error_std}")
print(f"Original noise vector (size {TN}, mean {mean}, std {std})")
print(f"Interpolated noise vector (size {TN_}, mean {mean_int}, std {std_int})")

plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, 1, TN_), noise_matrix_int_tx[:,0], label=f"Interpolated Noise ({TN_} points)", linewidth=2)
plt.scatter(np.linspace(0, 1, TN), noise_matrix[:,0], color='red', label=f"Original Noise ({TN} points)", zorder=5)
plt.legend()
plt.title("Original and Interpolated Noise Vectors")
plt.xlabel("Index")
plt.ylabel("Noise Value")
plt.grid(True)

savepath = 'simulations/noise_figs'
plt.savefig(f"{savepath}/interpolated_noise_T={TN}_TN={TN_}.png")

#########
from matplotlib.ticker import ScalarFormatter


fig = plt.figure(figsize=(10, 23))
ax = fig.add_subplot(projection='3d')

XX, TT = np.meshgrid(x_new, t_new)

plt.grid(False)
plt.axis('off')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surface = ax.plot_surface(XX, TT, noise_matrix_int_tx, cmap='coolwarm')

fig.colorbar(surface, ax=ax, shrink=0.3, aspect=5)

ax.set_xlabel('X')
ax.set_ylabel('T')
ax.set_zlabel('noise')

# mesh_coordinates = np.array(mesh.coordinates.dat.data)
# print(mesh_coordinates)
# x,y,z = [mesh_coordinates[:,i] for i in range(3)]
# scatter = ax.scatter(x, y, z, c=noise_matrix_int_tx[0,:], cmap='coolwarm')
# cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)  # Force scientific notation
# cbar.ax.yaxis.set_major_formatter(formatter)

fig.savefig(f'{savepath}/noise_3d_X={XN}_T={TN}.png', bbox_inches="tight", dpi=300)

#######################################################
# Power spectrum

# fs = 1 / (Y*0.1) # sampling frequency

# frequencies, psd = welch(interpolated_noise, fs, nperseg=1024)

# plt.figure(figsize=(10, 6))
# plt.semilogy(frequencies, psd)
# plt.title('Power Spectral Density (PSD) using Welch\'s method')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power/Frequency (dB/Hz)')
# plt.grid(True)
# plt.savefig(f"power_spectrum_Y={Y}.png")



