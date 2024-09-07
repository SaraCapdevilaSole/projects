import matplotlib.pyplot as plt
import numpy as np
import os

from src.config.params import size, noise
from src.config.config import SphereConfig
from src.utils.utils import find_XN_TN
from src.utils.one_over_f_functions import compute_spectrum, estimate_aperiodic_exponent, make_plots

n = 1
fig, ((ax1, ax2,), (ax3, ax4)) = plt.subplots(2, 2,figsize=(12, 6), dpi=80)

data_dict = [
    {'XN_TN': [50,100], 'sn': 1e-3, 'dt': 1e-3, 'T': 2},
    {'XN_TN': [50,50], 'sn': 1e-3, 'dt': 1e-3, 'T': 1},
    # {'XN_TN': [None,None], 'sn': 1e-4, 'dt': 1e-4, 'T': 1},
]

for i, d in enumerate(data_dict):
    T, dt, sn = d['T'], d['dt'], d['sn']
    if i==0:
        ax, ax_ = ax1, ax3
    elif i==1:
        ax, ax_ = ax2, ax4
    else:
        break
    XN_i, TN_i = find_XN_TN(d['XN_TN'][0], d['XN_TN'][1], T, dt, size, r=100)
    ax.set_title(f"XN={XN_i}, TN={TN_i}")
    c = SphereConfig(T=T, dt=dt, sampling_nom=sn, noise=noise, size=size, XN=XN_i, TN=TN_i, extra="interpolated_TEST")
    data = np.load(c.pinns_input_file, allow_pickle=True).item()
    [S,f] = compute_spectrum(data['phi_e'][:,n], data['t_star'])
    [aperiodic_exponent, x_linear_fit, y_linear_fit] = estimate_aperiodic_exponent(S,f,[50,1000])
    make_plots(data['t_star'], data['phi_e'][:,n], f,S, x_linear_fit,y_linear_fit,aperiodic_exponent, ax,ax_)

XNs, TNs = zip(*[find_XN_TN(d['XN_TN'][0], d['XN_TN'][1], T, dt, size, r=100) for d in data_dict])
savedir = 'simulations/spectral_density'
os.makedirs(savedir, exist_ok=True)
fig.savefig(f'{savedir}/spectral_XN={XNs}_TN={TNs}.png')
