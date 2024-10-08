import matplotlib.pyplot as plt
import meshio
import numpy as np
import os

from src.utils.utils import get_config
from matplotlib.ticker import FuncFormatter
from src.config.params import noise, T, sn, dt, size, XN, TN, max_size, _extra_str, shape, config_kwargs
from src.utils.utils_helmholtz_linear import calculate_errors

"""Script for visualisation of different voxel signals on the mesh."""

n = 0

dts = [1e-6, 5e-6, 1e-5, 5e-5] 
sizes = [164, 131, 80, 65, 40]

ls = ['-', '--', '-.', ':']
alpha = [0.5, 1.0]
colors = ['green', '#5A92DE', 'orange', 'pink', 'orange', 'black', 'gray', 'pink']
linestyle = ['-', '--', '-.', ':']

fig, axes = plt.subplots(1, len(sizes), figsize=(10, 2), sharex=True, sharey=True)

c_ref = get_config(T, dts[0], dts[0], noise, sizes[0], XN, TN, _extra_str, shape, **config_kwargs)
data_ref = np.load(c_ref.pinns_input_file, allow_pickle=True).item()

# Plotting
for i, size in enumerate(sizes):
    ax = axes[i]
    ax.plot(data_ref['t_star']*1e3, data_ref['phi_e'][:, n], color='red', alpha=0.5, linestyle='--', label='Reference', zorder=1)
    for i_dt, dti in enumerate(dts):
        sn = dti
        c = get_config(T, dti, sn, noise, size, XN, TN, _extra_str, shape, **config_kwargs)
        data = np.load(c.pinns_input_file, allow_pickle=True).item()
        color = colors[i_dt % len(colors)]
        z = 2 if dti!=5e-5 else 1
        ax.plot(data['t_star']*1e3, data['phi_e'][:, n], color=color, alpha=0.9 - 0.1*i_dt, zorder=z)

    ax.set_ylim([np.min(data_ref['phi_e'][:, n])*0.9, np.max(data_ref['phi_e'][:, n])*1.2])
    ax.grid(True, which="both", ls="--")
    ax.set_title(f'{size}k', fontsize=10)
    ax.set_xticks([0, 1])

legend_lines = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(len(dts))]
legend_lines.append(plt.Line2D([0], [0], color='red', lw=2, linestyle='--'))
ratio = [int(dti/dts[0])  for dti in dts]
ratio = [r if r != 1 else '' for r in ratio]
legend_labels = [fr'{r_dt}dt' for r_dt in ratio]
legend_labels.append('dt, 164k') 

fig.legend(legend_lines, legend_labels, loc='upper right', bbox_to_anchor=(0.95, 0.87), fontsize=10)
axes[int(len(sizes)//2)].set_xlabel(r"Time ($10^{-3}$ s)", fontsize=12)
axes[0].set_ylabel(r'$\varphi_e$', fontsize=12)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.subplots_adjust(right=0.8) 
plt.savefig('error_dts_sizes_0.999.pdf')
plt.show()
