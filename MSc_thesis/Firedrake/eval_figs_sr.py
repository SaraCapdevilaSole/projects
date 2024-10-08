import matplotlib.pyplot as plt
import meshio
import numpy as np
import os

from src.utils.utils import get_config
from matplotlib.ticker import FuncFormatter
from src.config.params import noise, T, sn, dt, size, XN, TN, max_size, _extra_str, shape, config_kwargs

runs = [
    {'sn': sn, 'size': 131, 'dt': dt, 'T': T, 'extra': _extra_str},
    {'sn': sn, 'size': 164, 'dt': dt, 'T': T, 'extra': _extra_str},
]

N = 10

colors = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'black', 'gray', 'pink']
markers = ['+', 'o', 'x', '*', '^']
linestyle = ['-', '--', '-.', ':']

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
for ri, r in enumerate(runs):
    dt = r['dt']
    size = r['size']
    sn = r['sn']
    T = r['T']
    extra = r['extra']
    c = get_config(T, dt, sn, noise, size, XN, TN, _extra_str, shape, **config_kwargs)
    output_path = c.output_phi_e
    pinns_input_file = c.pinns_input_file
    data = np.load(pinns_input_file, allow_pickle=True).item()
    color = colors[ri % len(colors)]
    marker = markers[ri % len(markers)]
    np.random.seed(42)
    random_indices = np.random.randint(0, data['phi_e'].shape[1], N)
    ls = linestyle[ri % len(linestyle)]
    for i in range(N):
        if i==0:
            alpha=1
        else:
            alpha=0.1
            color = 'gray'
        n = random_indices[i]
        if i == 0:
            labels = [rf'164k', rf'131k']
            label = labels[ri]
        else:
            label = None
        plt.plot(data['t_star'][:], data['phi_e'][:,n], label=label, color=color, linestyle=ls, alpha=alpha)

format_sigfig = lambda x: float(f'{x:.1g}')
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([format_sigfig(min(data['phi_e'][:,random_indices[0]])*1.1), format_sigfig((max(data['phi_e'][:,random_indices[0]])*0.9 + min(data['phi_e'][:,random_indices[0]])*1.1)/2), format_sigfig(max(data['phi_e'][:,random_indices[0]])*0.9)])
ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel("Time (s)")
plt.ylabel(r'$\varphi_e$', fontsize=12)
plt.tight_layout()

config_kwargs_str = '_'.join([f'{key}={value}' for key, value in config_kwargs.items()])
filename = f'simulations/error_dt_XN={XN}_TN={TN}_{_extra_str}_T={T}_{shape}_{config_kwargs_str}.png'
fig.savefig(filename, bbox_inches="tight", dpi=300)
plt.show()
