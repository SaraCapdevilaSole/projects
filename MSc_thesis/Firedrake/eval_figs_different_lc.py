import matplotlib.pyplot as plt
import meshio
import numpy as np
import os

from config import SphereConfig

runs = [
    {'name': 'lc=10_r=100', 'dt': ['TEST', 'TEST2']},
]

ls = ['-', '--', '-']

fig, ax = plt.subplots(figsize=(10, 4))
for r in runs:
    for i, test in enumerate(r['dt']):
        c = SphereConfig(dt=1e-5, T=0.01, sampling_nom=1e-4, other=r['name'], Test=test)
        data = np.load(c.pinns_input_file, allow_pickle=True).item()
        plt.plot(data['t_star'], data['phi_e'][:,0], label=f'dt={c.dt}s, {c.other.split("_")[0]}', linestyle=ls[i])

plt.ylim([data['phi_e'][:,0].min()*2, data['phi_e'][:,0].max()*2])
plt.xlim([data['t_star'].min(), data['t_star'].max()])
ax.legend(fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("Time (s)")
plt.ylabel(r'$\varphi_e$')
plt.tight_layout()
fig.savefig('simulations/error_test.png', bbox_inches="tight", dpi=300)
plt.show()