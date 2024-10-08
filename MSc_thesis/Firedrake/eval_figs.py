import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import meshio
import numpy as np
import os

from config import SphereConfig

c_hp = SphereConfig(dt=1e-6, T=0.1, sampling_nom=0.001)
c3 = SphereConfig(dt=1e-3, T=0.1, sampling_nom=0.01)
c4 = SphereConfig(dt=1e-4, T=0.1, sampling_nom=0.01)
c5 = SphereConfig(dt=1e-5, T=0.1, sampling_nom=0.01)
c6 = SphereConfig(dt=1e-6, T=0.1, sampling_nom=0.01)

data_hp = np.load(c_hp.pinns_input_file, allow_pickle=True).item()
data3 = np.load(c3.pinns_input_file, allow_pickle=True).item()
data4 = np.load(c4.pinns_input_file, allow_pickle=True).item()
data5 = np.load(c4.pinns_input_file, allow_pickle=True).item()
data6 = np.load(c4.pinns_input_file, allow_pickle=True).item()

fig = plt.figure(figsize=(10, 6)) 
plt.plot(data_hp['t_star'], data_hp['phi_e'][:,0], label=f'dt={c_hp.dt}s hp')
plt.scatter(data_hp['t_star'], data_hp['phi_e'][:,0])
plt.plot(data4['t_star'], data4['phi_e'][:,0], label=f'dt={c4.dt}s')
plt.scatter(data4['t_star'], data4['phi_e'][:,0])
plt.plot(data5['t_star'], data5['phi_e'][:,0], label=f'dt={c5.dt}s')
plt.scatter(data5['t_star'], data5['phi_e'][:,0])
plt.plot(data6['t_star'], data6['phi_e'][:,0], label=f'dt={c6.dt}s')
plt.scatter(data6['t_star'], data6['phi_e'][:,0], marker='x', color='black')
plt.legend()
fig.savefig('simulations/error.png', bbox_inches="tight", dpi=300)
plt.show()


u1 = data3['phi_e']
u2 = data4['phi_e']
u3 = data5['phi_e']

coords = data3['mesh_coordinates']

dt = 0.01
time_frames = [0.02, 0.04]
files_frames = [int(tr/dt) for tr in time_frames]
u_abs_diff = lambda tf: np.abs(u1[tf, :] - u2[tf, :])

data = [
[(u1[files_frames[0], :], '1e-3', time_frames[0]),
(u2[files_frames[0], :], '1e-4', time_frames[0]),
(u3[files_frames[0], :], '1e-5', time_frames[0])],
[(u1[files_frames[1], :], '1e-3', time_frames[1]),
(u2[files_frames[1], :], '1e-4', time_frames[1]),
(u3[files_frames[1], :], '1e-5', time_frames[1])],
[(u1[files_frames[1], :], '1e-3', time_frames[1]),
(u2[files_frames[1], :], '1e-4', time_frames[1]),
(u3[files_frames[1], :], '1e-5', time_frames[1])]
]

title_fontsize = 14

x = coords[:, 0]
y = coords[:, 1]
z = coords[:, 2]

for n in range(3):
    fig3 = plt.figure(figsize=(10, 23))
    data_fig = data[n]
    for i, (u_data, title, time_frame) in enumerate(data_fig, start=1):
        ax = fig3.add_subplot(1, 3, i, projection='3d')
        scatter = ax.scatter(x, y, z, c=u_data, cmap='coolwarm')
        if time_frame is not None:
            ax.set_title(f'{title}', fontsize=title_fontsize, y=1.15)
        else:
            ax.set_title(f'{title} Average', fontsize=title_fontsize, y=1.15)
        plt.grid(False)
        plt.axis('off')
        
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        cbar.ax.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    
    fig_path = f"simulations/third_attempt_pinns_{n}.pdf"
    fig3.savefig(fig_path, bbox_inches="tight", dpi=300)
