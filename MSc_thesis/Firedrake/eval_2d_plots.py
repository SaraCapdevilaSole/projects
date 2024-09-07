import matplotlib.pyplot as plt
import meshio
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter
from src.config.config import SphereConfig, SquareConfig
from src.config.params import noise, T, sn, dt, size, XN, TN, max_size, _extra_str, shape

n = 0

runs = [
    {'sn': sn, 'size': 100, 'dt': dt, 'T': T, 'extra': _extra_str}, 
    {'sn': sn, 'size': 90, 'dt': dt, 'T': T, 'extra': _extra_str}, 
    # {'sn': 1e-4, 'size': 50, 'dt': 1e-4, 'T': 0.01, 'noise'=True},
]

def plot_comparison(data, fname, dt, data_prev, dt_prev):
    coords = data['mesh_coordinates']
    u_prev = data_prev['phi_e']
    u_now = data['phi_e']
    x = coords[:, 0]
    y = coords[:, 1]

    time_frames = [0.001, 0.005]
    files_frames = [1, 10]

    u_error = np.abs(u_now - u_prev)  # Calculate element-wise error

    data = [
        [(u_now[files_frames[0], :], f'dt={dt}', time_frames[0]),
         (u_prev[files_frames[0], :], f'dt={dt_prev}', time_frames[0]),
         (u_error[files_frames[0], :], 'Absolute Error', time_frames[0])],
        [(u_now[files_frames[1], :], f'dt={dt}', time_frames[1]),
         (u_prev[files_frames[1], :], f'dt={dt_prev}', time_frames[1]),
         (u_error[files_frames[1], :], 'Absolute Error', time_frames[1])],
        [(np.mean(u_now, axis=0), f'dt={dt}', None),
         (np.mean(u_prev, axis=0), f'dt={dt_prev}', None),
         (np.mean(u_error, axis=0), 'Absolute Error', None)]
    ]

    title_fontsize = 14

    for n in range(3):
        fig3 = plt.figure(figsize=(15, 5))
        data_fig = data[n]
        
        # Loop to create subplots
        for i, (u_data, title, time_frame) in enumerate(data_fig, start=1):
            ax = fig3.add_subplot(1, 3, i)
            scatter = ax.scatter(x, y, c=u_data, cmap='coolwarm')
            
            if time_frame is not None:
                ax.set_title(f'{title} at t={time_frame:.3f}s', fontsize=title_fontsize)
            else:
                ax.set_title(f'{title} Average', fontsize=title_fontsize)
            
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)  # Remove the frame
            
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)  # Force scientific notation
            cbar.ax.yaxis.set_major_formatter(formatter)
        
        plt.tight_layout()
        
        fig_path = os.path.join(f'{fname}_comparison_to_dt={dt_prev}_{n}.pdf')
        fig3.savefig(fig_path, bbox_inches="tight", dpi=300)

data_prev = None

fig, ax = plt.subplots(figsize=(10, 4))
for ri, r in enumerate(runs):
    dt = r['dt']
    size = r['size']
    sn = r['sn']
    T = r['T']
    extra = r['extra']
    c = SquareConfig(T=T, dt=dt, sampling_nom=sn, noise=noise, size=size, XN=XN, TN=TN, extra=extra)
    output_path = c.output_phi_e
    pinns_input_file = c.pinns_input_file

    data = np.load(pinns_input_file, allow_pickle=True).item()
    # print(data)
    if data_prev is None:
        dt_prev = dt
        data_prev = data.copy()
    else:
        plot_comparison(data=data, data_prev=data_prev, fname=output_path, dt=dt, dt_prev=dt_prev)
        data_prev = data.copy()  # Update data_prev for the next iteration
