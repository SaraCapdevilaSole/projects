from src.utils.utils_downsampling import calculate_downsampling_error, balloon_downsample_normalise, load_data
from src.utils.utils_fc import compute_static_FC
from src.config.config_PINNs import OUTPUT_PATH
from src.config.config_balloon import T, dt_files
from src.config.config_fc import KIND

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

T = int(60*10 - 20)
t_avoid = 20
n = 53 # parcellation to plot
data_path = f'{OUTPUT_PATH}/nft_firedrake_{58000}.npy'
colors = ['#8D8C8C', '#E34E04', '#BC23D1', '#158FB6', '#136416', 'blue']
trs = [0.02, 0.04, 0.1, 0.2] # new temporal resolution

def main(trs):
    u_ref, t_star, coords = load_data(data_path)
    
    y, t_o = balloon_downsample_normalise(t_star, u_ref, coords, tr=dt_files, t_avoid=t_avoid, T=T)

    y_list = [y]
    times = [t_o]
    errors = []

    for i, tr in enumerate(trs):
        resolution = int(tr/dt_files)
        yd, td = balloon_downsample_normalise(t_star, u_ref, coords, tr=tr, t_avoid=t_avoid, T=T)
        e1 = np.mean([calculate_downsampling_error(y[:, n], downsampled_data=yd[:, n], resolution=resolution)['relative'] for n in range(len(yd[0, :]))])
        y_list.append(yd)
        times.append(td)
        errors.append(e1)

    return y_list, times, errors


if __name__ == "__main__":

    y_list, times, errors = main(trs)

    fig = plt.figure(figsize=(15, 8), dpi=300)
    gs = fig.add_gridspec(2, len(trs) + 1, height_ratios=[1, 0.2], hspace=-0.5)
    FC_o = compute_static_FC(y_list[0], kind=KIND, plot_it=False)
    ax = fig.add_subplot(gs[0, 0])
    plt.imshow(FC_o, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(shrink=0.2)  
    plt.clim(-1,1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'tr: {0.01}s')

    empirical_fc_flat = FC_o[np.triu_indices_from(FC_o, k=1)]
    results = []
    for i, y in enumerate(y_list[1:]):
        i += 1
        FCmatrix = compute_static_FC(y, kind=KIND, plot_it=False)
        simulated_fc_flat = FCmatrix[np.triu_indices_from(FCmatrix, k=1)]

        ax = fig.add_subplot(gs[0, i])
        plt.imshow(FCmatrix, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(shrink=0.2)  
        plt.clim(-1,1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'tr: {trs[i-1]}s')
        correlation, p_value = pearsonr(simulated_fc_flat, empirical_fc_flat)
        results.append((correlation, p_value))

    ax7 = fig.add_subplot(gs[1, :])
    ax7.plot(times[0], y_list[0][:,n], label=f"original tr={dt_files}s", color=colors[0])
    for i, y in enumerate(y_list[1:]):
        ax7.plot(times[i+1], y[:,n],  label=r"tr={:.2}s, $\epsilon$={:.1e}, r={:.1}".format(trs[i], errors[i], results[i][0]), color=colors[i+1])
        
    ax7 = plt.gca()
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    ax7.set_xlim([times[0][0], times[0][-1]])
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Signal')
    ax7.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()

    plt.show()

    
    
