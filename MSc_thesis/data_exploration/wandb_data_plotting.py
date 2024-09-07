import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


path = '../../wandb_data'

# Plot T_windows:
# Load the CSV data
# file_path_data = f'{path}/wandb_export_2024-08-06T11_09_18.205+01_00.csv'
# df_data = pd.read_csv(file_path_data)
# file_path_res = f'{path}/wandb_export_2024-08-06T11_10_00.967+01_00.csv'
# df_res = pd.read_csv(file_path_res)

# # Extract the columns of interest
# steps_res = df_data['Step']
# res_loss = df_data['hemlholtz_3d_pink_noise_s=6_T_windows_corrected_continuous - res_loss']

# steps_data = df_res['Step']
# data_loss = df_res['hemlholtz_3d_pink_noise_s=6_T_windows_corrected_continuous - data_loss']

# # Plot the da
# fig, ax1 = plt.subplots(figsize=(6, 3), dpi=100)

# data_line, = ax1.plot(steps_data, data_loss, 'b-', label=r'$\mathcal{L}_{\text{data}}(\theta)$', alpha=0.6)
# ax1.set_xlabel('Iterations')
# ax1.set_ylabel(r'$\mathcal{L}_{\text{data}}(\theta)$', zorder=2)
# ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax1.set_yscale('log')
# ax1.set_ylim([0.8e-5,1e-1])
# ax1.set_xlim([0,max(steps_res)])
# # ax1.grid(True)

# ax2 = ax1.twinx()
# residual_line, = ax2.plot(steps_res, res_loss, 'r-', label=r'$\mathcal{L}_{\text{r}}(\theta)$', alpha=0.6, zorder=1)
# ax2.set_ylabel(r'$\mathcal{L}_{\text{r}}(\theta)$')
# ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax2.set_yscale('log')
# ax2.set_ylim([0.8e-7,1e-4])
# ax2.set_xlim([0,max(steps_res)])

# ax1.spines['top'].set_visible(False)
# ax2.spines['top'].set_visible(False)

# ax1.legend(loc='center left', bbox_to_anchor=(0.3, 1.1), prop={'size': 8})
# ax2.legend(loc='center left', bbox_to_anchor=(0.5, 1.1), prop={'size': 8})

# # ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# # ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# # ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# fig.tight_layout()
# plt.show()

# Plot comparison sizes:

# file_path_res = f'{path}/res_loss_wandb_extract.csv'
# file_path_data = f'{path}/data_loss_wandb_extract.csv'

# df_data = pd.read_csv(file_path_data)
# df_res = pd.read_csv(file_path_res)

# steps = df_data['Step']
# res_loss_6 = df_res['resolution=6 - res_loss']
# res_loss_5 = df_res['resolution=5 - res_loss']
# data_loss_6 = df_data['resolution=6 - data_loss']
# data_loss_5 = df_data['resolution=5 - data_loss']

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=100)

# # Plot data loss for each resolution on the left subplot
# ax1.plot(steps, data_loss_6, 'b-', zorder=2, alpha=0.6)
# ax1.plot(steps, data_loss_5, 'r-', zorder=2, alpha=0.6)
# ax1.set_xlabel('Iterations')
# ax1.set_ylabel(r'$\mathcal{L}_{\text{data}}(\theta)$')
# ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax1.set_yscale('log')
# ax1.spines['top'].set_visible(False)

# # Plot residual loss for each resolution on the right subplot
# ax2.plot(steps, res_loss_5, 'r-', label='Resolution 5', zorder=1, alpha=0.6)
# ax2.plot(steps, res_loss_6, 'b-', label='Resolution 6', zorder=1, alpha=0.6)
# ax2.set_xlabel('Iterations')
# ax2.set_ylabel(r'$\mathcal{L}_{\text{r}}(\theta)$')
# ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax2.set_yscale('log')
# ax2.spines['top'].set_visible(False)

# # ax1.legend(loc='center left', bbox_to_anchor=(0.6, 1.1), prop={'size': 8})
# ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), prop={'size': 8})

# # Adjust layout to make space for the legend
# fig.tight_layout() #rect=[0, 0, 0.85, 1])

# # Show the plot
# plt.show()


# Plot showing white noise training:

# file_path_res = f'{path}/res_loss_WRONG.csv'
# file_path_data = f'{path}/data_loss_WRONG.csv'

# df_data = pd.read_csv(file_path_data)
# df_res = pd.read_csv(file_path_res)

# steps_res = df_data['Step']
# steps_data = df_res['Step']
# res_loss = df_res['hemlholtz_32k_uev1 - res_loss']
# data_loss = df_data['hemlholtz_32k_uev1 - data_loss']

# # Plot the da
# fig, ax1 = plt.subplots(figsize=(6, 3), dpi=100)

# data_line, = ax1.plot(steps_data, data_loss, 'b-', label=r'$\mathcal{L}_{\text{data}}(\theta)$', alpha=0.6)
# ax1.set_xlabel('Iterations')
# ax1.set_ylabel(r'$\mathcal{L}_{\text{data}}(\theta)$', zorder=2)
# ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax1.set_yscale('log')
# ax1.set_ylim([5e-4,1e-2])
# ax1.set_xlim([0,max(steps_res)])
# ax1.set_yticks([1e-3, 1e-2])
# # ax1.grid(True)

# ax2 = ax1.twinx()
# residual_line, = ax2.plot(steps_res, res_loss, 'r-', label=r'$\mathcal{L}_{\text{r}}(\theta)$', alpha=0.6, zorder=1)
# ax2.set_ylabel(r'$\mathcal{L}_{\text{r}}(\theta)$')
# ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax2.set_yscale('log')
# ax2.set_ylim([1e0,1e5])
# ax2.set_xlim([0,max(steps_res)])

# ax1.spines['top'].set_visible(False)
# ax2.spines['top'].set_visible(False)

# ax1.legend(loc='center left', bbox_to_anchor=(0.3, 1.1), prop={'size': 8})
# ax2.legend(loc='center left', bbox_to_anchor=(0.5, 1.1), prop={'size': 8})

# # ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# # ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# # ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# fig.tight_layout()
# plt.show()


# Plot showing r_s convergence:
# import numpy as np
# file_path = f'{path}/rs_prediction_iPINN.csv'

# df_data = pd.read_csv(file_path)

# steps_data = df_data['Step']
# data_loss = df_data['helm_3d_r=1_pink_noise_Q_rs_squared - rs_pred'].values

# # Plot the da
# fig, ax1 = plt.subplots(figsize=(6, 3), dpi=100)


# std_prop = int(np.shape(data_loss)[0] * 0.7) # last 10%
# mean, std = np.mean(data_loss[std_prop:]), np.std(data_loss[std_prop:])
# print(mean, std)
# mean_curve = np.array([mean]*len(steps_data))
# std_curve = np.array([std]*len(steps_data))
# ax1.plot(steps_data, mean_curve, color='red', alpha=0.5, label=r'$\mu_{r_{s}}$' + f'={mean:.3f}', linestyle='--')
# ax1.fill_between(np.array(steps_data), mean_curve + std_curve, mean_curve - std_curve, label=r'$\sigma_{r_{s}}$ (30%)', alpha=0.5, color='orange')
# ax1.scatter(steps_data, data_loss, color='k', alpha=0.6, s=1, zorder=2, label=r'$r_{s}$')
# ax1.set_xlabel('Iterations')
# ax1.set_ylabel(r'$r_{s}$')
# ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# # ax1.set_yscale('log')
# ax1.set_ylim([0.00000,0.02])
# ax1.set_xlim([0,max(steps_data)])
# # ax1.set_yticks([1e-3, 1e-2])
# ax1.grid(axis='y', alpha=0.3)

# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.legend(loc='center right', bbox_to_anchor=(1.0, 1.0), prop={'size': 8})

# # ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# # ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# # ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# fig.tight_layout()
# plt.show()

# Derive additional errors for gamma and rs:
# import numpy as np
# csv_file_name = 'gamma_pred_with_rs_run.csv'
# column_gamma_alone = 'helm_3d_r=1_pink_noise_Q_gamma_squ_longer - gamma_pred'
# column_gamma_with_rs = 'helm_3d_r=1_pink_noise_Q_gamma_rs_squared_longer - gamma_pred'

# csv_file_name = 'rs_pred_clipped_rs_gamma.csv'
# column_gamma_with_rs = 'helm_3d_r=1_pink_noise_Q_gamma_rs_CLIPPED_sigmoid - rs_pred'

# csv_file_name = 'gamma_and_rs_rs_pred.csv'
# column_rs_with_gamma = 'helm_3d_r=1_pink_noise_noQ_gamma_rs_NOTCLIPPED - rs_pred'

# csv_file_name_g = 'gamma_and_rs_gamma_pred.csv'
# column_gamma_with_rs = 'helm_3d_r=1_pink_noise_noQ_gamma_rs_NOTCLIPPED - gamma_pred'

# csv_file_name = 'rs_predicted_with_gamma.csv'
# column_rs_with_gamma = 'helm_3d_r=1_pink_noise_Q_gamma_rs_squared_longer - rs_pred'

# file_path = f'{path}/{csv_file_name}'
# file_path_g = f'{path}/{csv_file_name_g}'

# df_data = pd.read_csv(file_path)
# df_data_g = pd.read_csv(file_path_g)

# steps_data = df_data['Step']
# data_loss = df_data[column_rs_with_gamma].values
# data_loss_g = df_data_g[column_gamma_with_rs].values

# # # Plot the data
# std_prop = int(np.shape(data_loss)[0] * 0.7) # last 10%
# mean, std = np.mean(data_loss[std_prop:]), np.std(data_loss[std_prop:])
# mean_g, std_g = np.mean(data_loss_g[std_prop:]), np.std(data_loss_g[std_prop:])
# print(mean, std)
# mean_curve = np.array([mean]*len(steps_data))
# std_curve = np.array([std]*len(steps_data))
# mean_curve_g = np.array([mean_g]*len(steps_data))
# std_curve_g = np.array([std_g]*len(steps_data))
# fig, ax1 = plt.subplots(figsize=(6, 3), dpi=100)
# ax1.plot(steps_data, mean_curve, color='red', alpha=0.5, label=r'$\mu_{r_{s}}$' + f'={mean:.3f}', linestyle='--')
# ax1.plot(steps_data, mean_curve_g, color='blue', alpha=0.5, label=r'$\mu_{\gamma_{s}}$' + f'={mean_g:.3f}', linestyle='--')
# ax1.fill_between(np.array(steps_data), mean_curve + std_curve, mean_curve - std_curve, label=r'$\sigma_{r_{s}}$ $(30\%)$', alpha=0.5, color='orange')
# ax1.fill_between(np.array(steps_data), mean_curve_g + std_curve_g, mean_curve_g - std_curve_g, label=r'$\sigma_{\gamma_{s}}$ $(30\%)$', alpha=0.5, color='orange')
# ax1.scatter(steps_data, data_loss, color='k', alpha=0.6, s=1, zorder=2, label=r'$r_{s}$')
# ax1.scatter(steps_data, data_loss_g, color='k', alpha=0.6, s=1, zorder=2, label=r'$\gamma_{s}$')
# ax1.set_xlabel('Iterations')
# ax1.set_ylabel('PDE parameters [a.u.]')
# ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# # ax1.set_yscale('log')
# ax1.set_ylim([-5,14])
# ax1.set_xlim([0,max(steps_data)])
# # ax1.set_yticks([1e-3, 1e-2])
# ax1.grid(axis='y', alpha=0.3)

# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.legend(loc='center right', bbox_to_anchor=(1.0, 1.0), prop={'size': 8})

# # ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# # ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# # ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# fig.tight_layout()
# plt.show()


# Recreate the plot
# fig, ax1 = plt.subplots(figsize=(6, 3), dpi=100)

# # Plotting mean and standard deviation curves for r_s on ax1
# # line1, = ax1.plot(steps_data, mean_curve, color='red', alpha=0.5, label=r'$\mu_{r_{s}}$' + f'={mean*1e3:.2f}$\pm${std*1e3:.2f}' + r'x$10^{-3}$', linestyle='--')
# line1 = ax1.fill_between(steps_data, mean_curve + std_curve, mean_curve - std_curve, color='red', alpha=0.5, label=r'$\mu_{r_{s}}$' + f'={mean*1e3:.2f}$\pm${std*1e3:.2f}' + r'x$10^{-3}$') #label=r'$\sigma_{r_{s}}$ $(30\%)$')
# scat1 = ax1.scatter(steps_data, data_loss, color='brown', alpha=0.3, s=1, zorder=2) #, label=r'$r_{s}$')
# # ax1.scatter(steps_data, data_loss, color='brow', alpha=0.1, s=1, zorder=2) #, label=r'$r_{s}$')

# # Create a twin axis
# ax2 = ax1.twinx()

# # Plotting mean and standard deviation curves for gamma_s on ax2
# # ax2.plot(steps_data, mean_curve_g, color='blue', alpha=0.5, label=r'$\mu_{\gamma_{s}}$' + rf'={mean_g:.2f}$\pm${std_g:.2f}', linestyle='--')
# line2 = ax2.fill_between(steps_data, mean_curve_g + std_curve_g, mean_curve_g - std_curve_g, color='blue', alpha=0.3, label=r'$\mu_{\gamma_{s}}$' + rf'={mean_g:.2f}$\pm${std_g:.2f}')
# scat2 = ax2.scatter(steps_data, data_loss_g, color='#093C82', alpha=0.3, s=1, zorder=2) #, label=r'$\gamma_{s}$')

# # Setting labels and formatting for ax1
# ax1.set_xlabel('Iterations')
# ax1.set_ylabel(r'$r_{s}$ [m]', color='k')
# ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax1.set_ylim([-0.0005, 0.008])
# ax1.set_xlim([0, max(steps_data)])
# ax1.grid(axis='y', alpha=0.3)
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)

# # Setting labels and formatting for ax2
# ax2.set_ylabel(r'$\gamma_{s}$ [$s^{-1}$]', color='k')
# ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax2.set_ylim([9, 11])
# ax1.set_xticks([0,0.5e5,1e5,1.5e5,2e5])

# # Adding legends for both axes
# lines = [line1, scat1, line2, scat2]
# labels = [line.get_label() for line in lines]
# ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.8, 1), prop={'size': 8}, ncol=1)

# # Tight layout and show plot
# fig.tight_layout()
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Data provided
# data = [
#     (164, 5e-06, 5.479420608221186e-06),
#     (164, 1e-05, 1.4016233646296696e-05),
#     (164, 5e-05, 0.0001269526348100143),
#     (131, 1e-06, 0.00023329978056271686),
#     (131, 5e-06, 0.0002336440115846175),
#     (131, 1e-05, 0.00023420301921380963),
#     (131, 5e-05, 1.6995480076556807e+24),
#     (80, 1e-06, 0.0003056252692458993),
#     (80, 5e-06, 0.0003068876469373113),
#     (80, 1e-05, 0.000308927940605572),
#     (80, 5e-05, 6185581843878.701),
#     (65, 1e-06, 0.00034352869531477235),
#     (65, 5e-06, 0.00034437700562009417),
#     (65, 1e-05, 0.0003454671218185618),
#     (65, 5e-05, 4809.323153102527),
#     (40, 1e-06, 0.0003668424559494051),
#     (40, 5e-06, 0.00036760571903573293),
#     (40, 1e-05, 0.00036857453988301545),
#     (40, 5e-05, 77971013.25090858)
# ]



# # Separate data into arrays
# resolutions, dts, errors = zip(*data)

# # Convert to numpy arrays for plotting
# dts = np.array(dts)
# errors = np.array(errors)

# unique_resolutions = sorted(set(resolutions))

# # Assign a color to each resolution
# colors = plt.cm.viridis(np.linspace(0, 1, len(unique_resolutions)))

# # Plotting with different colors for each resolution
# plt.figure(figsize=(10, 6))

# for res, color in zip(unique_resolutions, colors):
#     # Filter data for the current resolution
#     indices = [i for i, r in enumerate(resolutions) if r == res]
#     plt.loglog(dts[indices], errors[indices], 'o-', color=color, label=f'Res: {res}')

# # Customize plot
# plt.ylim([1e-7, 1e-3])
# plt.xlabel('Time Step (dt)')
# plt.ylabel('Absolute Error')
# plt.title('Log-Log Plot of Absolute Error vs. Time Step for Different Resolutions')
# plt.grid(True, which="both", ls="--")
# plt.legend(title="Resolutions")
# plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
import matplotlib.ticker as ticker

dataset_names = ['data', 'res', 'H', 'n']
# loss_dataset = 'data_loss'

min_values_dict = {}
xticks_dict = {} 

for loss_dataset in dataset_names:
    # file_path = os.path.join(path, f'surface_constraints/{loss_dataset}_loss_sweep_surface.csv')
    # file_path = os.path.join(path, f'surface_constraints_r=1/{loss_dataset}_loss_r=1_surface_constraints.csv')
    file_path = os.path.join(path, f'surf_constraints/{loss_dataset}_loss_surf_constraints.csv')
    # file_path = os.path.join(path, f'surf_constraints_r=1/{loss_dataset}_loss_surf_r=1.csv')
    # file_path = os.path.join(path, f'test/{loss_dataset}_loss.csv')
    loss_df = pd.read_csv(file_path).dropna() #axis=0, how='all')
    cols = loss_df.columns[1::3]
    print(len(cols))

    values_array = loss_df[cols].values
    min_values_array = values_array.min(axis=0)
    min_values_dict[loss_dataset] = min_values_array
    xticks_dict[loss_dataset] = [min(min_values_array), max(min_values_array)]

# min_values_dict['nH'] = min_values_dict['n'] + min_values_dict['H']
# xticks_dict['nH'] = [min(min_values_dict['nH']), max(min_values_dict['nH'])]
# combinations = [
#     # ('H', 'res'),
#     ('nH', 'res'),
#     ('data', 'res'),
#     ('data', 'nH'),
#     # ('data', 'n'),
#     # ('H', 'n'),
# ]

combinations = [
    ('H', 'res'),
    ('n', 'res'),
    ('H', 'n'),
    ('data', 'H'),
    ('data', 'n'),
    ('data', 'res'),
]

fig, axs = plt.subplots(2, 3, figsize=(10, 6), dpi=150)
axs = axs.flatten()

for i, (x_key, y_key) in enumerate(combinations):
    correlation_coefficient, p_value = pearsonr(min_values_dict[x_key], min_values_dict[y_key])
    print(correlation_coefficient, p_value)
    
    min_max_x = xticks_dict[x_key]
    min_max_y = xticks_dict[y_key]
    axs[i].scatter(min_values_dict[x_key], min_values_dict[y_key], alpha=0.7, label=f'r = {correlation_coefficient:.2f}')
    if x_key == 'res':
        x_key = 'r'
    if y_key == 'res':
        y_key = 'r'
    if x_key == 'nH':
        x_key = 'curv'
    if y_key == 'nH':
        y_key = 'curv'
    
    axs[i].set_xticks(min_max_x)
    axs[i].set_yticks(min_max_y)
    # axs[i].tick_params(axis='x', which='major', size=6)
    # axs[i].tick_params(axis='y', which='major', size=6)
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')
    axs[i].set_xlabel(r'$\mathcal{L}_{%s}$' % x_key, fontsize=15)
    axs[i].set_ylabel(r'$\mathcal{L}_{%s}$' % y_key, fontsize=15)
    # if x_key == 'data':
    #     axs[i].yaxis.set_minor_locator(ticker.NullLocator())
    # elif y_key == 'data':
    #     axs[i].xaxis.set_minor_locator(ticker.NullLocator())
    # else:
    #     axs[i].xaxis.set_minor_locator(ticker.NullLocator())
    #     axs[i].yaxis.set_minor_locator(ticker.NullLocator())
    # axs[i].xaxis.set_major_formatter(ticker.ScalarFormatter())
    # axs[i].yaxis.set_major_formatter(ticker.ScalarFormatter())
    axs[i].legend()
    axs[i].grid(False)

plt.tight_layout()
plt.show()


sns.set_theme(style="white")

df = pd.DataFrame(min_values_dict)
df = df.rename(columns={'res': 'r'})
formatted_columns = [r'$\mathcal{L}_{%s}$' % name for name in df.columns]
df.columns = formatted_columns

corr_matrix = df.corr()
np.fill_diagonal(corr_matrix.values, 0)

mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

fig = plt.figure(figsize=(5, 4), dpi=150)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
ax = sns.heatmap(corr_matrix, 
                 mask=mask, 
                 cmap=cmap, 
                 vmin=-1.0, 
                 vmax=1.0, 
                 center=0,
                 square=True, 
                 linewidths=.5, 
                 cbar_kws={"shrink": .8})
cbar = ax.collections[0].colorbar
cbar.set_ticks([-1,0,1])
cbar.set_label('Correlation')
cbar.set_ticklabels(['-1', '0', '1']) 
ax.tick_params(axis='x', labelsize=14)  
ax.tick_params(axis='y', labelsize=14)

plt.show()


# plt.xlabel('Iterations')
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax1.set_ylim([-0.0005, 0.008])
# ax1.set_xlim([0, max(steps_data)])
# ax1.grid(axis='y', alpha=0.3)
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)

# # Setting labels and formatting for ax2
# ax2.set_ylabel(r'$\gamma_{s}$ [$s^{-1}$]', color='k')
# ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
# ax2.set_ylim([9, 11])
# plt.xticks([0,0.5e5,1e5,1.5e5,2e5])

# plt.ylabel('Minimum Value')
# plt.legend(title="Dataset")
# plt.title(f'(Pearson r = {correlation_coefficient:.4f})')
# plt.xlabel('Data Loss Minimum Values')
# plt.ylabel('Residual Loss Minimum Values')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid(True, which="both", ls="--")
# plt.show()
