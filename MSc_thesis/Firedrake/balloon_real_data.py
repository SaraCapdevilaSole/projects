import numpy as np
import os
import matplotlib.pyplot as plt

from src.config.params import noise, T, sn, dt, size, XN, TN, max_size, _extra_str, shape, config_kwargs
from src.utils.utils import plot_sample_curves, get_config
from src.utils.ode_solvers import EulerSolver, RK4Solver, ODEIntSolver, HeunsSolver
from src.utils.balloon_model import Balloon
from src.utils.utils_fc import compute_FC_matrix, compute_cdist_matrix, compute_FC_average, compute_static_FC, normalise_bold_data

path_data = '/vol/bitbucket/sc3719/firedrake_simulation/data/output_pinn_inflated_real_data'

# For first report plot:
# pinn_file = 'rs_and_Q0_prediction.npy'

# Second:
pinn_file = 'rs_and_Q0_prediction_2s.npy'
data = np.load(os.path.join(path_data, pinn_file), allow_pickle=True).item()

savedir = 'simulations/firedrake'
os.makedirs(savedir, exist_ok=True)
plot_sample_curves(data_array=data['u_ref'], time_array=data['t_star'], n_samples_to_plot=data['u_ref'].shape[1], total_samples=data['u_ref'].shape[1], sn=sn, savefig=f'{savedir}/phi_XN={XN}_TN={TN}.png')

x0 = np.array([0,1,1,1,0])

data['t_star'] -= min(data['t_star'])
print(data['t_star'].shape)
print(data['u_ref'].shape)
BalloonODE = Balloon(data['t_star'], data=data['u_ref'])
n_parcellations: int = data['u_ref'].shape[1]
odes = BalloonODE.ode_system
# Solvers 
ode_solver = EulerSolver(odes)
solution = ode_solver.solve(x0, sn, T, n_parcellations)
y = solution[:, -1, :]
del BalloonODE, n_parcellations, odes, ode_solver

print(data['t_star'].shape)
print(data['u_pred'].shape)
BalloonODE = Balloon(data['t_star'], data=data['u_pred'])
n_parcellations: int = data['u_pred'].shape[1]
odes = BalloonODE.ode_system
# Solvers 
ode_solver = EulerSolver(odes)
solution_pred = ode_solver.solve(x0, sn, T, n_parcellations)
y_pred = solution_pred[:, -1, :]

print(y.shape)
savedir = 'simulations/bold'
os.makedirs(savedir, exist_ok=True)
plot_sample_curves(data_array=y[:data['t_star'].shape[0], :], time_array=data['t_star'], n_samples_to_plot=data['u_ref'].shape[1], total_samples=data['u_ref'].shape[1], sn=sn, savefig=f'{savedir}/bold_{pinn_file}.png')
plot_sample_curves(data_array=y_pred[:data['t_star'].shape[0], :], time_array=data['t_star'], n_samples_to_plot=data['u_pred'].shape[1], total_samples=data['u_pred'].shape[1], sn=sn, savefig=f'{savedir}/bold_PINN_{pinn_file}.png')

### FC:
solution = normalise_bold_data(data=y, filename=None)
fc_data = compute_static_FC(solution, kind='correlation', plot_it=True, savename='sim')
# solution = normalise_bold_data(data=data['u_ref'], filename=None)
# fc_data = compute_static_FC(data['u_ref'], kind='correlation', plot_it=True, savename='sim')
np.fill_diagonal(fc_data, 0)

del solution
solution = normalise_bold_data(data=y_pred, filename=None)
fc_pinn = compute_static_FC(solution, kind='correlation', plot_it=True, savename='pinn')
# solution = normalise_bold_data(data=data['u_pred'], filename=None)
# fc_pinn = compute_static_FC(data['u_pred'], kind='correlation', plot_it=True, savename='pinn')
np.fill_diagonal(fc_pinn, 0)

fc_error = np.abs(fc_data-fc_pinn)

vmin = -1 # min(fc_data.min(), fc_pinn.min())
vmax = 1 #max(fc_data.max(), fc_pinn.max())

fig, axes = plt.subplots(1, 3, figsize=(7, 3), dpi=100)

im1 = axes[0].imshow(fc_data, vmin=vmin, vmax=vmax, cmap='coolwarm', interpolation='nearest')
axes[0].set_title('Data')
# axes[0].axis('off')
axes[0].set_xticks([])
axes[0].set_yticks([])

cbar = fig.colorbar(im1, ax=axes[0], orientation='vertical', shrink=0.3)
cbar.set_ticks([-1, 0, 1])

im2 = axes[1].imshow(fc_pinn, vmin=vmin, vmax=vmax, cmap='coolwarm', interpolation='nearest')
axes[1].set_title('Predicted')
# axes[1].axis('off')
axes[1].set_xticks([])
axes[1].set_yticks([])

cbar = fig.colorbar(im1, ax=axes[1], orientation='vertical', shrink=0.3)
cbar.set_ticks([-1, 0, 1])

im3 = axes[2].imshow(fc_error, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
axes[2].set_title('Error')
# axes[2].axis('off')
axes[2].set_xticks([])
axes[2].set_yticks([])
print(np.mean(fc_error))

cbar = fig.colorbar(im3, ax=axes[2], orientation='vertical', shrink=0.3)
cbar.set_ticks([0, 0.5, 1])

plt.tight_layout()
plt.savefig('fc_pinn_minus_fem.png')



num_plots = 3
indices = [0, 10, -1]

fig, axes = plt.subplots(1, num_plots, figsize=(10, 3), dpi=100, sharey=True, sharex=True)

for idx, i in enumerate(indices):
    ax = axes[idx] 
    ax.scatter(data['t_star'], data['u_ref'][:, i], color='black', label='Data', s=5)
    ax.plot(data['t_star'], data['u_pred'][:, i], color='#F8522D', label='Prediction', linewidth=2)
    # ax.set_xticks([1,1.05,1.1])
    
    ax.grid(which='major',alpha=0.3)


axes[-1].legend(loc='lower right')
axes[1].set_xlabel('Time [s]', fontsize=14)
axes[0].set_ylabel(r'$\varphi_e$', fontsize=14)
axes[0].set_yticks([-0.5,0,0.5])
plt.tight_layout()
plt.savefig('comparison_solution.png')


