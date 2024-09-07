import numpy as np
import os
import matplotlib.pyplot as plt

from src.config.params import noise, T, sn, dt, size, XN, TN, max_size, _extra_str, shape, config_kwargs
from src.utils.utils import plot_sample_curves, get_config
from src.utils.ode_solvers import EulerSolver, RK4Solver, ODEIntSolver, HeunsSolver
from src.utils.balloon_model import Balloon
from src.utils.utils_fc import compute_FC_matrix, compute_cdist_matrix, compute_FC_average, compute_static_FC, normalise_bold_data

c = get_config(T, dt, sn, noise, size, XN, TN, _extra_str, shape, **config_kwargs)
data = np.load(c.pinns_input_file, allow_pickle=True).item()
# pinn_file = 'helm_subset_jm_ic=Q_dt=1e-6_tr=1e-4_T=1e-2_emb=50_samplers_nondim=True'
pinn_file = 'helm_subset_jm_ic=Q_dt=1e-6_T=0.01_emb50_nondim_repeated'
data_pinn = np.load(f'/vol/bitbucket/sc3719/firedrake_simulation/data/output_pinn_inflated/{pinn_file}.npy', allow_pickle=True).item()
output_path = c.bold_file

# savedir = 'simulations/firedrake'
# os.makedirs(savedir, exist_ok=True)
# plot_sample_curves(data_array=data['phi_e'], time_array=data['t_star'], n_samples_to_plot=data['phi_e'].shape[1], total_samples=data['phi_e'].shape[1], sn=sn, savefig=f'{savedir}/phi_XN={XN}_TN={TN}.png')

x0 = np.array([0,1,1,1,0])

print(data['t_star'].shape)
print(data['phi_e'].shape)
BalloonODE = Balloon(data['t_star'], data=data['phi_e'])
n_parcellations: int = data['phi_e'].shape[1]
odes = BalloonODE.ode_system
# Solvers 
ode_solver = EulerSolver(odes)
solution = ode_solver.solve(x0, sn, T, n_parcellations)
y = solution[:, -1, :]
del BalloonODE, n_parcellations, odes, ode_solver

print(data_pinn['t_star'].shape)
print(data_pinn['u_pred'].shape)
BalloonODE = Balloon(data_pinn['t_star'], data=data_pinn['u_pred'])
n_parcellations: int = data_pinn['u_pred'].shape[1]
odes = BalloonODE.ode_system
# Solvers 
ode_solver = EulerSolver(odes)
solution_pred = ode_solver.solve(x0, sn, T, n_parcellations)
y_pred = solution_pred[:, -1, :]

print(y.shape)
savedir = 'simulations/bold'
os.makedirs(savedir, exist_ok=True)
plot_sample_curves(data_array=y[:data['t_star'].shape[0], :], time_array=data['t_star'], n_samples_to_plot=data['phi_e'].shape[1], total_samples=data['phi_e'].shape[1], sn=sn, savefig=f'{savedir}/bold_{pinn_file}.png')
plot_sample_curves(data_array=y_pred[:data_pinn['t_star'].shape[0], :], time_array=data_pinn['t_star'], n_samples_to_plot=data_pinn['u_pred'].shape[1], total_samples=data_pinn['u_pred'].shape[1], sn=sn, savefig=f'{savedir}/bold_PINN_{pinn_file}.png')

### FC:
solution = normalise_bold_data(data=data['phi_e'], filename=None)
# solution = normalise_bold_data(data=y, filename=None)
fc_data = compute_static_FC(solution, kind='correlation', plot_it=True, savename='sim')

del solution
solution = normalise_bold_data(data=data_pinn['u_pred'], filename=None)
# solution = normalise_bold_data(data=y_pred, filename=None)
fc_pinn = compute_static_FC(solution, kind='correlation', plot_it=True, savename='pinn')

fc_error = np.abs(fc_data-fc_pinn)

vmin = 0 # min(fc_data.min(), fc_pinn.min())
vmax = 1 #max(fc_data.max(), fc_pinn.max())

fig, axes = plt.subplots(1, 3, figsize=(10, 5), dpi=100)

im1 = axes[0].imshow(fc_data, vmin=vmin, vmax=vmax, cmap='coolwarm')
axes[0].set_title('FEM')
axes[0].axis('off')
axes[0].set_xticks([])
axes[0].set_yticks([])

fig.colorbar(im1, ax=axes[0], orientation='horizontal', fraction=0.05)

im2 = axes[1].imshow(fc_pinn, vmin=vmin, vmax=vmax, cmap='coolwarm')
axes[1].set_title('PINN')
axes[1].axis('off')
axes[1].set_xticks([])
axes[1].set_yticks([])

fig.colorbar(im1, ax=axes[1], orientation='horizontal', fraction=0.05)

im3 = axes[2].imshow(fc_error, cmap='coolwarm')
axes[2].set_title('Error')
axes[2].axis('off')
axes[2].set_xticks([])
axes[2].set_yticks([])
print(np.mean(fc_error))

fig.colorbar(im3, ax=axes[2], orientation='horizontal', fraction=0.05)

plt.tight_layout()
plt.savefig('fc_pinn_minus_fem.png')
# plt.figure(figsize=(10, 4), dpi=100)
# plt.plot(data['t_star'], data['phi_e'][:, 0])
# plt.plot(data_pinn['t_star'], data_pinn['u_pred'][:, 0])
# plt.savefig('fc_pinn_minus_fem.png')


