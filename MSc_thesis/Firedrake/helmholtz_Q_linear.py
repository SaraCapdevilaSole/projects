import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.config.params import noise, T, sn, dt, size, XN, TN, max_size, _extra_str, shape, config_kwargs
from src.utils.utils import get_config
from src.utils.utils_helmholtz_linear import (
    ic_conditions, 
    init_params, 
    gen_funcs,
    gen_mesh, 
    residual_equ,
    solve_helmholtz,
    calculate_errors,
    plot_errors,
    plot_fd
)

def run_helmholtz(c, max_size, shape='square'):
    """Runs the Helmholtz simulation based on the given configuration."""
    mesh, V, mesh_ref, mesh_ref_c = gen_mesh(c, max_size, shape)
    u_nm1, u_n, u_np1, v, u, Qs = gen_funcs(V)
    gamma_s, r_s = init_params(c)
    u_nm1, u_n, u_np1, u = ic_conditions(c, mesh, u_nm1, u_n, u_np1, u, V)
    F_residual = residual_equ(u_nm1, u_n, u_np1, u, v, c, gamma_s, r_s, Qs)
    u_last_t = solve_helmholtz(c, u_nm1, u_n, u_np1, u, V, mesh, mesh_ref, F_residual, shape, Qs, mesh_ref_c)
    if shape=='square':
        plot_fd(u_np1, c)
    return u_last_t

if __name__ == "__main__":
    sizes = [size] 
    dt = [dt]
    reference_solution = None
    errors = []

    for size in tqdm(sizes):
        for dti in tqdm(dt):
            config = get_config(T, dti, sn, noise, size, XN, TN, _extra_str, shape, **config_kwargs)

            u_projected = run_helmholtz(config, max_size=max_size, shape=shape)
            if reference_solution is None:
                reference_solution = u_projected.copy()
            else:
                calculate_errors(reference_solution, u_projected, size, dti, errors)
    plot_errors(errors, sizes, dt, config._dir)


