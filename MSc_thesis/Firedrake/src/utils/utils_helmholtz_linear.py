import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from firedrake.pyplot import tricontourf
from scipy.io import loadmat
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from src.config.params import use_noise, ic_is_0, save_derivatives, noise_type, bands_names, config_kwargs, noise_args
from src.config.waves_config import bands_mean, bands_range, bands_amplitude, sources_3d, speed_sources_3d, sources_2d, speed_sources_2d, frequencies_pink
from src.utils.utils_waves import find_bands

def subset_coords(mesh, prop):
    coordinates = mesh.coordinates
    num_to_keep = [int(coordinates.dat.data.shape[0] * prop_i) for prop_i in prop]
    subset_coordinates = coordinates.dat.data[num_to_keep[0]:num_to_keep[1], :]
    return subset_coordinates

def find_mesh_patch(mesh, prop, factor):
    subset = subset_coords(mesh, prop)
    if factor is not None:
        interp_subset = interpolate_coords(subset, factor=factor)
        return interp_subset
    else:
        return subset

def find_n_curv_patch(n_path, curv_path, mask):
    curv = np.loadtxt(curv_path, delimiter=',')
    normals = np.loadtxt(n_path, delimiter=',')
    curv = np.array(curv) # (p,)
    normals = np.array(normals) # (p, 3)
    return normals[mask], curv[mask]

def real_mesh_patch(mesh, subject='jm', return_data=False, n_curv_path=None):
    """extract patch used in real data from mesh"""
    ecog_coords, ecog_data = load_ecog_coords(subject, return_data)
    coordinates = np.array(mesh.coordinates.dat.data)

    kdtree = KDTree(coordinates)
    distances, indices = kdtree.query(ecog_coords)
    coords_masked = coordinates[indices]

    if n_curv_path is not None:
        ecog_min = np.min(ecog_coords, axis=0) # just bc this was done like this before
        ecog_max = np.max(ecog_coords, axis=0) 
    else:
        ecog_min = np.min(coords_masked, axis=0) # for real data 
        ecog_max = np.max(coords_masked, axis=0) 
    
    mask = np.all((coordinates >= ecog_min) & (coordinates <= ecog_max), axis=1)
    region_of_interest = coordinates[mask]

    return_list = [region_of_interest] # f.d runs on region of interest -> more data points

    if return_data:
        return_list += [coords_masked, ecog_data, ecog_coords]
    
    if n_curv_path is not None:
        n_curv_patch = find_n_curv_patch(*n_curv_path, mask)
        return_list += n_curv_patch

    return return_list 

def load_ecog_coords(subject='jm', return_data=False):
    real_data_path = '/vol/bitbucket/sc3719/JAXPI/jaxpi/examples/helmholtz_3d_inverse/real_data/fixation_PAC/data'
    file_path = os.path.join(real_data_path, subject, f'{subject}_base.mat')
    mat_data = loadmat(file_path)
    coords = mat_data['locs']
    data = mat_data['data'] if return_data else None
    return coords, data

def gen_mesh(c, max_size, shape):
    mesh_ref_coords_sub = None
    if shape == 'square':
        mesh = UnitSquareMesh(c.size, c.size)  
        mesh_ref = UnitSquareMesh(max_size, max_size)
    elif shape == 'sphere':
        mesh = IcosahedralSphereMesh(radius=c.radius, refinement_level=c.size)
        mesh_ref = IcosahedralSphereMesh(radius=c.radius, refinement_level=max_size)
    elif shape == 'real_mesh':
        try:
            mesh = Mesh(c.path_msh, dim=3)
            mesh_ref = Mesh(c.path_func(max_size), dim=3)
            if isinstance(c.prop_to_keep, str):
                mesh_ref_coords_sub = real_mesh_patch(mesh_ref, subject=c.prop_to_keep)
            elif isinstance(c.prop_to_keep, list):
                if c.prop_to_keep[0] < 1:
                    mesh_ref_coords_sub = find_mesh_patch(mesh_ref, c.prop_to_keep, None)
        except FileNotFoundError as e:
            print("File not found: ", e)
    else:
        raise NotImplementedError
    V = FunctionSpace(mesh, "CG", 1) 
    return mesh, V, mesh_ref, mesh_ref_coords_sub

def gen_funcs(V):
    u_nm1 = Function(V, name="VelocityPrev")
    u_n = Function(V, name="Velocity")
    u_np1 = Function(V, name="VelocityNext")
    v = TestFunction(V)
    u = TrialFunction(V)
    if use_noise:
        Qs = Function(V, name="Q_s")
    else:
        Qs = None
    return u_nm1, u_n, u_np1, v, u, Qs

def init_params(c):
    gamma_s = Constant(c.gamma) 
    r_s = Constant(c.r) 
    return gamma_s, r_s

def load_noise(c, V):
    mesh_size = V.dim()
    x_old = np.linspace(0, 1, c.XN)
    t_old = np.linspace(0, 1, c.TN) 
    noise_matrix = np.load(f'data/input_data/Qs_3d_tx/Qs_X={c.XN}_T={c.TN}_n={c.noise}.npy')
    
    if mesh_size < c.XN:
        raise ValueError(f"Mesh size too small, must be >= {((c.XN)**0.5-1)}") # for square mesh
    x_new = np.linspace(0, 1, mesh_size)
    t_new = np.linspace(0, 1, int(np.ceil(c.T/c.dt)))

    interpolator_x = interp1d(x_old, noise_matrix, kind='quadratic', axis=1) # spatial interpolator
    noise_matrix_int_x = interpolator_x(x_new)
    interpolator_t = interp1d(t_old, noise_matrix_int_x, kind='quadratic', axis=0) # temporal interpolator
    noise_matrix_int_tx = interpolator_t(t_new)

    np.save(f'{c.output_phi_e}/noise.npy', noise_matrix_int_tx)
    return noise_matrix_int_tx

def ic_conditions(c, mesh, u_nm1, u_n, u_np1, u, V):
    if ic_is_0:
        u_nm1.assign(0)
        u_n.assign(0)
        u_np1.assign(0)
    else:
        x = SpatialCoordinate(mesh)
        u_nm1 = space_time_signal(0, x, u_nm1, c.noise, **noise_args)
        u_n = space_time_signal(0, x, u_n, c.noise, **noise_args)
        u_np1 = space_time_signal(0, x, u_np1, c.noise, **noise_args)
    return u_nm1, u_n, u_np1, u

def sinusoidal_noise(x, t, c, p_T=1, p_R=200):
    return c.noise * cos(pi*t/p_T) * sin(pi*x[0]/p_R) * sin(pi*x[1]/p_R) 

def residual_equ(u_nm1, u_n, u_np1, u, v, c, gamma_s, r_s, Qs=None):
    _noise = inner(Qs, v) if Qs is not None else 0
    F_residual = (
        inner((u - 2 * u_n + u_nm1) / (c.dt**2 * gamma_s**2), v)
        + 2 / gamma_s * inner((u - u_n)/c.dt, v)
        + inner(u_n, v)
        + r_s**2 * inner(grad(v), grad(u_n))
        - _noise
     )*dx
    return F_residual

def calculate_errors(reference_solution, u_projected, size, dt, errors):
    """Calculates and stores the Absolute error."""
    error_abs = np.sqrt(np.mean((reference_solution - u_projected)**2))
    errors.append((size, dt, error_abs))
    print(f"Res: {size}, dt: {dt}, Abs error: {error_abs}")

def plot_errors(errors, sizes, dt, output_dir):
    """Plots the L2 errors."""
    if len(errors)>1:
        fig, axes = plt.subplots()
        for size, dti, error_abs in errors:
            print(f"Res: {size}, dt: {dti}, Absolute error: {error_abs}")
            plt.scatter(size, error_abs, s=15, label=f'dt={dti}')
        axes.set_yscale('log')
        axes.set_xlabel('Mesh Size')
        axes.set_ylabel('Absolute Error (log scale)')
        axes.set_xticks(sizes)
        path = os.path.join(output_dir, "errors")
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, f"error_res_dt={dt}_sizes={sizes}.pdf"))

def plot_fd(u, c):
    """plot firedrake solution: u"""
    fig, axes = plt.subplots()
    contours = tricontourf(u, axes=axes, cmap="inferno")
    axes.set_aspect("equal")
    fig.colorbar(contours)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    fig.savefig(c.output_phi_e + '.pdf')

def save_u_step(u, Qs, u_np1, u_n, u_nm1, mesh_ref, step, c, mesh_ref_c, print_u=False):
    if step % c.freq == 0:
        np.save(os.path.join(c.output_phi_e, f"phi_e_{step}.npy"), u)
        if noise_type != "load_from_file" and use_noise == True:
            Qs_values = ref_conversion(Qs, mesh_ref, mesh_ref_c)
            np.save(os.path.join(c.output_phi_e, f"Qs_{step}.npy"), Qs_values)
        if save_derivatives:
            save_u_derivatives(u_nm1, u_n, u_np1, mesh_ref, c, step, mesh_ref_c)
        if print_u:
            print(u)
    return step + 1

def save_u_derivatives(u_nm1, u_n, u_np1, mesh_ref, c, step, mesh_ref_c):
    u_t = (u_np1 - u_n) / c.dt # forward euler
    u_tt = (u_np1 - 2 * u_n + u_nm1) / c.dt**2 # central scheme
    u_t_values = ref_conversion(u_t, mesh_ref, mesh_ref_c)
    u_tt_values = ref_conversion(u_tt, mesh_ref, mesh_ref_c)
    np.save(os.path.join(c.output_phi_e, f"phi_t_e_{step}.npy"), u_t_values)
    np.save(os.path.join(c.output_phi_e, f"phi_tt_e_{step}.npy"), u_tt_values)

# def find_phases(x, y, V, v=0.1):
#     """find phase for each spatial point due to each source"""    
#     phase_function = [
#         phase(x, y, *source, *speed, v)
#         for source, speed in zip(sources, speed_sources)
#     ]
#     return phase_function

def phase(coords, t, coords0, coords_v, v, coords_div, norm=False): # v [m/s] := gamma [/s] * r [m] (10*0.001=0.01) 
    """compute the position of the source as it varies in time."""
    coords0_t = [x0 + vs_x * t for x0, vs_x in zip(coords0, coords_v)]
    # if norm: # was not doing this with r=1
    #     norm0_t = np.linalg.norm(coords0_t)
    #     norm0 = np.linalg.norm(coords0)
    #     coords0_t = [x0_t / norm0_t * norm0 for x0_t in coords0_t]
    phase = sqrt(sum((x / coords_div - x0_t)**2 for x, x0_t in zip(coords, coords0_t))) / v
    # phase = sqrt(sum((x - x0)**2 for x, x0 in zip(coords, coords0))) / v
    return phase 

def find_sources(x, mr_ms):
    mr, ms = mr_ms
    if len(x) == 3:
        sources = sources_3d * mr
        speed_sources = speed_sources_3d * ms
    elif len(x) == 2:
        sources = sources_2d
        speed_sources = speed_sources_2d
    else:
        raise ValueError("Coordinates must be either 2D or 3D.")
    return sources, speed_sources

def space_time_signal(t, x, Qs, noise, frequency_factor, mr_ms, alpha, v, coords_div): 
    # alpha = 1 if config_kwargs['radius']==100 else alpha
    sources, speed_sources = find_sources(x, mr_ms)
    a0 = noise/(len(frequencies_pink)*len(sources))
    Qs.assign(0)
    combined_expression = a0 * sum(
        sum(
            1/(f**alpha) * sin((2 * pi * f * (t + phase(x, t, source, speed, v, coords_div))) / frequency_factor) 
            for source, speed in zip(sources, speed_sources)
        )
        for f in frequencies_pink
    )
    Qs.interpolate(combined_expression)
    return Qs

def ref_conversion(u, mesh_ref, mesh_ref_c):
    mesh_ref_coords = mesh_ref.coordinates.dat.data if mesh_ref_c is None else mesh_ref_c
    return np.array([u([*coords]) for coords in mesh_ref_coords])

def initialise_noise(c, V, noise_type):
    if use_noise:
        if noise_type == "load_from_file":
            return load_noise(c, V)
        elif noise_type == "bands":
            return find_bands(bands_names, bands_mean)
        # elif noise_type == "pink_noise":
        #     return find_phases(x, y, V, v=c.gamma * c.r)
    return None

def update_Qs(t, x, Qs, V, noise_type, c, step, qs_args):
    if use_noise:
        if noise_type == "load_from_file":
            Qs_data = qs_args
            Qs.dat.data[:] = Qs_data[step, :]
        elif noise_type == "bands":
            selected_bands = qs_args
            Qs = space_time_signal(t, x, Qs, V, selected_bands, c.noise)
        elif noise_type == "pink_noise":
            # phases = qs_args
            Qs = space_time_signal(t, x, Qs, c.noise, **noise_args) #, phases)
        elif noise_type == "sinusoidal":
            sinusoidal_data = sinusoidal_noise(x, t, c)  
            Qs.interpolate(sinusoidal_data)  
        else: 
            raise NotImplementedError("The specified noise_type is not implemented")   
    return Qs

def solve_helmholtz(c, u_nm1, u_n, u_np1, u, V, mesh, mesh_ref, F_residual, shape, Qs, mesh_ref_c):
    t = 0.0
    step = 0
    x = SpatialCoordinate(mesh)
    qs_args = initialise_noise(c, V, noise_type)
    a, L = lhs(F_residual), rhs(F_residual)
    problem = LinearVariationalProblem(a, L, u_np1)
    solver = LinearVariationalSolver(problem)
    with tqdm(total=c.T) as pbar:
        while (t <= c.T):
            Qs = update_Qs(t, x, Qs, V, noise_type, c, step, qs_args)
            solver.solve()
            u_nm1.assign(u_n)
            u_n.assign(u_np1)
            t += c.dt
            u_values = ref_conversion(u_np1, mesh_ref, mesh_ref_c)
            step = save_u_step(u_values, Qs, u_np1, u_n, u_nm1, mesh_ref, step, c, mesh_ref_c, print_u=True)
            pbar.update(c.dt)
    return u_values