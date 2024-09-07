from src.config.config_balloon import T, dt_files
from src.utils.utils_fc import normalise_bold_data
from src.config.config_fc import FC_MAP_PATH
import balloon

import numpy as np
from scipy.interpolate import interp1d

def calculate_downsampling_error(original_data, downsampled_data, resolution):
    original_indices = np.arange(len(original_data))
    downsampled_indices = np.linspace(0, len(original_data) - 1, len(downsampled_data))
    interpolation_function = interp1d(downsampled_indices, downsampled_data, kind='cubic')
    interpolated_downsampled_data = interpolation_function(original_indices[::resolution])

    rmse = np.sqrt(np.mean((original_data[::resolution] - interpolated_downsampled_data) ** 2))
    mae = np.mean(np.abs(original_data[::resolution] - interpolated_downsampled_data))
    max_absolute_error = np.max(np.abs(original_data[::resolution] - interpolated_downsampled_data))
    relative = np.linalg.norm(interpolated_downsampled_data - original_data[::resolution]) / np.linalg.norm(original_data[::resolution])
    error_metrics = {
        "relative": relative,
        "RMSE": rmse,
        "MAE": mae,
        "Max Absolute Error": max_absolute_error
    }

    return error_metrics

def downsample_data(t_star, u_ref, coords, tr, parcellations_to_use=5125, dt=dt_files, t_avoid=0, T=T):
    files = lambda t, dt=dt: int(t/dt) # temporal resolution of files [a.u.]
    t_star = t_star[files(t_avoid):files(T):files(tr)]
    u_ref = u_ref[files(t_avoid):files(T):files(tr), :parcellations_to_use]
    coords = coords[:parcellations_to_use, :]
    return t_star, u_ref, coords

def load_data(data_path, allow_pickle=True):
    data = np.load(data_path, allow_pickle=allow_pickle).item()
    u_ref = data['phi_e']
    t_star = data['t_star']
    coords = data["mesh_coordinates"]
    return u_ref, t_star, coords

def balloon_downsample_normalise(t_star, u_ref, coords, tr,  t_avoid, T):
    t_star_d, u_ref_d, _ = downsample_data(t_star, u_ref, coords, tr=tr, t_avoid=0, T=T)
    y_d = balloon.main(write=False, white_noise_run=False, plot=False, data=u_ref_d, run=True, t_array=t_star_d, T=T, dt_files=tr)
    return normalise_bold_data(data=y_d[int(t_avoid/tr):, :], filename=FC_MAP_PATH), t_star_d[int(t_avoid/tr):]

