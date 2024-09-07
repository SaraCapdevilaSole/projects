import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d
from src.utils.one_over_f_functions import compute_spectrum, estimate_aperiodic_exponent, make_plots
from src.config.waves_config import with_amplitude, bands_range, bands_amplitude, bands_mean, frequencies_pink, sources_3d, speed_sources_3d, sources_2d, speed_sources_2d

savepath = 'simulations/spatial_noise_lfp'

def find_bands(bands_names, bands_mean=bands_mean):
    return {name: bands_mean[name] for name in bands_names if name in bands_mean}

def generate_composite_signal(t, noise, bands_names=["delta", "theta", "alpha"], with_amplitude=with_amplitude):
    """Generates a composite signal given time, frequency bands, and noise level."""
    selected_bands = find_bands(bands_names)
    signal = np.zeros_like(t)
    N = len(selected_bands)
    for name, freq in selected_bands.items():
        if with_amplitude:
            a0 = bands_amplitude[name] * 1e-6
        else:
            a0 = noise / N
        signal += a0 * np.sin(2 * np.pi * freq * t)
    return signal

def generate_spatial_composite_signal(t, x, y, noise, bands_names=["delta", "theta", "alpha", "beta", "gamma", "gamma high", "gamma very high"], with_amplitude=with_amplitude):
    """Generates a composite signal given time, frequency bands, and noise level."""
    selected_bands = find_bands(bands_names)
    signal = np.zeros_like(t)
    N = len(selected_bands)
    for name, freq in selected_bands.items():
        if with_amplitude:
            a0 = bands_amplitude[name] * 1e-6
        else:
            a0 = noise / N
        v_x, v_y = 0.001, 0.002 # v [m/s] := gamma [/s] * r [m]
        rs = 10
        signal += a0 * np.sin(2 * np.pi * freq * (t*v_x - x)/rs) * np.sin(2 * np.pi * freq * (t*v_y - y)/rs) 
    return signal 

# def phase(x, y, x0, y0, vs_x, vs_y, t, v=0.1):
#     x0_t, y0_t = x0 + vs_x * t, y0 + vs_y * t # position of source varies in time
#     # v [m/s] := gamma [/s] * r [m] (10*0.001=0.01) # made larger -> otherwise need larger grid
#     return np.sqrt((x-x0_t)**2 + (y-y0_t)**2) / v

def phase(coords: np.array, coords0: np.array, coords_v: np.array, t: float, radius: float, v: float=0.1): # v [m/s] := gamma [/s] * r [m] (10*0.001=0.01) 
    """compute the position of the source as it varies in time."""
    mv = 10 if radius == 100 else 10  # controls the synchorny between the different waves of the noise (much larger therefore similar phases -> mostly synchrony)
    # TODO: plot of mv against synchrony ? 
    v = v if len(coords)==2 else v*mv # make larger for sphere
    coords0_t = [x0 + vs_x * t for x0, vs_x in zip(coords0, coords_v)]
    norm0_t = np.linalg.norm(coords0_t)
    norm0 = np.linalg.norm(coords0)
    if radius==100:
        coords0_t = [x0_t / norm0_t * norm0 for x0_t in coords0_t]
    return np.sqrt(sum((x - x0_t)**2 for x, x0_t in zip(coords, coords0_t))) / v

def find_sources(coords, radius):
    if len(coords) == 3:
        mr = 100 if radius==100 else 1
        ms = 10 if radius==100 else 1
        sources = sources_3d * mr
        speed_sources = speed_sources_3d * ms
    elif len(coords) == 2:
        sources = sources_2d
        speed_sources = speed_sources_2d
    else:
        raise ValueError("Coordinates must be either 2D or 3D.")
    return sources, speed_sources

def generate_one_over_f_Q(t, coords, noise=0.3, radius=1, alpha=1):
    signal = np.zeros_like(t)
    sources, speed_sources = find_sources(coords, radius)
    a0 = noise/(len(frequencies_pink)*len(sources))
    for f in frequencies_pink:
        for i in range(len(sources)):
            x0 = sources[i]
            vs = speed_sources[i]
            # How far away (x,y) is from the source introduces a phase shift
            phase_xy = phase(coords, x0, vs, t, radius) 
            signal += 1/(f**alpha) * np.sin(2 * np.pi * f * (t + phase_xy)) 
    signal *= a0
    return signal

def check_time_correlation(signal, max_lag=100):
    autocorr = np.correlate(signal, signal, mode='full')
    corr_plot(autocorr, max_lag=max_lag, type_signal='Temporal', savename='timecorr.png')

def check_spatial_correlation(signal, max_lag=100):
    spatial_autocorr = np.correlate(signal.flatten(), signal.flatten(), mode='full')
    corr_plot(spatial_autocorr, max_lag=max_lag)

def check_variance_and_mean(signal, num_segments=10):
    mean_value = np.mean(signal)
    print(f"Mean of the signal: {mean_value}")
    segment_length = len(signal) // num_segments
    variances = [np.var(signal[i*segment_length:(i+1)*segment_length]) for i in range(num_segments)]
    print(f"Variances of the segments: {variances}")
    print(f"Variance of the variances: {np.var(variances)}")
    plt.figure()
    plt.plot(variances, 'o-')
    plt.xlabel('Segment')
    plt.ylabel('Variance')
    plt.savefig(f'{savepath}/variance.png')

def corr_plot(correlation, max_lag=100, type_signal='Spatial', savename='spacecorr.png'):
    correlation = correlation[correlation.size // 2:]
    correlation /= np.max(correlation)
    lags = np.arange(0, len(correlation))
    plt.figure()
    plt.stem(lags[:max_lag], correlation[:max_lag], use_line_collection=True)
    plt.xlabel('Lag')
    plt.ylabel(f'{type_signal} correlation')
    plt.savefig(os.path.join(savepath, savename))

def interpolate_signal(t_low, signal_low, t_high, ravel=False):
    """Interpolates a low-sampling-rate signal to match a high-sampling-rate time array."""
    if ravel:
        t_low = np.ravel(t_low)
        signal_low = np.ravel(signal_low)
    interpolator = interp1d(t_low, signal_low, kind='linear')
    signal_low_interp = interpolator(t_high)
    return signal_low_interp

def calculate_mse(signal_high, signal_low_interp):
    """Calculates the Mean Squared Error between two signals."""
    mse_error = np.mean((signal_high - signal_low_interp) ** 2)
    return mse_error

def plot_signals(t_high, signal_high, signal_low_interp, sr_high=None, sr_low=None, num_high=None, num_low=None):
    """Plots the high-sampling-rate signal and interpolated low-sampling-rate signal for comparison."""
    label_1 = f'{sr_high} Hz' if sr_high is not None else f'dim = {num_high}'
    label_2 = f'{sr_low} Hz' if sr_high is not None else f'dim = {num_low}'
    fig = plt.figure()
    plt.plot(t_high, signal_high, label=f'Signal ({label_1})')
    plt.plot(t_high, signal_low_interp, label=f'Interpolated Signal ({label_2})', linestyle='--', color="red")
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Comparison of Signals')
    fig.savefig(f'{savepath}/comparison_signals.png')
    plt.show()

def save_voxel_plot(t_high, Z, i, filename='trial_voxel.png'):
    """Saves a static plot of the signal at a specific voxel (for 2D or 3D)."""
    fig, ax = plt.subplots()
    for ii in i:
        if len(Z.shape) == 3:
            plt.plot(t_high, Z[ii, ii, :])
        elif len(Z.shape) == 2:
            plt.plot(t_high, Z[ii, :])
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
    fig.savefig(os.path.join(savepath, filename))
    plt.close(fig)

def plot_multiple_band_signals(t, bands, noise=0.3):
    """Plots signals for different sets of frequency bands in separate subplots."""
    num_bands = len(bands)
    fig, axs = plt.subplots(num_bands, 1, figsize=(15, 1.5*num_bands), sharex=True, dpi=150)
    for i, band_names in enumerate(bands):
        freqs = find_bands(bands_names=band_names, bands_mean=bands_range)
        signal = generate_composite_signal(t, noise=noise, bands_names=band_names)
        if len(band_names) > 1: 
            freqs_range = [freqs[band_names[0]][0], freqs[band_names[-1]][-1]]
            label = f"{band_names[0]} - {band_names[-1]}"
        else:
            freqs_range = freqs[band_names[0]]
            label = f"{band_names[0]}"
        label += f"\n({freqs_range[0]}-{freqs_range[1]} Hz)"
        
        axs[i].plot(t, signal, color="black")
        axs[i].annotate(f'{label}', xy=(1.05,0), xytext=(0, 0), textcoords='offset points', ha='left', va='center', fontsize=20, color='black')
        # axs[i].set_title(f'{label}')
        if i == 0:
            y_lim = [min(signal),max(signal)]
        axs[i].set_ylim(y_lim)
        axs[i].grid(False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        for loc in ['top', 'right', 'left', 'bottom']:
            axs[i].spines[loc].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(f'{savepath}/multiple_bands.png')
    plt.show()

def compare_res(noise=0.3, grid_size=10, radius=1):
    sampling_rate_high = 400  # Hz
    sampling_rate_low = 300  # Hz
    duration = 10  # seconds
    t_high = np.linspace(0, duration, int(sampling_rate_high * duration)) 
    t_low = np.linspace(0, duration, int(sampling_rate_low * duration)) 

    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)

    Z_high = np.zeros((grid_size, grid_size, len(t_high)))
    Z_low = np.zeros((grid_size, grid_size, len(t_low)))
    Z_low_interp = np.zeros_like(Z_high)

    for i in range(grid_size):
        for j in range(grid_size):
            Z_high[i, j, :] = generate_one_over_f_Q(t_high, (x[i], y[j]), noise, radius=radius)
            Z_low[i, j, :] = generate_one_over_f_Q(t_low, (x[i], y[j]), noise, radius=radius)
            Z_low_interp[i, j, :] = interpolate_signal(t_low, Z_low[i, j, :], t_high)
    
    mse_error = np.mean([calculate_mse(Z_high[i, j, :], Z_low_interp[i, j, :]) for i in range(grid_size) for j in range(grid_size)])
    print(f"Mean Squared Error between high and low sampling rate signals: {mse_error}")

    plot_signals(t_high, Z_high[0, 0, :], Z_low_interp[0, 0, :], sampling_rate_high, sampling_rate_low)

def generate_animated_wave_spatial(grid_size, t_high, Z, filename=f'{savepath}/wave_animation.gif'):
    """Generates an animated wave signal over a mesh grid and saves it as a GIF."""
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(X, Y, Z[:, :, 0], shading='auto', cmap='coolwarm')
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Noise')

    ax.set_xlabel('X') 
    ax.set_ylabel('Y')

    def update(frame):
        mesh.set_array(Z[:, :, frame].flatten())
        return mesh,

    ani = animation.FuncAnimation(fig, update, frames=len(t_high), interval=20, blit=True)
    Writer = animation.PillowWriter(fps=30)
    ani.save(filename, writer=Writer)
    plt.close(fig)

def generate_animated_wave_spatial_3d(num_points, t_high, Z, filename=f'{savepath}/wave_animation_spatial.gif'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    points = generate_spherical_mesh(num_points)
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=Z[:, 0], cmap='coolwarm')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Noise')
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()
    ax.set_aspect('equal')
    
    def update(frame):
        scatter.set_array(Z[:, frame])
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=len(t_high), interval=20, blit=True)
    Writer = animation.PillowWriter(fps=30)
    ani.save(filename, writer=Writer)
    plt.close(fig)

def spatial_source(noise, voxels=[2,3], animate=False, radius=1): #, bands_names=["delta", "theta", "alpha", "beta", "gamma", "gamma high", "gamma very high"]):
    sampling_rate_high = 200  # Hz
    duration = 3  # seconds
    t_high = np.linspace(0, duration, int(sampling_rate_high * duration)) 
    grid_size = 150
    Z = np.zeros((grid_size, grid_size, len(t_high)))
    x = np.linspace(0,1,grid_size)
    y = np.linspace(0,1,grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            Z[i, j, :] = generate_one_over_f_Q(t_high, (x[i], y[j]), noise, radius=radius)
            # generate_spatial_composite_signal(t_high, (x[i], y[j]), noise, bands_names=bands_names)
    if animate:
        generate_animated_wave_spatial(grid_size, t_high, Z, filename=f'{savepath}/wave_animation_spatial.gif')
    save_voxel_plot(t_high, Z, voxels, filename='trial_voxel.png')
    return Z, t_high

def compute_mean_spectrum(Z, t_star):
    """Compute the mean frequency spectrum over all (x, y)."""
    num_x, num_y, num_time = Z.shape
    all_spectra = []
    
    for n_x in range(num_x):
        for n_y in range(num_y):
            [S, f] = compute_spectrum(Z[n_x, n_y, :], t_star)
            all_spectra.append(S)
    
    mean_spectrum = np.mean(all_spectra, axis=0)
    return mean_spectrum, f

def analyse_and_plot(Z, t_star):
    """Estimate the aperiodic exponent of the mean spectrum and plot the results."""
    mean_spectrum, f = compute_mean_spectrum(Z, t_star)
    [aperiodic_exponent, x_linear_fit, y_linear_fit] = estimate_aperiodic_exponent(mean_spectrum, f, [8,20])
    
    fig, (ax, ax_) = plt.subplots(2, 1, figsize=(10, 8))
    make_plots(t_star, Z, f, mean_spectrum, x_linear_fit, y_linear_fit, aperiodic_exponent, ax, ax_)
    savedir = 'simulations/spectral_density'
    fig.savefig(f'{savedir}/spectral_pink_noise.png')

def generate_spherical_mesh(num_points, r=1):
    """
    Generate uniformly distributed points on the surface of a sphere with radius r.
    """
    theta = 2 * np.pi * np.random.rand(num_points)  
    phi = np.arccos(1 - 2 * np.random.rand(num_points))  
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    return np.vstack((x, y, z)).T

def spherical_spatial_source(noise, voxels=[0,10,-1], num_points=1000, animate=False, radius=1):
    """
    Generate 3D spatial source signals on the surface of a sphere.
    """
    sampling_rate_high = 100  # Hz
    duration = 10  # seconds
    t_high = np.linspace(0, duration, int(sampling_rate_high * duration))
    
    points = generate_spherical_mesh(num_points, r=radius)
    
    Z = np.zeros((num_points, len(t_high)))
    
    for i, (x, y, z) in enumerate(points):
        Z[i, :] = generate_one_over_f_Q(t_high, (x, y, z), noise, radius=radius)
    
    if animate:
        generate_animated_wave_spatial_3d(num_points, t_high, Z, filename=f'{savepath}/wave_animation_spatial_3d_r={radius}.gif')
    
    save_voxel_plot(t_high, Z, voxels, filename=f'trial_voxel_3d_r={radius}.png')
    
    return Z, t_high, points, num_points

def interpolate_Z(Z, t_h, points_high, num_high, Z_lower, t, points_lower, num_low):
    Z_interpolated = np.zeros_like(Z)

    for j in range(len(t)):
        Z_interpolated[:, j] = interpolate_signal(points_lower, Z_lower[:, j], points_high, ravel=True)

    plot_signals(t, Z_interpolated[10, :], Z_interpolated[10, :], num_high=num_high, num_low=num_low)
    return Z_interpolated