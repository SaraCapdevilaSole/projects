import sys
sys.path.append('..')
from src.config.config_fc import FC_to_plot, parcellation_path, KIND
from src.utils.utils import detrend_data, bandpass_data

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure
import logging
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO)

def normalise_bold_data(data: np.array, filename: str, lowcut=0.01, highcut=0.1, tr=0.01, k=2) -> np.array:
    mapped_data = map_voxel_to_parcel(filename=filename, data=data) # lowcut=1/60, highcut=0.15, tr=0.72, k=2
    detrended_data = detrend_data(mapped_data)
    bandpassed_data = np.apply_along_axis(bandpass_data, 0, detrended_data, lowcut, highcut, tr, k)
    # bandpassed_data = bandpass_data(detrended_data, lowcut, highcut, tr, k)
    z_scored_data = (bandpassed_data - np.mean(bandpassed_data, axis=0)) / np.std(bandpassed_data, axis=0)
    return z_scored_data

def map_voxel_to_parcel(data: np.array, filename: str) -> np.array:
    """map: (p, 5124), data: (T, 5124) -> new_data: (T, p)"""
    atlas_map = np.load(filename)
    new_data = data @ atlas_map.T
    return new_data

def _connectivity_measure(kind):
    try:
        measure = ConnectivityMeasure(kind=kind)
    except ValueError as e:
        logging.error(e)
    return measure

def compute_static_FC(X, kind, measure=None, plot_it=False, title="Static"):
    if not measure:
        measure = _connectivity_measure(kind=kind)
    FC_matrix = measure.fit_transform([X])[0]
    np.fill_diagonal(FC_matrix, 0)

    if plot_it:
        plot_fc_matrix(FC_matrix, title=title)

    return FC_matrix

def compute_FC_matrix(X, window_size=30, increment=2, kind='correlation', plot_it=True):
    num_time_steps, _ = X.shape
    measure = _connectivity_measure(kind=kind)
    
    logging.info(f'Computing {FC_to_plot} FC matrices with: TR={window_size}, increment={increment}, kind: {kind}')
    
    FC_matrices = []
    for i in tqdm(range(0, num_time_steps - window_size + 1, increment)):
        window_data = X[i:i+window_size, :]
        FC_matrix = compute_static_FC(X=window_data, measure=measure, kind=kind)
        FC_matrices.append(FC_matrix)
    
    if plot_it:
        plot_fc_matrix(FC_matrices, multiple=True)
    
    return FC_matrices

def compute_FC_average(FC_matrices, plot_it=True):
    average_FC_matrix = np.mean(FC_matrices, axis=0)

    if plot_it:
        plot_fc_matrix(average_FC_matrix)
    
    return average_FC_matrix

def compute_cdist_matrix(X, metric='minkowski', plot_it=True):
    # statistical connections in the brain
    cdist_metric = cdist(X.T, X.T, metric=metric)
    if plot_it:
        plot_fc_matrix(cdist_metric, title=metric + ' similarity')
    return cdist_metric

def plot_fc_matrix(fc_matrix, cmap='coolwarm', interpolation='nearest', multiple=False, title='Average'):
    if not multiple:
        _plot_matrix(fc_matrix, cmap=cmap, interpolation=interpolation, title=title)
    else:
        _plot_multiple_matrix(fc_matrix, cmap=cmap, interpolation=interpolation)

def _plot_matrix(fc_matrix, cmap, interpolation, title):
    plt.figure(figsize=(8,8), dpi=100)
    plt.imshow(fc_matrix, cmap=cmap, interpolation=interpolation)
    cbar = plt.colorbar(shrink=0.8)  
    cbar.set_ticks([-1, 0, 1])
    cbar.set_label(KIND, fontsize=12)
    plt.clim(-1,1)#np.min(fc_matrix),np.max(fc_matrix))
    # plt.title(f'{title} FC Matrix')
    # plt.xlabel('Brain Region')
    # plt.ylabel('Brain Region')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def _plot_multiple_matrix(fc_matrices, cmap, interpolation):
    num_windows=len(fc_matrices)
    fig, axs = _find_square_axis(num_windows)

    for i, ax in enumerate(axs.flat):
        if i < num_windows:
            _suplot_matrix(fc_matrices[i], cmap, interpolation, axis=(fig, ax), i=i)
        else:
            ax.axis('off') 
    plt.tight_layout()
    plt.show()

def _suplot_matrix(fc_matrix, cmap, interpolation, axis, i):
    im = axis[1].imshow(fc_matrix, cmap=cmap, interpolation=interpolation)
    axis[1].set_title(f'Window {i+1} FC Matrix')
    axis[1].set_xlabel('Brain Region')
    axis[1].set_ylabel('Brain Region')
    plt.colorbar(im, ax=axis[1])
    im.set_clim(vmin=-1, vmax=1)

def _find_square_axis(num_windows, dim=12):
    num_rows = int(np.ceil(np.sqrt(num_windows)))
    num_cols = int(np.ceil(num_windows / num_rows))
    return plt.subplots(num_rows, num_cols, figsize=(dim, dim))


""" To create the figure in the bg report:
tr = 6000*2
FC_matrices = [compute_static_FC(solution[tr*i:tr*(i+1),:], kind=KIND, plot_it=plot_static) for i in range(0,5)]

num_matrices = 5
matrix_size = 20

# Create a figure and set of subplots
fig, axs = plt.subplots(1, num_matrices, figsize=(num_matrices * 2, 4), dpi=100)

# Plot each matrix
for i, ax in enumerate(axs):
    
    im = ax.imshow(FC_matrices[2*i], cmap='coolwarm', interpolation='nearest')
    plt.colorbar(im, shrink=0.3)
    ax.axis('off') 

plt.figtext(0.5, 0.15, "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->", ha="center", fontsize=10, bbox={"alpha":0.0})
sep = 0.0968
plt.figtext(sep*0, 0.2, 
"0s", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(sep*0, 0.15, 
"|", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(sep*1, 0.21, 
r"$\}$", ha="center", fontsize=110, bbox={"alpha":0.0}, rotation=90, fontstretch='expanded', alpha=0.1)

plt.figtext(sep*2, 0.2, 
"120s", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(sep*2, 0.15, 
"|", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(sep*3, 0.21, 
r"$\}$", ha="center", fontsize=110, bbox={"alpha":0.0}, rotation=90, fontstretch='expanded', alpha=0.1)


plt.figtext(sep*4, 0.2, 
"240s", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(sep*4, 0.15, 
"|", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(sep*5, 0.21, 
r"$\}$", ha="center", fontsize=110, bbox={"alpha":0.0}, rotation=90, fontstretch='expanded', alpha=0.1)


plt.figtext(sep*6, 0.2, 
"360s", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(sep*6, 0.15, 
"|", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(sep*7, 0.21, 
r"$\}$", ha="center", fontsize=110, bbox={"alpha":0.0}, rotation=90, fontstretch='expanded', alpha=0.1)


plt.figtext(sep*8, 0.2, 
"480s", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(sep*8, 0.15, 
"|", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(sep*9, 0.21, 
r"$\}$", ha="center", fontsize=110, bbox={"alpha":0.0}, rotation=90, fontstretch='expanded', alpha=0.1)


plt.figtext(0.98, 0.2, 
"600s", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(0.98, 0.15, 
"|", ha="center", fontsize=10, bbox={"alpha":0.0})

plt.figtext(1.03, 0.19, 
"time (s)", ha="center", fontsize=10, bbox={"alpha":0.0})


plt.tight_layout()
plt.show()
"""
# and without arrows:
"""
um_matrices = 5
matrix_size = 20

# Create a figure and set of subplots
fig, axs = plt.subplots(1, num_matrices, figsize=(num_matrices * 2, 4), dpi=100)

# Plot each matrix
for i, ax in enumerate(axs):
    
    im = ax.imshow(FC_matrices[i], cmap='coolwarm', interpolation='nearest')
    plt.colorbar(im, shrink=0.3)
    ax.axis('off') 

y = 0.25

plt.figtext(0.5, y, "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------->", ha="center", fontsize=10, bbox={"alpha":0.0})

plt.figtext(0.00, y + 0.05, 
"0s", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(0.00, y, 
"|", ha="center", fontsize=10, bbox={"alpha":0.0})

plt.figtext(0.18, y + 0.05, 
"120s", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(0.18, y, 
"|", ha="center", fontsize=10, bbox={"alpha":0.0})

plt.figtext(0.38, y + 0.05, 
"240s", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(0.38, y, 
"|", ha="center", fontsize=10, bbox={"alpha":0.0})

plt.figtext(0.58, y + 0.05, 
"360s", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(0.58, y, 
"|", ha="center", fontsize=10, bbox={"alpha":0.0})

plt.figtext(0.78, y + 0.05, 
"480s", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(0.78, y, 
"|", ha="center", fontsize=10, bbox={"alpha":0.0})

plt.figtext(0.96, y + 0.05, 
"600s", ha="center", fontsize=10, bbox={"alpha":0.0})
plt.figtext(0.96, y, 
"|", ha="center", fontsize=10, bbox={"alpha":0.0})

plt.figtext(1.02, y + 0.04, 
"Time (s)", ha="center", fontsize=10, bbox={"alpha":0.0})

plt.tight_layout()
plt.show()
"""