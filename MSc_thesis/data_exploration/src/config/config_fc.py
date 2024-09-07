from src.config.config_balloon import OUTPUT_PATH, T, dt_files
"""
- FC measures the statistical dependencies or correlations between different brain regions or voxels. 
- It indicates how much activity in one region is associated with activity in another region over time.
- Spatiotemporal correlation

NOTE:
# for every time frame dt -> one file. extract signal from each parcellation
# compute signal correlationn of parcellation with every other parcellation
# find mean over all time
"""

cmap = ['coolwarm', 'jet']

parcellation_path = '../../../ML-NFT/eamon_data/labels_5124.txt'
FC_MAP_PATH = 'data/FC_data_map/fc_atlas.npy'

TR = 6000 # repetition time - ref: paper from Deco and Kringelbach (theirs: 30), A Comparison of Static and Dynamic Functional Connectivities for Identifying Subjects and Biological Sex Using Intrinsic Individual Brain Connectivity (use 60s)
TR_inc = 360 # Increment of TR - ref: paper from Deco and Kringelbach (theirs: 2), other: around 3.6s

FC_to_plot = 4 # number of functional matrices to plot
start_files = 3000 #5000 # plot from file at t=start_files*dt_files # don't go below 3000
end_files = start_files + TR + TR_inc*FC_to_plot - 1

# Functional connectivity kind options
kind_options = [
    'covariance',
    'precision',  # The inverse covariance matrix - partial covariance (covariances between two regions conditioned on all the others)
    'correlation',  # Pearson correlation coefficient
    'partial correlation'  # Accurately detect true brain network connections (mostly around 0)
]
KIND = kind_options[2] 

# assert end_files - start_files >= TR + TR_inc

normalise_solution = True

sliding_window_fc = True
compute_static = True # compute correlation across all T
plot_bold_curves = False  # Plot bold curves

compute_cdist = False  # Compute cross-distance matrices
compute_fc_mean = True  # Compute mean functional connectivity matrix

plot_fcs = True  # Plot individual functional connectivity matrices
plot_mean = True  # Plot mean functional connectivity matrix
plot_cdist = False  # Plot cross-distance matrices
plot_static = True

if normalise_solution:
    total_samples = 101
    plot_samples = 101
else:
    total_samples = 5124
    plot_samples = 1000