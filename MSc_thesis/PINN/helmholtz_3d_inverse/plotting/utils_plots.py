import numpy as np
from nilearn.connectome import ConnectivityMeasure

def _connectivity_measure(kind):
    try:
        measure = ConnectivityMeasure(kind=kind)
    except ValueError as e:
        logging.error(e)
    return measure

def compute_static_FC(X, kind, measure=None):
    if not measure:
        measure = _connectivity_measure(kind=kind)
    FC_matrix = measure.fit_transform([X])[0]
    np.fill_diagonal(FC_matrix, 0)
    return FC_matrix

def compute_FC_matrix(X, window_size=30, increment=2, kind='correlation', plot_it=True):
    num_time_steps, _ = X.shape
    measure = _connectivity_measure(kind=kind)
    
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

def compute_average_spatial_correlation(Q, u):
    num_timepoints = Q.shape[0]
    correlations = np.zeros(num_timepoints)
    
    for t in range(num_timepoints):
        Q_t = Q[t, :]
        u_t = u[t, :]
        
        correlation = np.corrcoef(Q_t, u_t)[0, 1]
        correlations[t] = correlation
    
    # average_correlation = np.mean(correlations)
    
    return correlations