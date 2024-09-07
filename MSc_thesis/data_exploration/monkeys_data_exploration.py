import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data_path = '/Users/saracapdevilasole/Downloads/Qu'

def read_ecog_data(file_identifier):
    if file_identifier == 'info':
        file_path = os.path.join(data_path, 'info.mat')
        mat_data = loadmat(file_path)
    elif file_identifier == 'positions':
        file_path = os.path.join(data_path, 'elPosition_Qu.mat')
        mat_data = loadmat(file_path)
    else:
        file_path = os.path.join(data_path, f'ECoG_ch{int(file_identifier):03d}.mat')
        with h5py.File(file_path, 'r') as mat_file:
            print(list(mat_file.keys()))
            mat_data = mat_file['data'][:]
    return mat_data

el_data = read_ecog_data('positions')
# print(el_data)
# print(el_data.keys() if isinstance(el_data, dict) else el_data.shape)
# print(el_data['elY'])

IMG = el_data['IMG']
IMG = IMG.astype(np.float32)
# 2D coordinates (3D projected onto a 2D plane)
elX = el_data['elX'].flatten()  
elY = el_data['elY'].flatten()  
print(elX.shape)
print(elY.shape)

# plt.figure()
# plt.imshow(IMG, cmap='gray')
# plt.axis('equal')
# plt.axis('off')
# plt.plot(elX, elY, 'ko')

# # Label the electrodes
# for iCh in range(len(elX)):
#     plt.text(elX[iCh], elY[iCh], f'{iCh+1}', color='red', fontsize=10)

# plt.show()

# Example: Read 'info.mat' file
info_data = read_ecog_data('info')
# print(info_data)
# print(info_data.keys() if isinstance(info_data, dict) else info_data.shape)
# print(info_data['prms'])
# print(info_data['eyeData'])

# Example: Read data from channel 1
ecog_data_ch1 = read_ecog_data('1') # trial x time
plt.plot(ecog_data_ch1[1])
plt.show()
print(ecog_data_ch1)
print(ecog_data_ch1.shape)

# nan_cols = np.isnan(ecog_data_ch1).any(axis=0)
# array_clean = ecog_data_ch1[:, ~nan_cols]

# print(array_clean.shape)
# print(array_clean)

# absolute_errors_dt_0_001 = [1e-10, 4.734587313893513e-11, 4.519300065002667e-11, 4.3749586781501926e-11]
# absolute_errors_dt_0_0001 = [0.9e-10, 4.3599191181421874e-12, 7.7507827292602e-12, 1.2e-11]
# run_time_seconds = [10, 120, 1000, 6000] #, 54000]
# # absolute_errors_dt_0_0001 = [, , 1e-11]
# s = [20, 30, 40, 50]

# # Create the plot
# fig, ax1 = plt.subplots(figsize=(5, 3))

# ax1.set_xlabel('Mesh Size')
# ax1.set_ylabel('Absolute Error', color='k')
# ax1.scatter(s, absolute_errors_dt_0_001, color='k', s=20, alpha=0.7, marker='x')
# ax1.plot(s, absolute_errors_dt_0_001, color='k', linestyle='-', label="dt = 0.001")
# # ax1.scatter(s, absolute_errors_dt_0_0001, color=color, s=50, alpha=0.7)
# # ax1.plot(s, absolute_errors_dt_0_0001, color=color, linestyle='--', label="dt = 0.0001")
# ax1.set_yscale('log')
# ax1.tick_params(axis='y', labelcolor='k')
# ax1.set_xticks(s)
# # ax1.set_yticks([1e-10,8e-11, 6e-11, 5e-11])

# ax1.spines['top'].set_visible(False)

# ax2 = ax1.twinx()  
# color = 'tab:red'
# ax2.set_ylabel('Run Time (seconds)', color=color)
# # ax2.scatter(s, run_time_seconds, color=color, label='Run Time', s=50, alpha=0.7)
# ax2.plot(s, run_time_seconds, color=color, linestyle='-', alpha=0.3)
# ax2.set_yscale('log')
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()
# # ax1.legend(loc='upper left')
# # ax2.legend(loc='upper right')
# ax1.grid(False)

# ax2.spines['top'].set_visible(False)

# # Display the plot
# plt.show()