import mne
import numpy as np
from src.config.config import *

"""
Script to combine two FreeSurfer surfaces (surf)
"""

# read free surfer file
data_lh = mne.read_surface(surfer_file('lh'), read_metadata=True) 
# returns: coordinates, triangulation/faces, volume

data_rh = mne.read_surface(surfer_file('rh'), read_metadata=True) 

combined_data_cc = np.vstack((data_lh[0],data_rh[0]))

# TODO: need to change the faces
# add length of coordinates (axis=1)
combined_data_fc = np.vstack((data_lh[1],data_rh[1]), dtype='>i4')

combined_data_vol = data_lh[2] 

# write to surf
mne.write_surface(
    f'./both.{SURF}', 
    combined_data_cc, 
    combined_data_fc, 
    volume_info=combined_data_vol,
    overwrite=True, 
    file_format="freesurfer"
)
