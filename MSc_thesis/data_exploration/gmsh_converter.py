import simnibs 

from config.config import *

"""
Script to convert a FreeSurfer surface into a .msh file. 
"""

_BOTH = True

if _BOTH:
    hemi = 'both'
    surfer_file_name = f'./both.{SURF}' # saved in same dir

else:
    hemi = HEMI
    surfer_file_name = surfer_file()

# read surfer file
mesh_fs = simnibs.read_freesurfer_surface(fn=surfer_file_name)

print(mesh_fs)

# write to file
simnibs.write_msh(mesh_fs, file_name=output_msh_path(hemi))

