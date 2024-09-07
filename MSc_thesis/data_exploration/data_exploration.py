"""
Data exploration script:

mne: 
    - mne.read_surface():
        - read free surfer files
    - mne.write_surface(): 
        - write to 'auto', 'freesurfer', and 'obj'

simnibs: 
    - read free surfer files
    - convert: surf -> .msh
"""


#%%
import mne 
import os

from config.config import *

# use the code below to read from non-curved surfaces
labels = mne.read_labels_from_annot(
    subject=SUBJECT, 
    annot_fname=annot_file(),
    subjects_dir=SUBJECT_DIR,
    hemi=HEMI,
    surf_name='white',#SURF,
)

#%% - Using Nibabel
import nibabel.freesurfer.io as nfi
# read annotation file  .annot file
labels, ctab, names = nfi.read_annot(filepath=annot_file())
#%%
# Load in a Freesurfer .label file.
label_array = nfi.read_label(filepath=os.path.join(label_dir, HEMI + '.cortex.label'))
print(label_array)

#%% surf (curvature) files
curv_labels = nfi.read_morph_data(filepath=os.path.join(surf_dir, HEMI + '.white'))

#%% read .gii files
import nibabel as nib
from config.config import *

# img = nib.load(()))
# %% - convert from .gii ----> .vtk
from config.config import *
import itk

eamon_gii = 'data/eamon_data/cortex_5124.surf.gii' 
mesh = itk.meshread(eamon_gii)

itk.meshwrite(mesh, 'data/eamon_data/output_TEST.vtk')

# %% - convert from .vtk to ---> .msh
import meshio 

mesh = meshio.read(
    filename="data/Nature_files/converted/fsLR_32k_midthickness-lh.vtk",  # string, os.PathLike, or a buffer/open file
    # file_format="stl",  # optional if filename is a path; inferred from extension
    # see meshio-convert -h for all possible formats
)

meshio.write("./output.msh", mesh, file_format="gmsh22")


# %%
import vtk
from vtk import vtkAppendPolyData, vtkPolyDataWriter, vtkPolyDataReader, vtkPolyData
from config.config_nature import *

filename = "data/Nature_files/converted/fsLR_32k_midthickness-lh.vtk"

reader = vtkPolyDataReader()

reader.SetFileName(filename)
reader.Update()
polydata = reader.GetOutput()
# polydata.ShallowCopy(reader.GetOutput())

writer = vtk.vtkUnstructuredGridWriter()
writer.SetFileName('output.vtk')
writer.SetInputData(polydata)
writer.Write()

# %% - using vtk to extract data
# import vtk
# from vtk import *
# from vtk.util.numpy_support import vtk_to_numpy

# reader = vtk.vtkXMLUnstructuredGridReader()
# reader.SetFileName(in_data_path)
# reader.Update()
# points = np.array(reader.GetOutput().GetPoints().GetData() )
# phi_vtk_array = reader.GetOutput().GetPointData().GetArray("phi_e")
# phi_numpy_array = vtk_to_numpy(phi_vtk_array )

# writer = vtk.vtkXMLUnstructuredGridWriter()