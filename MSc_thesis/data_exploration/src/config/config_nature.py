
hemisphere = 'lh' 
parc_name = 'Glasser360'

mesh_interest = 'inflated' # or 'midthickness'

data_dir = lambda mesh_interest, hemisphere: \
            f'data/Nature_files/converted/fsLR_32k_{mesh_interest}-{hemisphere}.vtk'