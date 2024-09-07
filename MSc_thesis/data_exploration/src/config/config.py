import os

# to write .msh file, run gmsh_converter with simnibs_env activated
# to write .txt file, run label_converter with local .venv activated

SUBJECT_DIR = 'data/FreeSurfer5.3'
SUBJECT = 'fsaverage6'
label_dir = os.path.join(SUBJECT_DIR, SUBJECT, 'label')
surf_dir = os.path.join(SUBJECT_DIR, SUBJECT, 'surf')

ANNOT_NAME = 'Schaefer2018_100Parcels_7Networks_order.annot'
LABEL_NAME = 'cortex.label'

SURF = 'white' # ['white', 'inflated', 'orig', 'pial'] // Curvature: ['curv', 'sulc', 'avg_curv', 'avg_sulc'] 
HEMI = 'rh'

annot_file = lambda hemi=HEMI: os.path.join(label_dir, f'{hemi}.{ANNOT_NAME}')
label_file = lambda hemi=HEMI: os.path.join(label_dir, f'{hemi}.{LABEL_NAME}')
surfer_file = lambda hemi=HEMI: os.path.join(surf_dir, f'{hemi}.{SURF}')

# where to converted .msh files
OUTPUT_DIR = '../ML-NFT/data'
output_msh_path = lambda hemi=HEMI: os.path.join(OUTPUT_DIR, f'cortex100_7nets_{hemi}.msh')
output_txt_path = lambda hemi=HEMI: os.path.join(OUTPUT_DIR, f'cortex_labels_{hemi}.txt')
