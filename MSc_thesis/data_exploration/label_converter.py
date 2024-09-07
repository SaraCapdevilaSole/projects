import mne 
import nibabel.freesurfer.io as nfi
import numpy as np

from config.config import *
from utils.utils import save_list_to_txt

_BOTH = False

labels_annot, _, _ = nfi.read_annot(filepath=annot_file())

"""
Script to convert FreeSurfer labels to .txt files. Optional: combining the two hemispheres.
"""

if _BOTH:
    if HEMI == 'lh':
        other_HEMI = 'rh'
        labels_annot_lh = np.copy(labels_annot)
        labels_annot_rh, _, _ = nfi.read_annot(filepath=annot_file(other_HEMI))
    elif HEMI == 'rh':
        other_HEMI = 'lh'
        labels_annot_rh = np.copy(labels_annot)
        labels_annot_lh, _, _ = nfi.read_annot(filepath=annot_file(other_HEMI))

    # add +50 to right labels # TODO: keep 0 index constant?
    labels_annot_rh += 50
    print(labels_annot_rh)

    # combine: left first, right after, 
    combined_labels = np.hstack((labels_annot_lh, labels_annot_rh))

    print(len(labels_annot_rh), len(labels_annot_lh))
    print(len(combined_labels))

    _output_txt_path = output_txt_path('both')

else:
    _output_txt_path = output_txt_path()
    combined_labels = labels_annot

# save .txt file
save_list_to_txt(combined_labels, filename=_output_txt_path)