# Project Overview

This project contains various utilities, configuration files, and data processing scripts related to solving ODEs, handling meshes, and processing brain data (ECoG). The primary purpose of the codebase is to solve ODE problems, format data for use with Physics-Informed Neural Networks (PINNs), and preprocess brain connectivity data. Below is a brief description of the key files in the directory.

## Folder Structure

### `/src`

This folder contains the utility functions, configuration files, and key methods for the project.

- **utils**: Includes ODE solvers and various methods related to numerical analysis and data processing.
- **config**: Includes configuration files for different methods and functions e.g., file paths, parameters for ODE solvers...
  
---

### Key Files:

- **`PINNs_data_formatter.py`**: 
  - This script is responsible for formatting data simulated using FEniCS/Dolfin (Firdrake) into a format compatible with PINNs (Physics-Informed Neural Networks). It ensures that the data is structured for easy handling during model training and evaluation.

- **`balloon.py`**: 
  - The main file for running ODE solver methods. This converts the simulated signal into a signal representing fMRI-BOLD data.

- **`create_white_noise.py`**: 
  - This script generates white noise spectra, which are used to test the robustness and accuracy of the ODE solvers.

- **`curvature_and_normals.py`**: 
  - Extracts curvature and normal vectors from an input mesh. This script is useful for geometric processing and for analysing surface properties of 3D meshes.

- **`human_ecog.py`**: 
  - This file loads and visualises real human ECoG (Electrocorticography) data from a Stanford study. It provides functionality for analysing and interpreting brain signal data in a meaningful way.

- **`Functional_connectivity.py` & `visualise_downsampled_data.py`**: 
  - These scripts handle preprocessing and visualising statistical dependencies between different brain regions and time series data. They are used for studying functional connectivity and brain activity patterns.

- **`combine_surf.py`, `gmsh_converter.py`, `label_converter.py`, `mesh_data_formatter.py`**: 
  - These scripts are used for modifying data types, handling mesh manipulations, and processing labels. They are essential for tasks such as combining mesh surfaces, converting mesh data between different formats, and formatting mesh data for analysis.

