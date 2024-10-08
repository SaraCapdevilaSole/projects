# Brain Function Modelling with Physics-Informed Neural Networks

This repository contains the source code developed for my MSc project titled **"Brain Function Modelling with Physics-Informed Neural Networks."**

## Project Overview

This project aims to model brain functions using Physics-Informed Neural Networks (PINNs) to solve the Helmholtz equation across different dimensions and geometries. The key components of this project include:

### Key Components 

1. **2D Forward PINN**  
   - Solves the Helmholtz equation on a unit 2D square domain.

2. **3D Forward PINN**  
   - Solves the Helmholtz equation on:
     - A 3D sphere of a specified radius.
     - An input mesh (e.g., a brain mesh).

3. **3D Forward PINN with Surface Constraints**  
   - Solves the Helmholtz equation on a mesh (either a sphere or input mesh) while incorporating additional loss terms that impose surface constraints. 
   - Surface constraints are derived by enforcing that the standard gradients match the intrinsic gradients on the mesh's surface【1】.

4. **3D Inverse PINN**  
   - Capable of solving the Helmholtz equation on arbitrary meshes.
   - Can utilize real data to address inverse problems using the Helmholtz equation.

## Getting Started

This code builds on the source code from the [PirateNets](https://github.com/PredictiveIntelligenceLab/jaxpi/tree/main) repository【2】. Follow the steps below to run the code:

### Installation Steps

1. **Clone the PirateNets Repository**  
   Clone the PirateNets repository to your local machine:
   ```bash
   git clone https://github.com/PredictiveIntelligenceLab/jaxpi.git
   ```

2. **Clone this Repository**  
   Place the files from this repository into the `/examples` folder of the PirateNets repository.

3. **Run the Code**  
   Execute the `main.py` file from each subproject. For further configurations, refer to the [PirateNets GitHub repository](https://github.com/PredictiveIntelligenceLab/jaxpi/tree/main).

## References

1. Fang, Z., Zhang, J., & Yang, X. (2021). A Physics-Informed Neural Network Framework For Partial Differential Equations on 3D Surfaces: Time-Dependent Problems. arXiv preprint arXiv:2103.13878. [Link to Paper](https://arxiv.org/abs/2103.13878)

2. Wang, S., Li, B., Chen, Y., & Perdikaris, P. (2024). PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks. arXiv preprint arXiv:2402.00326. [Link to Paper](https://arxiv.org/abs/2402.00326)

3. PirateNets Repository: [GitHub Link](https://github.com/PredictiveIntelligenceLab/jaxpi/tree/main)
