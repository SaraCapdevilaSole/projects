# Helmholtz Equation Simulation using Firedrake

This project simulates the Helmholtz equation using the Firedrake finite element library. The main file contains the implementation of the simulation, while the other files provide methods for evaluating the accuracy and convergence of the simulation under various conditions.

## Project Structure

### Main File:

- **`helmholtz_Q_linear.py`**: 
  - This is the primary script of the project. It contains the core implementation for simulating the dynamics of the Helmholtz equation using Firedrake. The script handles the setup of the equation, initial conditions, and calls relevant methods to solve the equation numerically.

### Evaluation Files:

The remaining files in the project focus on evaluating the simulation to ensure its accuracy and reliability. These evaluation methods check for convergence with finer meshes and temporal integration steps across varying dimensions.

- **Evaluation goals include**:
  - Verifying that the simulation results converge as the mesh resolution is refined.
  - Ensuring that the temporal integration methods produce accurate results at smaller time steps.
  - Analysing the simulation's performance across different spatial dimensions (e.g., 2D, 3D).

---

## How to Use

1. **Run the Main Simulation**:
   - Execute `helmholtz_Q_linear.py` to perform the simulation of the Helmholtz equation. The script handles the setup and computation using Firedrake.

2. **Evaluate Convergence and Accuracy**:
   - Use the provided evaluation methods to test the accuracy of the simulation. These methods will check if the solution converges as expected with more refined meshes and smaller time steps.

---

## Dependencies

Ensure the following key dependencies are installed before running the scripts:

- Python (>= 3.x)
- [Firedrake](https://www.firedrakeproject.org/) (required for solving the Helmholtz equation using finite element methods)
- NumPy
- SciPy
- Matplotlib

---

## About the Helmholtz Equation

The Helmholtz equation is a partial differential equation that describes how physical quantities such as wave fields behave in space and time. This project provides a numerical solution using finite element methods through the Firedrake library, which is particularly useful for solving PDEs efficiently.

