import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

refinement_levels = [1,2,3,4,5,6,7,8,9,10]
num_vertices = []
# num_vertices_100 = []

for size in refinement_levels:
    mesh = IcosahedralSphereMesh(radius=1, refinement_level=size)
    mesh_coordinates = np.array(mesh.coordinates.dat.data)
    num_vertices.append(mesh_coordinates.shape[0])
    # mesh_100 = IcosahedralSphereMesh(radius=100, refinement_level=size)
    # mesh_coordinates_100 = np.array(mesh_100.coordinates.dat.data)
    # num_vertices_100.append(mesh_coordinates_100.shape[0])

plt.figure(figsize=(5, 4), dpi=150)
plt.semilogy(refinement_levels, num_vertices, color='gray', linestyle='-', zorder=1)
plt.scatter(refinement_levels, num_vertices, color='black', marker='o', zorder=2)
# plt.semilogy(refinement_levels, num_vertices_100, marker='o', label='r=100')
plt.xticks(refinement_levels, fontsize=10)
# plt.yticks(num_vertices, fontsize=10)
plt.xlabel('Refinement', fontsize=12)
plt.ylabel('Vertices', fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='black', alpha=0.2)
plt.savefig('num_vertices.png')