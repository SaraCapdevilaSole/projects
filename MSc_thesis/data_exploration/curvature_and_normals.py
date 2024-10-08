import numpy as np
import pyvista as pv
import os
import meshio

file_path = '/Users/saracapdevilasole/Desktop/meshes/hemispheres/lh' 
# read data
n = 164
save_dir = os.path.join(file_path, f'../norm_curv/norm_curv_{n}k')
os.makedirs(save_dir, exist_ok=True)
brain = pv.read(f"{file_path}/lh_inflated_{n}k.stl")
print(brain)
mesh = meshio.read(f"{file_path}/lh_inflated_{n}k.msh")
print(mesh.points.shape)

# compute normals
brain = brain.compute_normals(consistent_normals=False)  # correct behaviour

# normalised unit vectors: (5124,3)
normals = brain.point_normals

# print(len(normals))

plotter = pv.Plotter()
plotter.add_mesh(brain, color='silver')
plotter.add_mesh(brain.glyph(geom=pv.Arrow(), orient='Normals', scale=0.0001), color='red')
plotter.view_vector([-10, -10, 10])
plotter.show()
np.savetxt(f'{save_dir}/normals_{n}.txt', normals, delimiter=',')


# VISUALISE HALF: Slice the mesh to get the first half
bounds = brain.bounds
x_min, x_max = bounds[0], bounds[1]
x_mid = (x_max + x_min) / 2
first_half = brain.clip_box(bounds=[x_min, x_mid, bounds[2], bounds[3], bounds[4], bounds[5]])

plotter = pv.Plotter()
plotter.add_mesh(first_half, color='silver')
plotter.add_mesh(first_half.glyph(geom=pv.Arrow(), orient='Normals', scale=2.0), color='red')
plotter.view_vector([-10, -10, 10])
plotter.show()


# compute curvature - at each vertex there is a scalar curvature value: (5124,1)
curv = brain.curvature(curv_type="mean")
brain["mean_curvature"] = curv
np.savetxt(f'{save_dir}/curv_{n}.txt', curv, delimiter=',')

cmin = np.percentile(curv, 0)
cmax = np.percentile(curv, 95)
print(cmin, cmax)
brain.plot(scalars="mean_curvature", 
          cmap="jet", clim=[cmin,  cmax], cpos="xy")
