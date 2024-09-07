# from vtk import (vtkSphereSource, vtkPolyData, vtkDecimatePro)
import trimesh as tr
import pyfqmr
import meshio
from src.config.config import output_msh_path, output_txt_path

"""
Script to simplify mesh (NOT USED)
NOTE: Use SurfIce instead! -> can't do labels
"""
mesh = meshio.read(output_msh_path('lh'))

# mesh = tr.load_mesh('./Stanford_Bunny_sample.stl')

# mesh_simplifier = pyfqmr.Simplify()
# # mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
# mesh_simplifier.setMesh(mesh.vertices, mesh.faces)
# mesh_simplifier.simplify_mesh(target_count = 1000, aggressiveness=7, preserve_border=True, verbose=10)

