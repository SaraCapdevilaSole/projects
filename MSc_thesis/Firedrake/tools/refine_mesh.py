import numpy as np
import meshio
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import SphereConfig

def make_3d_animation(mesh_file, scale):
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))

    mesh = meshio.read(mesh_file)
    pos = mesh.points
    faces = mesh.cells[0].data

    ax.cla()
    ax.set_aspect('equal')
    ax.set_axis_off()

    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)

    anim_path = f'refined_meshes/32k_scale={scale}.pdf'
    fig.savefig(anim_path)
    plt.show(block=True)

def max_edge_length(mesh_cells, new_mesh_points):
    max_edge_length = 0
    edge_length = lambda p1, p2: np.linalg.norm(p1 - p2)
    for triangle in mesh_cells[0].data:
        p0, p1, p2 = new_mesh_points[triangle]
        edges = [
            edge_length(p0, p1),
            edge_length(p1, p2),
            edge_length(p2, p0)
        ]
        max_edge_length = max(max_edge_length, *edges)

    print(f'Max edge length: {max_edge_length * 1e3:.2f}e-3 mm')


def main(scale, save_msh, msh_vis):
    c = SphereConfig(dt=1e-4, T=0.5, sampling_nom=1e-3, prec='32k', scale=None)
    mesh = meshio.read(c.path_msh)

    scale_factor = np.abs(mesh.points).max() * scale
    print("Scale factor: ", 1/scale_factor)
    new_mesh_points = mesh.points / scale_factor
    max_edge_length(mesh.cells, new_mesh_points)
  
    if save_msh:
        print("USE GMSH -> Tools -> Manipulator -> scale")
    if msh_vis:
        c_out = SphereConfig(dt=1e-4, T=0.5, sampling_nom=1e-3, prec='32k', scale=scale)
        mesh = meshio.read(c_out.path_msh)
        print(mesh.points.max())

        c = SphereConfig(dt=1e-4, T=0.5, sampling_nom=1e-3, prec='32k', scale=None)
        mesh = meshio.read('/vol/bitbucket/sc3719/firedrake_simulation/data/input_data/sphere_hemis/sphere.msh')
        print(mesh.points.max())
        # path = c.path_msh
        path = '/vol/bitbucket/sc3719/firedrake_simulation/data/input_data/sphere_hemis/sphere.msh'
        make_3d_animation(c.path_msh, scale=scale)
        print("Mesh triangles visualised.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transform mesh coordinates and save to a new file.')
    parser.add_argument('scale', default=20, type=int, help='Scale factor for the mesh transformation.')
    parser.add_argument('save_msh', default=False, type=bool, help='Save refined mesh.')
    parser.add_argument('msh_vis', default=True, type=bool, help='Create visualisation.')

    args = parser.parse_args()
    main(args.scale, args.save_msh, args.msh_vis)
