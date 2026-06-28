"""
Generate a face-centroid triangulation for the Tsunami dataset from mesh.nc.

Background
----------
The Tsunami binary files are face-centred: each "vertex" in coordinates.dat
is a triangle face-centroid, and connectivity.dat is the face-edge dual graph
(max degree 3, no triangles).  build_parameterization needs a proper triangular
face list so that ordered_link can trace cyclic 1-rings.

Method: node-fan triangulation
-------------------------------
For every mesh node n with incident faces [f0, f1, ..., f_{k-1}] (ordered
cyclically by the angle of their centroid around n), the fan

    (f0, f1, f2),  (f0, f2, f3),  ...,  (f0, f_{k-2}, f_{k-1})

tiles the region around n in face-centroid space.  Repeating for all nodes
produces a full triangulation of the 113 885 face centroids that is derived
directly from the mesh topology — no Delaunay computation required.

Usage
-----
    python generate_centroid_triangulation.py <mesh_nc> <out_dir>

    <mesh_nc>  path to mesh.nc (contains Mesh2_face_nodes, Mesh2_node_x/y)
    <out_dir>  directory where centroid_triangulation.dat will be written

Output
------
    <out_dir>/centroid_triangulation.dat  -- int32, shape (T, 3)
        Each row gives the indices of three face centroids that form a
        triangle in centroid space.  Loaded by test_parameterization_tsunami.
"""

import sys
import os
import numpy as np
import netCDF4 as nc


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <mesh_nc> <out_dir>")
        sys.exit(1)

    mesh_nc = sys.argv[1]
    out_dir = sys.argv[2]

    # ------------------------------------------------------------------
    # Load mesh topology from NC file
    # ------------------------------------------------------------------
    ds = nc.Dataset(mesh_nc, 'r')
    node_x   = np.array(ds.variables['Mesh2_node_x'][:],    dtype=np.float64)
    node_y   = np.array(ds.variables['Mesh2_node_y'][:],    dtype=np.float64)
    triangles = np.array(ds.variables['Mesh2_face_nodes'][:], dtype=np.int32)
    ds.close()

    n_nodes = len(node_x)
    n_faces = len(triangles)
    print(f"Nodes: {n_nodes},  Faces: {n_faces}")

    # ------------------------------------------------------------------
    # Face centroids (same formula as read_mesh.txt)
    # ------------------------------------------------------------------
    cell_x = (node_x[triangles[:, 0]] + node_x[triangles[:, 1]] + node_x[triangles[:, 2]]) / 3.0
    cell_y = (node_y[triangles[:, 0]] + node_y[triangles[:, 1]] + node_y[triangles[:, 2]]) / 3.0

    # ------------------------------------------------------------------
    # Build node-to-face lists
    # ------------------------------------------------------------------
    node_faces = [[] for _ in range(n_nodes)]
    for fi in range(n_faces):
        for ni in triangles[fi]:
            node_faces[ni].append(fi)

    # ------------------------------------------------------------------
    # Fan-triangulate each node's incident faces in cyclic angular order
    # ------------------------------------------------------------------
    ctris = []
    for ni in range(n_nodes):
        faces = node_faces[ni]
        k = len(faces)
        if k < 3:
            continue
        # Sort by angle of face centroid around node ni
        dx = cell_x[faces] - node_x[ni]
        dy = cell_y[faces] - node_y[ni]
        order = np.argsort(np.arctan2(dy, dx))
        sf = [faces[i] for i in order]
        # Fan anchored at sf[0]: (sf[0], sf[i], sf[i+1]) for i in [1, k-2]
        for i in range(1, k - 1):
            ctris.append((sf[0], sf[i], sf[i + 1]))

    ctris = np.array(ctris, dtype=np.int32)
    print(f"Centroid triangles: {len(ctris)}")

    out_path = os.path.join(out_dir, 'centroid_triangulation.dat')
    ctris.tofile(out_path)
    print(f"Saved -> {out_path}  ({os.path.getsize(out_path):,} bytes)")


if __name__ == '__main__':
    main()
