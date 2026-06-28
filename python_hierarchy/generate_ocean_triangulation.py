"""
Generate a consistently-oriented surface triangulation for the Ocean dataset.

Background
----------
The Ocean points all lie on a sphere of radius ~6371 km (Earth's radius): they
are an ocean-surface mesh embedded in 3-D.  Its connectivity is stored only as
an undirected edge list (conn.dat, 2 int32 per edge), which the barycentric
hierarchy cannot use — that needs actual triangular faces to trace cyclic
1-rings, fan-retriangulate holes, and form barycentric anchors.

Because the points sit on a convex sphere, the CONVEX HULL of the 3-D points is
exactly the surface triangulation.  We additionally orient every triangle so its
winding matches the outward (radial) normal, which `ordered_link` in
parameterization.hpp relies on to trace a consistent 1-ring.

Usage
-----
    python generate_ocean_triangulation.py <data_dir>

    <data_dir>/coords.dat       float32, (N, 3)   -> input
    <data_dir>/triangulation.dat int32,  (T, 3)   -> output (consistently oriented)
"""

import sys
import os
import numpy as np
from scipy.spatial import ConvexHull


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/Ocean"

    coords = np.fromfile(
        os.path.join(data_dir, "coords.dat"), dtype=np.float32
    ).reshape(-1, 3).astype(np.float64)
    print(coords)
    n = len(coords)
    r = np.linalg.norm(coords, axis=1)
    print(f"Vertices: {n}   radius mean={r.mean():.1f} std={r.std():.4f} "
          f"(should be ~constant => points on a sphere)")

    hull = ConvexHull(coords)
    tris = hull.simplices.astype(np.int64)            # (T, 3)
    print(f"Convex-hull triangles: {len(tris)}")

    # Orient each triangle so its winding matches the outward (radial) normal,
    # so the surface is consistently oriented for ordered_link.
    p0 = coords[tris[:, 0]]
    p1 = coords[tris[:, 1]]
    p2 = coords[tris[:, 2]]
    face_normal = np.cross(p1 - p0, p2 - p0)
    outward = (p0 + p1 + p2) / 3.0                     # centroid ~ radial dir (sphere @ origin)
    flip = np.einsum("ij,ij->i", face_normal, outward) < 0.0
    tris[flip] = tris[flip][:, ::-1]                   # reverse winding where needed
    print(f"Flipped {int(flip.sum())} triangles for consistent outward orientation")

    # out = tris.astype(np.int32)
    # out_path = os.path.join(data_dir, "triangulation.dat")
    # out.tofile(out_path)
    # print(f"Saved -> {out_path}  ({os.path.getsize(out_path):,} bytes, {len(out)} tris)")


if __name__ == "__main__":
    main()
