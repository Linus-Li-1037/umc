"""
Pre-compute the Delaunay triangulation for the Tsunami dataset and save it
as a binary int32 file so the C++ test_parameterization_tsunami can load it.

Usage:
    python generate_tsunami_triangulation.py <data_dir>

Output:
    <data_dir>/triangulation.dat  -- int32 triangle indices, 3 per row (M x 3)
"""

import sys
import os
import numpy as np
from scipy.spatial import Delaunay

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/Tsunami"

    coords = np.fromfile(
        os.path.join(data_dir, "coordinates.dat"), dtype=np.float32
    ).reshape(-1, 2)
    print(f"Vertices: {len(coords)}")

    tri = Delaunay(coords).simplices.astype(np.int32)
    print(f"Triangles: {len(tri)}")

    out_path = os.path.join(data_dir, "triangulation.dat")
    tri.tofile(out_path)
    print(f"Saved -> {out_path}  ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    main()
