"""
Two fast mesh-coarsening algorithms with parameterization tracking:

  1. vertex_cluster(...)         -- Rossignac-Borrel spatial binning, O(n)
  2. independent_set_decimate(...) -- one level of IS-based vertex removal, O(n)

Both return a coarsened mesh plus a parameterization map that records, for every
input vertex, where it lives in the coarsened mesh. The parameterization is the
data you need to drive a butterfly-wavelet (or any other multiresolution) analysis.

Only dependency: numpy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------
# 1. Vertex clustering (Rossignac & Borrel, 1993)
# ----------------------------------------------------------------------

@dataclass
class ClusterResult:
    vertices: np.ndarray         # (K, 3) cluster representatives
    faces: np.ndarray            # (M', 3) non-degenerate faces in the coarse mesh
    vertex_to_cluster: np.ndarray  # (N,) mapping fine vertex index -> coarse vertex index


def vertex_cluster(vertices: np.ndarray, faces: np.ndarray, cell_size: float) -> ClusterResult:
    """
    Overlay the mesh with a uniform 3D grid of side `cell_size` and replace
    every vertex falling into a cell with the centroid of that cell.

    Time complexity: O(n) with a hash-based unique-cell pass.
    Quality: low (grid is geometry-blind), but suitable as a fast first pass.
    """
    # 1. Integer cell index per vertex.
    cell_ids = np.floor(vertices / cell_size).astype(np.int64)

    # 2. Unique cells get consecutive cluster indices. `inverse` is the
    #    fine -> coarse map, which doubles as the parameterization.
    _, vertex_to_cluster = np.unique(cell_ids, axis=0, return_inverse=True)
    K = int(vertex_to_cluster.max()) + 1

    # 3. Cluster representative = centroid of the fine vertices in that cell.
    centroids = np.zeros((K, 3), dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)
    np.add.at(centroids, vertex_to_cluster, vertices)
    np.add.at(counts, vertex_to_cluster, 1)
    centroids /= counts[:, None]

    # 4. Remap face indices through the cluster map.
    remapped = vertex_to_cluster[faces]

    # 5. Drop faces whose vertices collapsed to fewer than 3 clusters.
    keep = (
        (remapped[:, 0] != remapped[:, 1])
        & (remapped[:, 1] != remapped[:, 2])
        & (remapped[:, 2] != remapped[:, 0])
    )
    remapped = remapped[keep]

    # 6. Drop duplicate triangles (same triple ignoring orientation).
    canon = np.sort(remapped, axis=1)
    _, first_occurrence = np.unique(canon, axis=0, return_index=True)
    coarse_faces = remapped[np.sort(first_occurrence)]

    return ClusterResult(
        vertices=centroids,
        faces=coarse_faces,
        vertex_to_cluster=vertex_to_cluster,
    )


# ----------------------------------------------------------------------
# 2. Independent-set decimation (second-generation wavelets, Schroeder/Sweldens)
# ----------------------------------------------------------------------

@dataclass
class DecimateResult:
    vertices: np.ndarray         # (N - |IS|, 3) surviving vertex positions
    faces: np.ndarray            # (M', 3) faces after hole retriangulation
    old_to_new: np.ndarray       # (N,) coarse index per surviving vertex; -1 if removed
    detail_parameter: dict       # removed_fine_v -> (anchor_triangle, barycentric)
    detail_coefficient: dict     # removed_fine_v -> displacement vector (3,)


def _build_vertex_neighbors(faces: np.ndarray, n: int) -> list[set[int]]:
    neighbors = [set() for _ in range(n)]
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return neighbors


def _maximal_independent_set(neighbors: list[set[int]], priority: np.ndarray) -> np.ndarray:
    """Greedy MIS: iterate vertices in order of `priority` (low first), add to set
    if no neighbor is already in the set."""
    n = len(neighbors)
    in_set = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    for v in np.argsort(priority, kind="stable"):
        if not blocked[v]:
            in_set[v] = True
            blocked[v] = True
            for u in neighbors[v]:
                blocked[u] = True
    return in_set


def _ordered_link(v: int, incident_faces: list[int], faces: np.ndarray) -> list[int] | None:
    """Return the ring of v's neighbors in cyclic order. Returns None for
    boundary or non-manifold vertices (we leave those alone)."""
    next_in_link: dict[int, int] = {}
    for fi in incident_faces:
        tri = faces[fi]
        i = int(np.where(tri == v)[0][0])
        a, b = int(tri[(i + 1) % 3]), int(tri[(i + 2) % 3])
        if a in next_in_link:           # repeated edge -> non-manifold
            return None
        next_in_link[a] = b

    start = next(iter(next_in_link))
    ring = [start]
    cur = start
    while True:
        nxt = next_in_link.get(cur)
        if nxt is None:                 # boundary
            return None
        if nxt == start:
            break
        ring.append(nxt)
        cur = nxt
        if len(ring) > len(next_in_link):
            return None
    if len(ring) != len(next_in_link):
        return None
    return ring


def _barycentric(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of p projected into the plane of triangle (a, b, c)."""
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-14:
        return np.array([1.0, 0.0, 0.0])
    beta = (d11 * d20 - d01 * d21) / denom
    gamma = (d00 * d21 - d01 * d20) / denom
    return np.array([1.0 - beta - gamma, beta, gamma])


def independent_set_decimate(vertices: np.ndarray, faces: np.ndarray) -> DecimateResult:
    """
    One level of independent-set decimation. Finds a maximal independent set of
    vertices (no two adjacent), removes all of them simultaneously, retriangulates
    each resulting hole by a fan, and records the parameterization + detail
    coefficient for every removed vertex.

    Time complexity: O(n) per level. Repeat for additional levels to build a full
    wavelet hierarchy.
    """
    n = len(vertices)
    neighbors = _build_vertex_neighbors(faces, n)

    # Prefer to remove low-valence vertices first: holes are smaller and easier
    # to retriangulate cleanly.
    valence = np.array([len(neighbors[v]) for v in range(n)])
    to_remove = _maximal_independent_set(neighbors, priority=valence)

    incident = defaultdict(list)
    for fi, tri in enumerate(faces):
        for v in tri:
            incident[int(v)].append(fi)

    removed_face_set: set[int] = set()
    new_triangles: list[list[int]] = []
    detail_parameter: dict[int, tuple[tuple[int, int, int], np.ndarray]] = {}
    detail_coefficient: dict[int, np.ndarray] = {}

    for v in range(n):
        if not to_remove[v]:
            continue
        ring = _ordered_link(v, incident[v], faces)
        if ring is None or len(ring) < 3:
            to_remove[v] = False        # leave boundary / non-manifold vertices in place
            continue

        # Fan triangulation of the hole, anchored at ring[0].
        retri = [[ring[0], ring[i], ring[i + 1]] for i in range(1, len(ring) - 1)]
        new_triangles.extend(retri)
        removed_face_set.update(incident[v])

        # Pick the retri triangle whose plane is closest to v, and record
        # the barycentric position of v in that triangle (the parameterization),
        # plus the residual displacement from the plane (the detail coefficient).
        best = None
        for tri in retri:
            a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            bary = _barycentric(vertices[v], a, b, c)
            proj = bary[0] * a + bary[1] * b + bary[2] * c
            displacement = vertices[v] - proj
            dist = np.linalg.norm(displacement)
            if best is None or dist < best[0]:
                best = (dist, tri, bary, displacement)
        _, tri, bary, displacement = best
        detail_parameter[v] = (tuple(tri), bary)
        detail_coefficient[v] = displacement

    # Build the surviving face list.
    survivors = [tri.tolist() for fi, tri in enumerate(faces) if fi not in removed_face_set]
    survivors.extend(new_triangles)
    all_faces = np.asarray(survivors, dtype=np.int64)

    # Compact the vertex array.
    keep_mask = ~to_remove
    old_to_new = -np.ones(n, dtype=np.int64)
    old_to_new[keep_mask] = np.arange(int(keep_mask.sum()))
    coarse_vertices = vertices[keep_mask]
    coarse_faces = old_to_new[all_faces]

    return DecimateResult(
        vertices=coarse_vertices,
        faces=coarse_faces,
        old_to_new=old_to_new,
        detail_parameter=detail_parameter,
        detail_coefficient=detail_coefficient,
    )


# ----------------------------------------------------------------------
# Demo: build a triangulated grid mesh, then coarsen with both methods.
# ----------------------------------------------------------------------

def make_grid_mesh(nx: int = 17, ny: int = 17, amplitude: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """A nx-by-ny grid of vertices with a small sinusoidal bump, regularly triangulated."""
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = amplitude * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    V = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    F = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = j * nx + (i + 1)
            v01 = (j + 1) * nx + i
            v11 = (j + 1) * nx + (i + 1)
            F.append([v00, v10, v11])
            F.append([v00, v11, v01])
    return V, np.asarray(F, dtype=np.int64)


def _print_mesh_stats(name: str, V: np.ndarray, F: np.ndarray) -> None:
    edges = set()
    for tri in F:
        a, b, c = sorted(int(x) for x in tri)
        edges.add((a, b))
        edges.add((b, c))
        edges.add((a, c))
    print(f"  {name:<28s} V={len(V):5d}  F={len(F):5d}  E={len(edges):5d}")


if __name__ == "__main__":
    V, F = make_grid_mesh(nx=17, ny=17)
    print("Input mesh:")
    _print_mesh_stats("(17 x 17 grid)", V, F)

    print("\n[1] Vertex clustering with cell_size = 0.125 (target ~9x9 grid of cells):")
    cr = vertex_cluster(V, F, cell_size=0.125)
    _print_mesh_stats("after clustering", cr.vertices, cr.faces)
    print(f"  parameterization: {len(cr.vertex_to_cluster)} fine vertices -> "
          f"{int(cr.vertex_to_cluster.max()) + 1} coarse clusters")

    print("\n[2] Independent-set decimation, three successive levels:")
    cur_V, cur_F = V.copy(), F.copy()
    for level in range(3):
        result = independent_set_decimate(cur_V, cur_F)
        n_removed = len(result.detail_coefficient)
        max_disp = (
            max(np.linalg.norm(d) for d in result.detail_coefficient.values())
            if result.detail_coefficient else 0.0
        )
        print(f"  level {level + 1}: removed {n_removed:4d} vertices, "
              f"max detail magnitude = {max_disp:.4f}")
        _print_mesh_stats(f"    after level {level + 1}", result.vertices, result.faces)
        cur_V, cur_F = result.vertices, result.faces
