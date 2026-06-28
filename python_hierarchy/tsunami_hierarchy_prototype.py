"""
Prototype: build a multi-level hierarchy on a real UMC dataset (Tsunami) and
predict the scalar attribute at decimated nodes using the embedded-node
(barycentric) parameterization recorded by independent_set_decimate.

Pipeline
--------
1. Read coordinates.dat (float32, N*2) and connectivity.dat (int32, M*3) for
   the Tsunami triangle mesh. Read height.dat.0 (float32, N) as the attribute.
2. Run independent_set_decimate(V, F) to remove a maximal independent set of
   vertices. The decimator records, for every removed vertex v,
   detail_parameter[v] = (anchor_triangle, barycentric_coords),
   which IS the "embedded node" representation in coarse-mesh space.
3. Predict the attribute at every removed v as
       a_pred(v) = b0 * a(t0) + b1 * a(t1) + b2 * a(t2)
   where (t0, t1, t2) = anchor_triangle (always surviving vertices, since v was
   selected from an independent set), and (b0, b1, b2) = barycentric.
   The residual is a_true(v) - a_pred(v) -- the wavelet detail coefficient
   for the scalar field.
4. Repeat on the coarsened mesh for level 2.
5. Compare residual magnitudes against (a) the original attribute range and
   (b) the residual of the trivial "average of 1-ring neighbors" predictor,
   which is roughly what UMC's existing IDW/NBP predictors do.

Outputs:
  outputs/tsunami_residuals.png    -- histogram of residuals per level
  stdout                            -- summary statistics
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.spatial import Delaunay
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure we can import the existing mesh_decimation module that lives next
# to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_decimation import (
    independent_set_decimate,
    _build_vertex_neighbors,
    _ordered_link,
    _barycentric,
)
from collections import defaultdict


DATA_DIR = "/sessions/beautiful-stoic-albattani/mnt/umc/data/Tsunami"


def load_tsunami() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (V, F, attr) for the Tsunami mesh.

    V is (N, 3) float64 with z=0 (the IS decimator expects 3D positions).
    F is (M, 3) int64.
    attr is (N,) float64, the height field at timestep 0.
    """
    coords2 = np.fromfile(os.path.join(DATA_DIR, "coordinates.dat"),
                          dtype=np.float32).reshape(-1, 2).astype(np.float64)
    V = np.column_stack([coords2, np.zeros(len(coords2))])

    # Tsunami's connectivity.dat stores edges (from_edge=1 path in
    # adjacent_prediction.hpp), not triangles. We Delaunay-triangulate the 2D
    # coordinates to recover a manifold triangulation compatible with the
    # IS decimator. Shallow-water meshes are typically Delaunay anyway, so
    # this should reproduce the original triangulation up to tie-breaking.
    tri = Delaunay(coords2)
    F = tri.simplices.astype(np.int64)

    attr = np.fromfile(os.path.join(DATA_DIR, "height.dat.0"),
                       dtype=np.float32).astype(np.float64)
    assert len(V) == len(attr), (len(V), len(attr))
    return V, F, attr


def repick_anchor_2d(v_pos, ring_pts):
    """Choose the fan triangle (ring[0], ring[i], ring[i+1]) that contains v
    in barycentric coords; if v lies on a boundary, choose the closest one by
    out-of-range penalty.

    This is a 2D-safe replacement for the perpendicular-distance heuristic in
    mesh_decimation._pick_anchor, which degenerates in 2D (all distances zero).
    """
    best_tri_i = None
    best_score = None
    best_bary = None
    a = ring_pts[0]
    for i in range(1, len(ring_pts) - 1):
        b = ring_pts[i]
        c = ring_pts[i + 1]
        bary = _barycentric(v_pos, a, b, c)
        # negative penalty: how far below zero is the worst component?
        # for an inside point, this is 0.
        penalty = -min(bary[0], bary[1], bary[2], 0.0)
        if best_score is None or penalty < best_score:
            best_score = penalty
            best_tri_i = i
            best_bary = bary
    return best_tri_i, best_bary


def build_parameterization(
    V: np.ndarray,
    F: np.ndarray,
    removed_mask: np.ndarray,
    old_to_new: np.ndarray,
) -> dict:
    """Reconstruct (anchor_triangle_in_coarse_indices, bary) for every removed
    vertex, using the 2D-safe anchor picker.

    Returns: {removed_v -> (tri_coarse_indices_3tuple, bary_3-tuple)}
    """
    incident = defaultdict(list)
    for fi, tri in enumerate(F):
        for v in tri:
            incident[int(v)].append(fi)

    out = {}
    n = len(V)
    for v in range(n):
        if not removed_mask[v]:
            continue
        ring = _ordered_link(v, incident[v], F)
        if ring is None or len(ring) < 3:
            continue
        ring_pts = [V[r] for r in ring]
        _, bary = repick_anchor_2d(V[v], ring_pts)
        if bary is None:
            continue
        # We pick (ring[0], ring[i], ring[i+1]).
        # The bary was computed for the winning i; reconstruct it cleanly.
        # repick_anchor_2d already returns the best (i, bary); refetch i.
        best_i = None
        best_pen = None
        best_b = None
        a = ring_pts[0]
        for i in range(1, len(ring_pts) - 1):
            b_ = ring_pts[i]; c_ = ring_pts[i + 1]
            b3 = _barycentric(V[v], a, b_, c_)
            pen = -min(b3[0], b3[1], b3[2], 0.0)
            if best_pen is None or pen < best_pen:
                best_pen = pen; best_i = i; best_b = b3
        tri_fine = (ring[0], ring[best_i], ring[best_i + 1])
        # All three ring vertices are survivors (IS guarantees this) -> map to
        # coarse-mesh indices.
        tri_coarse = tuple(int(old_to_new[r]) for r in tri_fine)
        out[int(v)] = (tri_coarse, tuple(float(x) for x in best_b))
    return out


def predict_attr_from_parameterization(
    param: dict, attr_at_coarse_level: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict the scalar at every removed vertex from coarse-level neighbors.

    `param[v] = (tri_in_COARSE_indexing, bary)`.
    `attr_at_coarse_level` is indexed by coarse-mesh vertex ids.
    """
    removed_ids = np.fromiter(param.keys(), dtype=np.int64)
    predictions = np.empty(len(removed_ids), dtype=np.float64)
    for k, v in enumerate(removed_ids):
        tri, bary = param[int(v)]
        predictions[k] = (
            bary[0] * attr_at_coarse_level[tri[0]]
            + bary[1] * attr_at_coarse_level[tri[1]]
            + bary[2] * attr_at_coarse_level[tri[2]]
        )
    return removed_ids, predictions


def predict_attr_neighbor_mean(
    removed_ids: np.ndarray,
    neighbors: list[set[int]],
    attr_at_current_level: np.ndarray,
) -> np.ndarray:
    """Baseline: predict each removed vertex by averaging its 1-ring."""
    predictions = np.empty(len(removed_ids), dtype=np.float64)
    for k, v in enumerate(removed_ids):
        ring = list(neighbors[int(v)])
        if not ring:
            predictions[k] = attr_at_current_level[int(v)]
            continue
        predictions[k] = float(np.mean(attr_at_current_level[ring]))
    return predictions


def summary(name: str, residuals: np.ndarray, attr_range: float) -> None:
    abs_res = np.abs(residuals)
    print(f"  {name}: n={len(residuals):6d}  "
          f"mean={residuals.mean():+.4e}  std={residuals.std():.4e}  "
          f"|max|={abs_res.max():.4e}  "
          f"|max|/range={abs_res.max()/attr_range:.3%}  "
          f"rms/range={np.sqrt((residuals**2).mean())/attr_range:.3%}")


def main() -> None:
    print("Loading Tsunami mesh...")
    V, F, attr = load_tsunami()
    print(f"  vertices: {len(V):,}  faces: {len(F):,}")
    print(f"  attribute (height.dat.0): min={attr.min():.4f} max={attr.max():.4f} "
          f"range={attr.max() - attr.min():.4f}")
    attr_range = float(attr.max() - attr.min())

    # ---- Level 1 ----
    print("\nLevel 1 decimation (this takes ~15-30s for ~114k verts in pure Python)...")
    result1 = independent_set_decimate(V, F)
    n_removed_1 = len(result1.detail_parameter)
    print(f"  removed {n_removed_1:,} vertices "
          f"({n_removed_1 / len(V):.1%}); kept {(result1.old_to_new >= 0).sum():,}")
    print(f"  coarse mesh: {len(result1.vertices):,} V, {len(result1.faces):,} F")

    removed_mask_1 = (result1.old_to_new < 0)  # True for removed verts
    # Build parameterization in COARSE indexing so predictions consume the
    # coarse-level attribute array directly.
    param1 = build_parameterization(V, F, removed_mask_1, result1.old_to_new)
    attr_coarse_1 = attr[result1.old_to_new >= 0]  # survivor attributes
    ids1, pred1 = predict_attr_from_parameterization(param1, attr_coarse_1)
    truth1 = attr[ids1]
    res1 = truth1 - pred1

    # Baseline: simple neighbor-mean predictor on the fine mesh.
    neighbors_fine = _build_vertex_neighbors(F, len(V))
    pred1_baseline = predict_attr_neighbor_mean(ids1, neighbors_fine, attr)
    res1_baseline = truth1 - pred1_baseline

    # ---- Level 2 ----
    print("\nLevel 2 decimation...")
    result2 = independent_set_decimate(result1.vertices, result1.faces)
    n_removed_2 = len(result2.detail_parameter)
    print(f"  removed {n_removed_2:,} vertices; "
          f"coarse mesh: {len(result2.vertices):,} V, {len(result2.faces):,} F")

    # Attribute at level 1 (in compacted level-1 indexing) = survivor subset.
    attr_l1 = attr[result1.old_to_new >= 0]

    removed_mask_2 = (result2.old_to_new < 0)
    param2 = build_parameterization(
        result1.vertices, result1.faces, removed_mask_2, result2.old_to_new
    )
    attr_coarse_2 = attr_l1[result2.old_to_new >= 0]
    ids2_l1, pred2 = predict_attr_from_parameterization(param2, attr_coarse_2)
    truth2 = attr_l1[ids2_l1]
    res2 = truth2 - pred2

    # Baseline on level-1 mesh.
    neighbors_l1 = _build_vertex_neighbors(result1.faces, len(result1.vertices))
    pred2_baseline = predict_attr_neighbor_mean(ids2_l1, neighbors_l1, attr_l1)
    res2_baseline = truth2 - pred2_baseline

    # ---- Summary ----
    print("\nResidual statistics (predictor: barycentric from anchor triangle):")
    summary("level 1", res1, attr_range)
    summary("level 2", res2, attr_range)
    print("\nBaseline residuals (predictor: mean of 1-ring neighbors):")
    summary("level 1 baseline", res1_baseline, attr_range)
    summary("level 2 baseline", res2_baseline, attr_range)

    # ---- Histogram ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    bins = 80
    axes[0].hist(res1, bins=bins, alpha=0.6, color="#1a3866",
                 label=f"barycentric  (rms={np.sqrt((res1**2).mean()):.3e})")
    axes[0].hist(res1_baseline, bins=bins, alpha=0.5, color="#c0392b",
                 label=f"1-ring mean (rms={np.sqrt((res1_baseline**2).mean()):.3e})")
    axes[0].set_title(f"Level 1 residuals (n={n_removed_1:,})")
    axes[0].set_xlabel("residual = truth - prediction")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=9)
    axes[0].axvline(0, color="black", linewidth=0.5)

    axes[1].hist(res2, bins=bins, alpha=0.6, color="#1a3866",
                 label=f"barycentric  (rms={np.sqrt((res2**2).mean()):.3e})")
    axes[1].hist(res2_baseline, bins=bins, alpha=0.5, color="#c0392b",
                 label=f"1-ring mean (rms={np.sqrt((res2_baseline**2).mean()):.3e})")
    axes[1].set_title(f"Level 2 residuals (n={n_removed_2:,})")
    axes[1].set_xlabel("residual = truth - prediction")
    axes[1].legend(fontsize=9)
    axes[1].axvline(0, color="black", linewidth=0.5)

    fig.suptitle(
        "Tsunami height.dat.0 -- prediction residuals at removed nodes\n"
        f"attribute range = {attr_range:.3f}", fontsize=11,
    )
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "tsunami_residuals.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved histogram -> {out_path}")


if __name__ == "__main__":
    main()
