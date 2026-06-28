"""
Aggressive multi-pass coarsening on the Tsunami mesh.

One IS pass removes ~28% of vertices on a regular triangulation -- bounded by
the four-color / independent-set ratio. To match the "remove 75% per level"
behavior of structured-grid downsampling, we stack several IS passes into a
single "super-level" and accumulate the per-pass parameterizations.

At every sub-step k we record, for each removed vertex v:
  - the anchor triangle (in the indexing of the COARSE mesh at sub-step k)
  - the barycentric weights for that triangle
  - the attribute residual = a_fine(v) - bary . a_coarse_k[anchor]

The aggregate removal after K sub-steps is ~(1 - 0.72^K):
  K=1: 28%,  K=2: 48%,  K=3: 63%,  K=4: 73%,  K=5: 80%

For comparison we also run vertex clustering at a cell size tuned to remove
~75% in one shot, and report its residual distribution. Clustering is faster
but anchors fine samples at cluster centroids (no triangle); the embedded-node
parameterization is just `vertex_to_cluster[v]`, with no barycentric.

Outputs:
  outputs/tsunami_aggressive_residuals.png  -- residuals vs cumulative removal
  stdout                                     -- per-sub-step statistics
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import Delaunay
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_decimation import (
    independent_set_decimate,
    vertex_cluster,
    _build_vertex_neighbors,
    _ordered_link,
    _barycentric,
)

DATA_DIR = "/sessions/beautiful-stoic-albattani/mnt/umc/data/Tsunami"


# ----------------------------------------------------------------------
# Data loading (same as the simpler prototype)
# ----------------------------------------------------------------------

def load_tsunami():
    coords2 = np.fromfile(os.path.join(DATA_DIR, "coordinates.dat"),
                          dtype=np.float32).reshape(-1, 2).astype(np.float64)
    V = np.column_stack([coords2, np.zeros(len(coords2))])
    F = Delaunay(coords2).simplices.astype(np.int64)
    attr = np.fromfile(os.path.join(DATA_DIR, "height.dat.0"),
                       dtype=np.float32).astype(np.float64)
    return V, F, attr


# ----------------------------------------------------------------------
# 2D-safe anchor picker + parameterization builder
# ----------------------------------------------------------------------

def build_parameterization(V, F, removed_mask, old_to_new):
    """For every removed vertex, find the fan triangle whose barycentric
    coordinates contain it, and store (anchor_in_coarse_indices, bary)."""
    incident = defaultdict(list)
    for fi, tri in enumerate(F):
        for v in tri:
            incident[int(v)].append(fi)

    out = {}
    for v in range(len(V)):
        if not removed_mask[v]:
            continue
        ring = _ordered_link(v, incident[v], F)
        if ring is None or len(ring) < 3:
            continue
        a = V[ring[0]]
        best_i = None; best_pen = None; best_b = None
        for i in range(1, len(ring) - 1):
            b3 = _barycentric(V[v], a, V[ring[i]], V[ring[i + 1]])
            pen = -min(b3[0], b3[1], b3[2], 0.0)   # 0 if v inside
            if best_pen is None or pen < best_pen:
                best_pen = pen; best_i = i; best_b = b3
        tri_coarse = (
            int(old_to_new[ring[0]]),
            int(old_to_new[ring[best_i]]),
            int(old_to_new[ring[best_i + 1]]),
        )
        out[int(v)] = (tri_coarse, tuple(float(x) for x in best_b))
    return out


# ----------------------------------------------------------------------
# Aggressive multi-pass coarsener
# ----------------------------------------------------------------------

def aggressive_coarsen(V, F, attr, n_substeps=4):
    """Run `n_substeps` IS passes. After each pass:
      - record (removed_v -> (anchor_in_coarse_indices, bary, truth, prediction))
      - update (V, F, attr) to the coarse-mesh state
    Returns a list of dicts, one per sub-step.
    """
    cur_V = V.copy()
    cur_F = F.copy()
    cur_attr = attr.copy()
    n0 = len(V)
    history = []

    for k in range(n_substeps):
        result = independent_set_decimate(cur_V, cur_F)
        removed_mask = (result.old_to_new < 0)
        param = build_parameterization(cur_V, cur_F, removed_mask, result.old_to_new)
        attr_coarse = cur_attr[result.old_to_new >= 0]

        ids = np.fromiter(param.keys(), dtype=np.int64)
        preds = np.empty(len(ids), dtype=np.float64)
        for j, v in enumerate(ids):
            tri, bary = param[int(v)]
            preds[j] = (bary[0] * attr_coarse[tri[0]]
                        + bary[1] * attr_coarse[tri[1]]
                        + bary[2] * attr_coarse[tri[2]])
        truths = cur_attr[ids]
        residuals = truths - preds

        history.append({
            "sub": k + 1,
            "n_in":  len(cur_V),
            "n_out": len(result.vertices),
            "n_removed": int(removed_mask.sum()),
            "cumulative_removed": n0 - len(result.vertices),
            "cumulative_ratio": (n0 - len(result.vertices)) / n0,
            "residuals": residuals,
        })

        cur_V = result.vertices
        cur_F = result.faces
        cur_attr = attr_coarse

    return history


# ----------------------------------------------------------------------
# Aggressive single-shot vertex clustering
# ----------------------------------------------------------------------

def aggressive_clustering(V, F, attr, target_ratio=0.75):
    """Pick a cell size that produces ~target_ratio removal, then run
    vertex_cluster. The 'parameterization' for clustering is simply
    vertex_to_cluster: each fine vertex maps to its cluster centroid.
    Prediction = a_coarse[cluster_id]; residual = a_fine - a_coarse[cluster_id].
    """
    # Compute bounding box and pick cell size by bisection.
    bbox = V[:, :2].max(axis=0) - V[:, :2].min(axis=0)
    span = float(np.linalg.norm(bbox))
    lo, hi = 1e-3 * span, span
    best = None
    for _ in range(20):
        cs = 0.5 * (lo + hi)
        cr = vertex_cluster(V, F, cell_size=cs)
        ratio = 1 - len(cr.vertices) / len(V)
        if abs(ratio - target_ratio) < 0.01:
            best = (cs, cr)
            break
        if ratio < target_ratio:
            lo = cs
        else:
            hi = cs
        best = (cs, cr)
    cs, cr = best

    cluster_attr = np.zeros(len(cr.vertices))
    counts = np.zeros(len(cr.vertices))
    np.add.at(cluster_attr, cr.vertex_to_cluster, attr)
    np.add.at(counts, cr.vertex_to_cluster, 1)
    cluster_attr /= counts                # cluster-mean attribute

    predictions = cluster_attr[cr.vertex_to_cluster]
    residuals = attr - predictions
    return {
        "cell_size": cs,
        "n_clusters": len(cr.vertices),
        "ratio": 1 - len(cr.vertices) / len(V),
        "residuals": residuals,
        "cluster_attr": cluster_attr,
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def summarize(name, residuals, attr_range):
    abs_r = np.abs(residuals)
    rms = float(np.sqrt((residuals ** 2).mean()))
    print(f"  {name:<24s} n={len(residuals):7d}  "
          f"rms={rms:.3e}  rms/range={rms/attr_range:.3%}  "
          f"|max|/range={abs_r.max()/attr_range:.2%}")


def main():
    print("Loading Tsunami mesh...")
    V, F, attr = load_tsunami()
    attr_range = float(attr.max() - attr.min())
    n0 = len(V)
    print(f"  vertices: {n0:,}, faces: {len(F):,}, attribute range: {attr_range:.2f}")

    print("\nAggressive coarsening: 5 IS sub-steps (cumulative removal target ~80%)")
    history = aggressive_coarsen(V, F, attr, n_substeps=5)
    for h in history:
        print(f"  sub-step {h['sub']}: "
              f"{h['n_in']:>7d} -> {h['n_out']:>7d}  "
              f"({h['n_removed']:>5d} removed at this step;  "
              f"cumulative {h['cumulative_ratio']:.1%})")
    print("\nResiduals per sub-step (predictor: barycentric from coarse anchor):")
    for h in history:
        summarize(f"sub-step {h['sub']}", h['residuals'], attr_range)

    print("\nSingle-shot aggressive vertex clustering (~75% target):")
    clust = aggressive_clustering(V, F, attr, target_ratio=0.75)
    print(f"  cell_size={clust['cell_size']:.3f}  "
          f"clusters={clust['n_clusters']:,}  "
          f"ratio={clust['ratio']:.1%}")
    summarize("clustering", clust['residuals'], attr_range)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: cumulative removal vs rms residual per sub-step
    cum_ratios = [h['cumulative_ratio'] for h in history]
    rms_per_step = [float(np.sqrt((h['residuals'] ** 2).mean())) for h in history]
    axes[0].plot([0] + cum_ratios, [0] + rms_per_step, marker="o",
                 color="#1a3866", label="stacked IS sub-steps")
    axes[0].axhline(np.sqrt((clust['residuals'] ** 2).mean()),
                    color="#c0392b", linestyle="--",
                    label=f"clustering @ {clust['ratio']:.0%}")
    for h, r in zip(history, rms_per_step):
        axes[0].annotate(f"step {h['sub']}\n{h['cumulative_ratio']:.0%}",
                         (h['cumulative_ratio'], r),
                         textcoords="offset points", xytext=(6, 6), fontsize=8)
    axes[0].set_xlabel("cumulative vertices removed (fraction of original)")
    axes[0].set_ylabel("rms residual at this sub-step")
    axes[0].set_title("Aggregating IS sub-steps into one 'super-level'")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Right: residual histograms per sub-step, normalized by attribute range
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(history)))
    bins = 60
    bounds = max(abs(h['residuals']).max() for h in history)
    bin_edges = np.linspace(-bounds, bounds, bins)
    for h, color in zip(history, colors):
        axes[1].hist(h['residuals'], bins=bin_edges, alpha=0.35,
                     color=color,
                     label=f"sub-step {h['sub']} (rms={np.sqrt((h['residuals']**2).mean()):.1f})")
    axes[1].hist(clust['residuals'], bins=bin_edges, alpha=0.35,
                 color="#c0392b",
                 label=f"clustering (rms={np.sqrt((clust['residuals']**2).mean()):.1f})")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("residual = truth - prediction")
    axes[1].set_ylabel("count (log)")
    axes[1].set_title(f"Per-sub-step residuals  (attribute range {attr_range:.0f})")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].axvline(0, color="black", linewidth=0.5)

    fig.suptitle(
        "Tsunami height.dat.0 -- aggressive coarsening at ~75% removal",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__),
                            "tsunami_aggressive_residuals.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot -> {out_path}")


if __name__ == "__main__":
    main()
