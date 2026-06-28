"""
Visualize the multilevel MIS hierarchy of barycentric_hierarchy.hpp on the
Ocean temperature field, ACCUMULATIVELY from coarsest (base) to finest.

It replicates the C++ level construction (greedy low-valence MIS -> prune
un-ringed removals -> fan-retriangulate holes, carrying global ids), tagging
every vertex with the level at which it was removed (0 = base survivor, L =
removed at level L; L=1 is the finest removal, L=num_levels the coarsest).

The decode/refinement order is coarsest -> finest:  base, then level N, N-1,
..., 1.  Each panel shows the points present so far, colored by their true
temperature, on a lon/lat map — so you watch the sampling densify from a coarse
sketch to the full-resolution field.  (A single level's removed points alone are
scattered and meaningless; the accumulation is the meaningful view.)

Usage:
    python visualize_ocean_levels.py <data_dir> [num_levels=6]

Reads : <data_dir>/coords.dat (float32 N*3), triangulation.dat (int32 T*3),
        temperature.dat.0 (float32 N)
Writes: <data_dir>/ocean_levels_accumulative.pdf
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Hierarchy primitives — mirror include/barycentric_hierarchy.hpp
# ----------------------------------------------------------------------

def build_adj(nv, faces):
    adj = [set() for _ in range(nv)]
    for t in faces:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
    return adj


def build_incident(nv, faces):
    inc = [[] for _ in range(nv)]
    for fi in range(len(faces)):
        t = faces[fi]
        inc[int(t[0])].append(fi); inc[int(t[1])].append(fi); inc[int(t[2])].append(fi)
    return inc


def ordered_link(v, inc_faces, faces):
    """Cyclic 1-ring of v via winding order; None for boundary/non-manifold."""
    nxt = {}
    for fi in inc_faces:
        t = faces[fi]
        if   t[0] == v: n1, n2 = int(t[1]), int(t[2])
        elif t[1] == v: n1, n2 = int(t[2]), int(t[0])
        else:           n1, n2 = int(t[0]), int(t[1])
        if n1 in nxt:
            return None                       # non-manifold edge
        nxt[n1] = n2
    if not nxt:
        return None
    start = next(iter(nxt)); ring = [start]; cur = start
    while True:
        if cur not in nxt:
            return None                       # boundary
        nx = nxt[cur]
        if nx == start:
            break
        ring.append(nx); cur = nx
        if len(ring) > len(nxt):
            return None
    return ring if len(ring) == len(nxt) else None


def greedy_mis(nv, adj):
    """Greedy maximal independent set, low-valence first."""
    order = sorted(range(nv), key=lambda x: len(adj[x]))
    blocked = np.zeros(nv, bool)
    removed = np.zeros(nv, bool)
    for v in order:
        if not blocked[v]:
            removed[v] = True; blocked[v] = True
            for u in adj[v]:
                blocked[u] = True
    return removed


def coarse_triangulation(faces, nv, inc, removed, old_to_new):
    coarse = []
    for t in faces:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        if removed[a] or removed[b] or removed[c]:
            continue
        coarse.append((old_to_new[a], old_to_new[b], old_to_new[c]))
    for v in range(nv):
        if not removed[v]:
            continue
        ring = ordered_link(v, inc[v], faces)
        if ring is None or len(ring) < 3:
            continue
        for i in range(1, len(ring) - 1):
            coarse.append((old_to_new[ring[0]], old_to_new[ring[i]], old_to_new[ring[i+1]]))
    return np.asarray(coarse, dtype=np.int64).reshape(-1, 3)


def level_assignment(coords, faces, num_levels):
    """Return level_of[g] for every global vertex (0=base, L=removed at level L)."""
    N = len(coords)
    cur_faces = faces.astype(np.int64).copy()
    cur_nv = N
    cur_to_global = np.arange(N)
    level_of = np.full(N, 0, dtype=np.int32)   # default base; overwritten if removed

    for level in range(1, num_levels + 1):
        if cur_nv < 4 or len(cur_faces) < 1:
            break
        adj = build_adj(cur_nv, cur_faces)
        removed = greedy_mis(cur_nv, adj)
        inc = build_incident(cur_nv, cur_faces)
        # prune un-ringed removals
        for v in range(cur_nv):
            if removed[v] and (ordered_link(v, inc[v], cur_faces) is None or
                               len(ordered_link(v, inc[v], cur_faces)) < 3):
                removed[v] = False
        n_removed = int(removed.sum())
        n_coarse = cur_nv - n_removed
        old_to_new = np.full(cur_nv, -1, np.int64)
        c = 0
        for i in range(cur_nv):
            if not removed[i]:
                old_to_new[i] = c; c += 1
        # tag removed globals with this level
        for v in range(cur_nv):
            if removed[v]:
                level_of[cur_to_global[v]] = level
        print(f"  level {level}: {cur_nv} -> {n_coarse}  ({n_removed} removed)")
        # coarsen
        next_global = np.empty(n_coarse, np.int64)
        for i in range(cur_nv):
            if not removed[i]:
                next_global[old_to_new[i]] = cur_to_global[i]
        cur_faces = coarse_triangulation(cur_faces, cur_nv, inc, removed, old_to_new)
        cur_to_global = next_global
        cur_nv = n_coarse

    print(f"  base survivors: {cur_nv}")
    return level_of


# ----------------------------------------------------------------------

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/Ocean"
    num_levels = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    coords = np.fromfile(os.path.join(data_dir, "coords.dat"),
                         dtype=np.float32).reshape(-1, 3).astype(np.float64)
    faces = np.fromfile(os.path.join(data_dir, "triangulation.dat"),
                        dtype=np.int32).reshape(-1, 3)
    temp = np.fromfile(os.path.join(data_dir, "temperature.dat.0"), dtype=np.float32)
    N = len(coords)
    print(f"vertices={N}  triangles={len(faces)}  levels={num_levels}")

    level_of = level_assignment(coords, faces, num_levels)

    # 3-D sphere -> lon/lat for a world-map view.
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r = np.linalg.norm(coords, axis=1)
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))

    # Accumulation order: base (0), then coarsest removed level down to finest.
    used = sorted(set(level_of.tolist()))
    Lmax = max(used)
    accum_levels = [0] + list(range(Lmax, 0, -1))   # [0, Lmax, Lmax-1, ..., 1]

    vmin, vmax = float(temp.min()), float(temp.max())
    nframes = len(accum_levels)
    ncol = min(3, nframes)
    nrow = (nframes + ncol - 1) // ncol
    # Larger canvas: ~11 in per panel column, ~6 in per row.
    fig, axes = plt.subplots(nrow, ncol, figsize=(11 * ncol, 6 * nrow), squeeze=False)

    included = np.zeros(N, bool)
    sc = None
    for k, lv in enumerate(accum_levels):
        included |= (level_of == lv)
        ax = axes[k // ncol][k % ncol]
        idx = np.where(included)[0]
        sc = ax.scatter(lon[idx], lat[idx], c=temp[idx], s=4.0,
                        cmap="turbo", vmin=vmin, vmax=vmax, linewidths=0,
                        rasterized=True)   # raster the points so the PDF renders
        frac = 100.0 * len(idx) / N
        if lv == 0:
            ax.set_title(f"base only  —  {len(idx):,} pts ({frac:.1f}%)", fontsize=18)
        else:
            ax.set_title(f"+ level {lv}  —  {len(idx):,} pts ({frac:.1f}%)", fontsize=18)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    for k in range(nframes, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle("Ocean temperature.dat.0 — accumulative MIS hierarchy "
                 "(coarsest base -> finest)", fontsize=24)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, label="temperature")
    cbar.set_label("temperature", fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    out = os.path.join(data_dir, "ocean_levels_accumulative.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")   # dpi controls the rasterized layer
    out_png = os.path.join(data_dir, "ocean_levels_accumulative.png")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}\nSaved -> {out_png}")


if __name__ == "__main__":
    main()
