"""
Visualize the EDGE-COLLAPSE hierarchy of include/edge_collapse_hierarchy.hpp on
the Ocean temperature field, ACCUMULATIVELY from coarsest (base) to finest.

It replicates the C++ edge-collapse decimation (shortest-edge-first, link-
condition-checked, half-edge collapse that KEEPS the lower-index endpoint),
collapsing each level until alive_count <= keep_ratio * before, and tags every
vertex with the level at which it was collapsed (0 = base survivor, L =
collapsed at level L; L=1 is the finest/first level, L=num_levels the coarsest).

(Only the decimation is replicated — the barycentric prediction is irrelevant
for this picture.)  Each panel shows the points present so far, colored by their
true temperature, on a lon/lat map — the same accumulative coarse->finest view
as visualize_ocean_levels.py, but driven by edge collapse instead of MIS.

Usage:
    python visualize_ocean_levels_edge_collapse.py <data_dir> [num_levels=6] [keep_ratio=0.5]

Reads : <data_dir>/coords.dat (float32 N*3), triangulation.dat (int32 T*3),
        temperature.dat.0 (float32 N)
Writes: <data_dir>/ocean_levels_edge_collapse_accumulative.pdf (+ .png)
"""

import sys
import os
import math
import heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def edge_collapse_levels(coords, faces, num_levels, keep_ratio):
    """Return level_of[g] for every vertex (0 = base survivor, L = collapsed at L),
    mirroring build_edge_collapse_hierarchy in edge_collapse_hierarchy.hpp."""
    N = len(coords)
    P = coords
    faces = [list(map(int, t)) for t in faces]     # mutable rows
    nfaces = len(faces)
    face_dead = bytearray(nfaces)
    vinc = [set() for _ in range(N)]               # vertex -> incident (alive) faces
    for fi, (a, b, c) in enumerate(faces):
        vinc[a].add(fi); vinc[b].add(fi); vinc[c].add(fi)
    alive = bytearray([1]) * N
    level_of = np.zeros(N, np.int32)

    def dist(a, b):
        d = P[a] - P[b]
        return float(math.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]))

    def neighbors(v):
        s = set()
        for fi in vinc[v]:
            for w in faces[fi]:
                if w != v:
                    s.add(w)
        return s

    def adjacent(a, b):
        for fi in vinc[a]:
            if b in faces[fi]:
                return True
        return False

    alive_count = N
    for level in range(1, num_levels + 1):
        before = alive_count
        target = max(4, math.ceil(before * keep_ratio))
        if before <= target:
            break

        # Min-heap of current alive edges by length (lazy: skip stale on pop).
        h = []
        for fi in range(nfaces):
            if face_dead[fi]:
                continue
            f = faces[fi]
            for j in range(3):
                a, b = f[j], f[(j + 1) % 3]
                if a > b:
                    a, b = b, a
                h.append((dist(a, b), a, b))
        heapq.heapify(h)

        collapsed = 0
        while alive_count > target and h:
            length, a, b = heapq.heappop(h)
            if not alive[a] or not alive[b] or not adjacent(a, b):
                continue
            keep, rem = (a, b) if a < b else (b, a)

            # Link condition (closed manifold): edge in exactly 2 faces, and the
            # endpoints' common neighbors == those faces' opposite vertices.
            fa = [fi for fi in vinc[keep] if rem in faces[fi]]
            if len(fa) != 2:
                continue
            opp = set()
            for fi in fa:
                for w in faces[fi]:
                    if w != keep and w != rem:
                        opp.add(w)
            if (neighbors(keep) & neighbors(rem)) != opp:
                continue                             # would create non-manifold/fold

            # Collapse rem -> keep.
            for fi in list(vinc[rem]):
                f = faces[fi]
                if keep in f:                        # shared face: delete
                    face_dead[fi] = 1
                    for w in f:
                        vinc[w].discard(fi)
                else:                                # re-point rem -> keep
                    faces[fi] = [keep if w == rem else w for w in f]
                    vinc[keep].add(fi)
            vinc[rem].clear()
            alive[rem] = 0
            level_of[rem] = level
            alive_count -= 1
            collapsed += 1

            for n in neighbors(keep):
                lo, hi = (keep, n) if keep < n else (n, keep)
                heapq.heappush(h, (dist(lo, hi), lo, hi))

        print(f"  EC level {level}: {before} -> {alive_count} "
              f"({collapsed} collapsed, keep {100.0*alive_count/before:.1f}%)")

    print(f"  base survivors: {alive_count}")
    return level_of


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/Ocean"
    num_levels = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    keep_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    coords = np.fromfile(os.path.join(data_dir, "coords.dat"),
                         dtype=np.float32).reshape(-1, 3).astype(np.float64)
    faces = np.fromfile(os.path.join(data_dir, "triangulation.dat"),
                        dtype=np.int32).reshape(-1, 3)
    temp = np.fromfile(os.path.join(data_dir, "temperature.dat.0"), dtype=np.float32)
    N = len(coords)
    print(f"vertices={N}  triangles={len(faces)}  levels={num_levels}  keep={keep_ratio}")

    level_of = edge_collapse_levels(coords, faces, num_levels, keep_ratio)

    # 3-D sphere -> lon/lat for a world-map view.
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r = np.linalg.norm(coords, axis=1)
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(np.clip(z / r, -1, 1)))

    used = sorted(set(level_of.tolist()))
    Lmax = max(used)
    accum_levels = [0] + list(range(Lmax, 0, -1))   # base, then coarsest -> finest

    vmin, vmax = float(temp.min()), float(temp.max())
    nframes = len(accum_levels)
    ncol = min(3, nframes)
    nrow = (nframes + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11 * ncol, 6 * nrow), squeeze=False)

    included = np.zeros(N, bool)
    sc = None
    for k, lv in enumerate(accum_levels):
        included |= (level_of == lv)
        ax = axes[k // ncol][k % ncol]
        idx = np.where(included)[0]
        sc = ax.scatter(lon[idx], lat[idx], c=temp[idx], s=4.0,
                        cmap="turbo", vmin=vmin, vmax=vmax, linewidths=0,
                        rasterized=True)
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

    fig.suptitle("Ocean temperature.dat.0 — accumulative EDGE-COLLAPSE hierarchy "
                 f"(keep {keep_ratio:g}/level, coarsest base -> finest)", fontsize=24)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label("temperature", fontsize=16)
    cbar.ax.tick_params(labelsize=13)

    out = os.path.join(data_dir, "ocean_levels_edge_collapse_accumulative.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    out_png = os.path.join(data_dir, "ocean_levels_edge_collapse_accumulative.png")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}\nSaved -> {out_png}")


if __name__ == "__main__":
    main()
