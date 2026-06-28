"""
Zoomed-in visualization of how BARYCENTRIC prediction works across the MIS
hierarchy (include/barycentric_hierarchy.hpp), accumulatively coarse -> fine.

For a small lon/lat region it shows, per accumulation frame for level L:
  - all coarser already-decoded points (base + levels > L)  in ONE color, and
  - the level-L points just added                            in ANOTHER color,
  - and for every level-L point, the triangle of its 3 barycentric ANCHORS
    (3 coarser survivors), drawn in the coarser-level color — i.e. exactly the
    3 points used to interpolate it.  A level-L point sits inside its anchor
    triangle (that is what "enclosure" means).

The anchors are the REAL ones: we replicate build_parameterization (cyclic
1-ring fan, min outside-penalty triangle) and record each removed vertex's 3
anchor global ids.

Usage:
    python visualize_barycentric_anchors.py <data_dir> [num_levels=4]
                                             [lon_c=-140] [lat_c=0] [span_deg=10]

Reads : <data_dir>/coords.dat, triangulation.dat, temperature.dat.0
Writes: <data_dir>/barycentric_anchors_accumulative.pdf (+ .png)
"""

import sys
import os
import numpy as np
from anchor_viz_common import barycentric, plot_frames


# ---- hierarchy primitives (mirror barycentric_hierarchy.hpp) -----------------

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
    nxt = {}
    for fi in inc_faces:
        t = faces[fi]
        if   t[0] == v: n1, n2 = int(t[1]), int(t[2])
        elif t[1] == v: n1, n2 = int(t[2]), int(t[0])
        else:           n1, n2 = int(t[0]), int(t[1])
        if n1 in nxt:
            return None
        nxt[n1] = n2
    if not nxt:
        return None
    start = next(iter(nxt)); ring = [start]; cur = start
    while True:
        if cur not in nxt:
            return None
        nx = nxt[cur]
        if nx == start:
            break
        ring.append(nx); cur = nx
        if len(ring) > len(nxt):
            return None
    return ring if len(ring) == len(nxt) else None


def greedy_mis(nv, adj):
    order = sorted(range(nv), key=lambda x: len(adj[x]))
    blocked = np.zeros(nv, bool); removed = np.zeros(nv, bool)
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


def hierarchy_with_anchors(coords, faces, num_levels):
    """level_of[g] (0=base, L=removed at L) and anchors_of[g] = (a0,a1,a2) global ids."""
    N = len(coords)
    cur_faces = faces.astype(np.int64).copy()
    cur_nv = N
    cur_to_global = np.arange(N)
    level_of = np.zeros(N, np.int32)
    anchors_of = {}

    for level in range(1, num_levels + 1):
        if cur_nv < 4 or len(cur_faces) < 1:
            break
        adj = build_adj(cur_nv, cur_faces)
        removed = greedy_mis(cur_nv, adj)
        inc = build_incident(cur_nv, cur_faces)
        for v in range(cur_nv):                       # prune un-ringed removals
            if removed[v]:
                ring = ordered_link(v, inc[v], cur_faces)
                if ring is None or len(ring) < 3:
                    removed[v] = False
        old_to_new = np.full(cur_nv, -1, np.int64)
        c = 0
        for i in range(cur_nv):
            if not removed[i]:
                old_to_new[i] = c; c += 1
        n_coarse = c

        # Record level + real barycentric anchors (fan, min outside-penalty).
        for v in range(cur_nv):
            if not removed[v]:
                continue
            gv = cur_to_global[v]
            level_of[gv] = level
            ring = ordered_link(v, inc[v], cur_faces)
            pv = coords[gv]
            a = coords[cur_to_global[ring[0]]]
            best_pen, best_i = np.inf, 1
            for i in range(1, len(ring) - 1):
                b3 = barycentric(pv, a, coords[cur_to_global[ring[i]]],
                                       coords[cur_to_global[ring[i+1]]])
                pen = -min(b3.min(), 0.0)
                if pen < best_pen:
                    best_pen, best_i = pen, i
            anchors_of[int(gv)] = (int(cur_to_global[ring[0]]),
                                   int(cur_to_global[ring[best_i]]),
                                   int(cur_to_global[ring[best_i+1]]))

        next_global = np.empty(n_coarse, np.int64)
        for i in range(cur_nv):
            if not removed[i]:
                next_global[old_to_new[i]] = cur_to_global[i]
        print(f"  level {level}: {cur_nv} -> {n_coarse}  ({int(removed.sum())} removed)")
        cur_faces = coarse_triangulation(cur_faces, cur_nv, inc, removed, old_to_new)
        cur_to_global = next_global
        cur_nv = n_coarse

    print(f"  base survivors: {cur_nv}")
    return level_of, anchors_of


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/Ocean"
    num_levels = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    lon_c = float(sys.argv[3]) if len(sys.argv) > 3 else -140.0
    lat_c = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    span  = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0

    coords = np.fromfile(os.path.join(data_dir, "coords.dat"),
                         dtype=np.float32).reshape(-1, 3).astype(np.float64)
    faces = np.fromfile(os.path.join(data_dir, "triangulation.dat"),
                        dtype=np.int32).reshape(-1, 3)
    N = len(coords)
    print(f"vertices={N}  triangles={len(faces)}  levels={num_levels}  "
          f"region=({lon_c}+-{span/2}, {lat_c}+-{span/2})")

    level_of, anchors_of = hierarchy_with_anchors(coords, faces, num_levels)

    out = os.path.join(data_dir, "barycentric_anchors_accumulative.pdf")
    plot_frames(coords, level_of, anchors_of, lon_c, lat_c, span,
                "Barycentric anchors — MIS hierarchy "
                "(each new point sits inside its 3 coarser anchors; triangles in gray)",
                out)


if __name__ == "__main__":
    main()
