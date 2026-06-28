"""
Edge-collapse counterpart of visualize_barycentric_anchors.py.

Replicates the EDGE-COLLAPSE decimation of include/edge_collapse_hierarchy.hpp
(shortest-edge-first, link-condition, half-edge collapse keeping the lower
index), and records each collapsed vertex's barycentric anchors — the best
triangle in the coarse 1-ring of the survivor it merged into (exactly the C++
prediction).  Then renders the same per-level-colored, gray-triangle,
accumulative coarse->fine zoom via the shared plot_frames.

Usage:
    python visualize_barycentric_anchors_edge_collapse.py <data_dir>
            [num_levels=4] [lon_c=-140] [lat_c=0] [span_deg=4] [keep_ratio=0.5]

Writes: <data_dir>/barycentric_anchors_edge_collapse_accumulative.pdf (+ .png)
"""

import sys
import os
import math
import heapq
import numpy as np
from anchor_viz_common import barycentric, plot_frames


def edge_collapse_with_anchors(coords, faces, num_levels, keep_ratio):
    """Mirror build_edge_collapse_hierarchy: returns (level_of, anchors_of)."""
    N = len(coords)
    P = coords
    faces = [list(map(int, t)) for t in faces]
    nfaces = len(faces)
    face_dead = bytearray(nfaces)
    vinc = [set() for _ in range(N)]
    for fi, (a, b, c) in enumerate(faces):
        vinc[a].add(fi); vinc[b].add(fi); vinc[c].add(fi)
    alive = bytearray([1]) * N
    level_of = np.zeros(N, np.int32)
    merge_to = np.full(N, -1, np.int64)
    anchors_of = {}

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

        removed_this = []
        while alive_count > target and h:
            _, a, b = heapq.heappop(h)
            if not alive[a] or not alive[b] or not adjacent(a, b):
                continue
            keep, rem = (a, b) if a < b else (b, a)

            fa = [fi for fi in vinc[keep] if rem in faces[fi]]
            if len(fa) != 2:
                continue
            opp = set()
            for fi in fa:
                for w in faces[fi]:
                    if w != keep and w != rem:
                        opp.add(w)
            if (neighbors(keep) & neighbors(rem)) != opp:
                continue

            for fi in list(vinc[rem]):
                f = faces[fi]
                if keep in f:
                    face_dead[fi] = 1
                    for w in f:
                        vinc[w].discard(fi)
                else:
                    faces[fi] = [keep if w == rem else w for w in f]
                    vinc[keep].add(fi)
            vinc[rem].clear()
            alive[rem] = 0
            merge_to[rem] = keep
            level_of[rem] = level
            removed_this.append(rem)
            alive_count -= 1

            for n in neighbors(keep):
                lo, hi = (keep, n) if keep < n else (n, keep)
                heapq.heappush(h, (dist(lo, hi), lo, hi))

        # Record anchors: best triangle in the merge-target survivor's coarse 1-ring.
        for rem in removed_this:
            s = rem
            while not alive[s]:
                s = int(merge_to[s])
            pv = P[rem]
            best_pen, best_tri = np.inf, None
            for fi in vinc[s]:
                f = faces[fi]
                b3 = barycentric(pv, P[f[0]], P[f[1]], P[f[2]])
                pen = -min(b3.min(), 0.0)
                if pen < best_pen:
                    best_pen, best_tri = pen, (f[0], f[1], f[2])
            if best_tri is not None:
                anchors_of[int(rem)] = (int(best_tri[0]), int(best_tri[1]), int(best_tri[2]))

        print(f"  EC level {level}: {before} -> {alive_count} "
              f"({len(removed_this)} collapsed, keep {100.0*alive_count/before:.1f}%)")

    print(f"  base survivors: {alive_count}")
    return level_of, anchors_of


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/Ocean"
    num_levels = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    lon_c = float(sys.argv[3]) if len(sys.argv) > 3 else -140.0
    lat_c = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    span  = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
    keep_ratio = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5

    coords = np.fromfile(os.path.join(data_dir, "coords.dat"),
                         dtype=np.float32).reshape(-1, 3).astype(np.float64)
    faces = np.fromfile(os.path.join(data_dir, "triangulation.dat"),
                        dtype=np.int32).reshape(-1, 3)
    print(f"vertices={len(coords)}  triangles={len(faces)}  levels={num_levels}  "
          f"keep={keep_ratio}  region=({lon_c}+-{span/2}, {lat_c}+-{span/2})")

    level_of, anchors_of = edge_collapse_with_anchors(coords, faces, num_levels, keep_ratio)

    out = os.path.join(data_dir, "barycentric_anchors_edge_collapse_accumulative.pdf")
    plot_frames(coords, level_of, anchors_of, lon_c, lat_c, span,
                f"Barycentric anchors — EDGE-COLLAPSE hierarchy (keep {keep_ratio:g}/level; "
                "anchors = merge-target 1-ring; triangles in gray)",
                out)


if __name__ == "__main__":
    main()
