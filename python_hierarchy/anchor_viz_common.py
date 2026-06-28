"""
Shared helpers for the barycentric-anchor visualizations (MIS and edge-collapse).

plot_frames() renders the accumulative coarse->fine panels for a small lon/lat
region: every level has its OWN fixed color (consistent across all frames), the
level just added in a frame is drawn larger/on top, and each current-level
point's barycentric anchor triangle (its 3 coarser anchors) is drawn in GRAY.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def barycentric(p, a, b, c):
    """Barycentric weights of p projected into the plane of triangle (a,b,c)."""
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-14:
        return np.array([1.0, 0.0, 0.0])
    beta  = (d11 * d20 - d01 * d21) / denom
    gamma = (d00 * d21 - d01 * d20) / denom
    return np.array([1.0 - beta - gamma, beta, gamma])


# Fixed, distinct color per level (assigned in accumulation order base, Lmax, ..., 1).
_PALETTE = ['#111111', '#d62728', '#1f77b4', '#2ca02c',
            '#9467bd', '#ff7f0e', '#8c564b', '#17becf']


def plot_frames(coords, level_of, anchors_of, lon_c, lat_c, span, title, out_pdf):
    N = len(coords)
    r = np.linalg.norm(coords, axis=1)
    lon = np.degrees(np.arctan2(coords[:, 1], coords[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(coords[:, 2] / r, -1, 1)))
    lo_lon, hi_lon = lon_c - span/2, lon_c + span/2
    lo_lat, hi_lat = lat_c - span/2, lat_c + span/2
    in_box = (lon >= lo_lon) & (lon <= hi_lon) & (lat >= lo_lat) & (lat <= hi_lat)
    print(f"  points in region: {int(in_box.sum())}")

    Lmax = int(level_of.max())
    accum = [0] + list(range(Lmax, 0, -1))         # base, coarsest -> finest
    level_color = {lv: _PALETTE[i % len(_PALETTE)] for i, lv in enumerate(accum)}
    lname = lambda lv: "base" if lv == 0 else f"level {lv}"

    nframes = len(accum)
    ncol = min(3, nframes)
    nrow = (nframes + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 7 * nrow), squeeze=False)

    included = np.zeros(N, bool)
    for k, lv in enumerate(accum):
        included |= (level_of == lv)
        ax = axes[k // ncol][k % ncol]

        # Gray anchor triangles for the level just added.
        ntri = 0
        if lv != 0:
            for g in np.where((level_of == lv) & in_box)[0]:
                anc = anchors_of.get(int(g))
                if anc is None:
                    continue
                xy = np.column_stack([lon[list(anc)], lat[list(anc)]])
                ax.add_patch(Polygon(xy, closed=True, fill=False,
                                     edgecolor="0.55", lw=0.9, alpha=0.7, zorder=1))
                ntri += 1

        # Points: each level keeps its fixed color; the current level is emphasized.
        for plv in accum[:k + 1]:
            m = (level_of == plv) & in_box
            if not m.any():
                continue
            cur = (plv == lv)
            ax.scatter(lon[m], lat[m], s=(64 if cur else 26), c=level_color[plv],
                       alpha=0.95, zorder=(6 if cur else 4),
                       linewidths=(0.7 if cur else 0),
                       edgecolors=("white" if cur else "none"),
                       label=lname(plv))

        ax.set_title(f"+ {lname(lv)}" + ("" if lv == 0 else f"   ({ntri} anchor triangles)"),
                     fontsize=15)
        ax.set_xlim(lo_lon, hi_lon); ax.set_ylim(lo_lat, hi_lat)
        ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    for k in range(nframes, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle(title, fontsize=19)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    out_png = out_pdf[:-4] + ".png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_pdf}\nSaved -> {out_png}")
