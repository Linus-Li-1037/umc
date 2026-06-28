#ifndef _UMC_CLUSTER_HIERARCHY_HPP
#define _UMC_CLUSTER_HIERARCHY_HPP

#include <vector>
#include <map>
#include <unordered_set>
#include <utility>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstdio>

#include "barycentric_hierarchy.hpp"   // reuse HierEntry

namespace UMC {

// Robust local point spacing: median edge length of the centroid triangulation.
// Far more reliable than sqrt(bbox_area / N), which a non-uniform point set with
// empty regions overestimates.  Deterministic, so both ends compute the same value.
inline double median_edge_length(const std::vector<double>& V,
                                 const std::vector<int>&    faces)
{
    std::vector<double> len;
    len.reserve(faces.size());
    for (size_t f = 0; f + 2 < faces.size(); f += 3) {
        int a = faces[f], b = faces[f+1], c = faces[f+2];
        const int e[3][2] = {{a,b},{b,c},{c,a}};
        for (auto& pr : e) {
            double dx = V[pr[0]*3+0]-V[pr[1]*3+0];
            double dy = V[pr[0]*3+1]-V[pr[1]*3+1];
            len.push_back(std::sqrt(dx*dx + dy*dy));
        }
    }
    if (len.empty()) return 1.0;
    std::nth_element(len.begin(), len.begin() + len.size()/2, len.end());
    return std::max(len[len.size()/2], 1e-12);
}

// Grid vertex-clustering multilevel hierarchy (ratio-controlled coarsening).
//
// Unlike MIS — whose removal fraction is capped by the independent-set ratio
// (~1/3 on a triangulation) — clustering controls the ratio through the cell
// size: ~k points per cell => keep ~1/k.  With cell size doubling each level a
// 2-D point set keeps ~1/4 per level, so the base shrinks like SZ3 (N / 4^L)
// instead of (2/3)^L, and survivors sit at cell centers (quasi-regular).
//
// Each level:
//   - bucket the current survivors into a uniform grid of side h
//   - elect one representative per occupied cell (point nearest the cell mean)
//   - every other point is removed and predicted by IDW from the survivors in
//     its 3x3 cell neighborhood (its own cell always has a representative, so
//     there are no zero-predictor removed points)
//   - recurse on the representatives with h *= 2
//
// Returns the same HierEntry plan as build_barycentric_hierarchy: base
// survivors first (causal grid IDW), then removed points coarsest -> finest.
// The construction is a pure function of V, so the decompressor reproduces it.
//
// keep_ratio: target survivors / points PER LEVEL.  Each level binary-searches
//             the cell size so that (occupied cells)/n ~ keep_ratio, giving
//             direct, density-independent control (e.g. 0.5 => halve per level).
// spacing0:   robust point spacing (median triangulation edge) used only to seed
//             the search bracket; the result is independent of it.
inline std::vector<HierEntry> build_cluster_hierarchy(
    const std::vector<double>& V,
    int                        num_vertices,
    int                        num_levels,
    double                     keep_ratio,
    double                     spacing0,
    bool                       verbose     = false)
{
    using Cell = std::pair<long long, long long>;
    auto cell_of = [&](int gid, double h) -> Cell {
        return { (long long)std::floor(V[gid*3+0] / h),
                 (long long)std::floor(V[gid*3+1] / h) };
    };

    std::vector<int> cur_ids(num_vertices);
    std::iota(cur_ids.begin(), cur_ids.end(), 0);

    std::vector<std::vector<HierEntry>> level_recs;
    double h = spacing0;

    for (int level = 1; level <= num_levels; level++) {
        int n = (int)cur_ids.size();
        if (n < 8) break;

        // Count distinct occupied cells for a given cell size (packed key).
        auto occupied = [&](double hh) -> int {
            std::unordered_set<long long> occ;
            occ.reserve(n * 2);
            for (int a = 0; a < n; a++) {
                long long ci = (long long)std::floor(V[cur_ids[a]*3+0]/hh) + (1LL<<31);
                long long cj = (long long)std::floor(V[cur_ids[a]*3+1]/hh) + (1LL<<31);
                occ.insert((ci << 32) | (cj & 0xffffffffLL));
            }
            return (int)occ.size();
        };
        // Binary-search h so occupied(h)/n ~ keep_ratio (occupied is monotone
        // decreasing in h).  Geometric bisection around the spacing seed.
        double target = keep_ratio * n;
        double lo = spacing0 * 0.05, hi = spacing0 * 2000.0;
        for (int it = 0; it < 40; it++) {
            double mid = std::sqrt(lo * hi);
            if (occupied(mid) > target) lo = mid;   // too many survivors -> larger cell
            else                        hi = mid;
        }
        h = std::sqrt(lo * hi);

        std::map<Cell, std::vector<int>> cells;   // cell -> indices into cur_ids
        for (int a = 0; a < n; a++)
            cells[cell_of(cur_ids[a], h)].push_back(a);

        // Representative per cell = point nearest the cell mean (well-centered).
        std::map<Cell, int> rep;                  // cell -> representative global id
        std::vector<int>  survivors;
        std::vector<char> is_rep(n, 0);
        survivors.reserve(cells.size());
        for (auto& kv : cells) {
            const auto& mem = kv.second;
            double mx = 0, my = 0;
            for (int a : mem) { mx += V[cur_ids[a]*3+0]; my += V[cur_ids[a]*3+1]; }
            mx /= mem.size(); my /= mem.size();
            int best = mem[0]; double bestd = 1e300;
            for (int a : mem) {
                double dx = V[cur_ids[a]*3+0] - mx, dy = V[cur_ids[a]*3+1] - my;
                double d = dx*dx + dy*dy;
                if (d < bestd) { bestd = d; best = a; }
            }
            rep[kv.first] = cur_ids[best];
            is_rep[best]  = 1;
            survivors.push_back(cur_ids[best]);
        }

        // Predict each removed point with BARYCENTRIC interpolation: gather the
        // survivors in its 3x3 cell neighborhood, try every triple, and keep the
        // triangle that best contains the point (min outside-penalty).  Same
        // predictor as the MIS path — only the survivor selection differs.
        // (<3 nearby survivors: IDW fallback over whatever exists.)
        std::vector<HierEntry> recs;
        recs.reserve(n - (int)survivors.size());
        for (auto& kv : cells) {
            long long ci = kv.first.first, cj = kv.first.second;
            for (int a : kv.second) {
                if (is_rep[a]) continue;
                int g = cur_ids[a];
                HierEntry e; e.global_id = g; e.level = level;

                std::vector<int> cand;
                for (long long di = -1; di <= 1; di++)
                    for (long long dj = -1; dj <= 1; dj++) {
                        auto it = rep.find({ci+di, cj+dj});
                        if (it != rep.end()) cand.push_back(it->second);
                    }

                auto pv = detail::get_pos3(V.data(), 3, g);
                if (cand.size() >= 3) {
                    int bi = -1, bj = -1, bk = -1;
                    double bpen = 1e300;
                    std::array<double,3> bb{};
                    for (int i = 0; i < (int)cand.size(); i++)
                        for (int j = i+1; j < (int)cand.size(); j++)
                            for (int k = j+1; k < (int)cand.size(); k++) {
                                auto b3 = detail::barycentric(pv,
                                    detail::get_pos3(V.data(), 3, cand[i]),
                                    detail::get_pos3(V.data(), 3, cand[j]),
                                    detail::get_pos3(V.data(), 3, cand[k]));
                                double pen = -std::min({b3[0], b3[1], b3[2], 0.0});
                                if (pen < bpen) { bpen=pen; bi=i; bj=j; bk=k; bb=b3; }
                            }
                    e.pred_ids = { cand[bi], cand[bj], cand[bk] };
                    e.weights  = { bb[0], bb[1], bb[2] };
                } else {
                    double wsum = 0;
                    for (int s : cand) {
                        double dx = V[g*3+0]-V[s*3+0], dy = V[g*3+1]-V[s*3+1];
                        double dist = std::sqrt(dx*dx + dy*dy);
                        if (dist < 1e-14) dist = 1e-14;
                        double w = 1.0 / dist;
                        e.pred_ids.push_back(s); e.weights.push_back(w); wsum += w;
                    }
                    for (double& w : e.weights) w /= wsum;
                }
                recs.push_back(std::move(e));
            }
        }
        if (verbose)
            std::printf("  cluster level %d: %d -> %zu survivors (keep %.1f%%), h=%.4g\n",
                        level, n, survivors.size(), 100.0*survivors.size()/n, h);

        level_recs.push_back(std::move(recs));
        cur_ids = std::move(survivors);
    }

    // Base: causal grid IDW (each base point predicted from earlier-index base
    // points in its 3x3 neighborhood).  Base is small here, so its cost is minor.
    std::vector<HierEntry> plan;
    plan.reserve(num_vertices);
    {
        int nb = (int)cur_ids.size();
        std::map<Cell, std::vector<int>> bcells;
        for (int a = 0; a < nb; a++)
            bcells[cell_of(cur_ids[a], h)].push_back(a);

        size_t zero = 0;
        for (int a = 0; a < nb; a++) {
            int g = cur_ids[a];
            double px = V[g*3+0], py = V[g*3+1];
            Cell c = cell_of(g, h);
            HierEntry e; e.global_id = g; e.level = 0;
            double wsum = 0;
            for (long long di = -1; di <= 1; di++)
                for (long long dj = -1; dj <= 1; dj++) {
                    auto it = bcells.find({c.first+di, c.second+dj});
                    if (it == bcells.end()) continue;
                    for (int b : it->second) {
                        if (b >= a) continue;            // causal: earlier only
                        int s = cur_ids[b];
                        double dx = px - V[s*3+0], dy = py - V[s*3+1];
                        double dist = std::sqrt(dx*dx + dy*dy);
                        if (dist < 1e-14) dist = 1e-14;
                        double w = 1.0 / dist;
                        e.pred_ids.push_back(s);
                        e.weights.push_back(w);
                        wsum += w;
                    }
                }
            for (double& w : e.weights) w /= wsum;
            if (e.pred_ids.empty()) zero++;
            plan.push_back(std::move(e));               // empty -> pred 0
        }
        if (verbose)
            std::printf("  cluster base: %d points (zero-neighbor: %zu)\n", nb, zero);
    }

    for (int L = (int)level_recs.size() - 1; L >= 0; L--)
        for (auto& e : level_recs[L]) plan.push_back(std::move(e));

    return plan;
}

} // namespace UMC
#endif
