#ifndef _UMC_EDGE_COLLAPSE_HIERARCHY_HPP
#define _UMC_EDGE_COLLAPSE_HIERARCHY_HPP

#include <vector>
#include <set>
#include <queue>
#include <tuple>
#include <array>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstdio>

#include "parameterization.hpp"        // detail::barycentric, detail::get_pos3
#include "barycentric_hierarchy.hpp"   // HierEntry
#include "reorder.hpp"                  // BPFS

namespace UMC {

// Multilevel hierarchy built by EDGE COLLAPSE instead of MIS.
//
// Motivation: MIS can only remove the independent-set fraction (~1/3 of a 2-D
// triangulation) per level, so the per-level ratio is not controllable.  Edge
// collapse removes one vertex per collapse, so we can collapse until any target
// count is reached — e.g. keep 1/4 per level (remove 75%) to mimic 2-D
// structured 4:1 downsampling.  Slower than MIS, but gives exact ratio control.
//
// Each level (shortest-edge-first, link-condition-checked, half-edge collapse
// that KEEPS the lower-index endpoint at its original position so survivors
// carry true data values):
//   - collapse edges until alive_count <= keep_ratio * alive_count_before
//   - every collapsed vertex is predicted barycentrically from the coarse 1-ring
//     of the survivor it merged into (same predictor as the MIS path)
// Base survivors use BPFS-ordered causal IDW, identical to build_barycentric_hierarchy.
//
// Works in GLOBAL vertex indices throughout (dead vertices/faces flagged), so
// HierEntry.pred_ids are global ids directly.  Pure function of (V, faces) =>
// the decompressor reproduces it.
// 
// # Slower than MIS, and require to tune keep ratio
inline std::vector<HierEntry> build_edge_collapse_hierarchy(
    const std::vector<double>& V,
    const std::vector<int>&    faces_in,
    int                        N,
    int                        num_levels,
    double                     keep_ratio)
{
    auto pos = [&](int v) { return detail::get_pos3(V.data(), 3, v); };
    auto dist = [&](int a, int b) {
        double dx = V[a*3+0]-V[b*3+0], dy = V[a*3+1]-V[b*3+1], dz = V[a*3+2]-V[b*3+2];
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    };

    // Mesh state (global indices, flagged dead as we collapse).
    std::vector<std::array<int,3>> faces;
    faces.reserve(faces_in.size() / 3);
    for (size_t f = 0; f + 2 < faces_in.size(); f += 3)
        faces.push_back({faces_in[f], faces_in[f+1], faces_in[f+2]});
    std::vector<char> face_dead(faces.size(), 0);
    std::vector<std::set<int>> vinc(N);            // vertex -> incident (alive) faces
    for (int fi = 0; fi < (int)faces.size(); fi++)
        for (int j = 0; j < 3; j++) vinc[faces[fi][j]].insert(fi);
    std::vector<char> alive(N, 1);

    std::vector<int> level_of(N, 0);
    std::vector<int> merge_to(N, -1);              // collapsed vertex -> kept vertex

    auto neighbors = [&](int v, std::set<int>& out) {
        out.clear();
        for (int fi : vinc[v])
            for (int j = 0; j < 3; j++)
                if (faces[fi][j] != v) out.insert(faces[fi][j]);
    };
    auto adjacent = [&](int a, int b) {
        for (int fi : vinc[a])
            if (faces[fi][0]==b || faces[fi][1]==b || faces[fi][2]==b) return true;
        return false;
    };

    std::vector<std::vector<HierEntry>> level_recs;
    int alive_count = N;

    for (int level = 1; level <= num_levels; level++) {
        int before = alive_count;
        int target = std::max(4, (int)std::ceil(before * keep_ratio));
        if (before <= target) break;

        // Min-heap of edges by length (lazy: stale/dead edges skipped on pop).
        std::priority_queue<std::tuple<double,int,int>,
                            std::vector<std::tuple<double,int,int>>,
                            std::greater<std::tuple<double,int,int>>> pq;
        for (int fi = 0; fi < (int)faces.size(); fi++) {
            if (face_dead[fi]) continue;
            const auto& f = faces[fi];
            for (int j = 0; j < 3; j++) {
                int a = f[j], b = f[(j+1)%3];
                if (a > b) std::swap(a, b);
                pq.push({dist(a, b), a, b});
            }
        }

        std::vector<int> removed_this_level;
        std::set<int> na, nb, fa, fb;
        while (alive_count > target && !pq.empty()) {
            auto [len, a, b] = pq.top(); pq.pop();
            if (!alive[a] || !alive[b] || !adjacent(a, b)) continue;

            int keep = std::min(a, b), rem = std::max(a, b);

            // Link condition (closed manifold): the edge is in exactly 2 faces,
            // and keep/rem share exactly those 2 faces' opposite vertices.
            fa.clear();
            for (int fi : vinc[keep])
                if (faces[fi][0]==rem || faces[fi][1]==rem || faces[fi][2]==rem) fa.insert(fi);
            if (fa.size() != 2) continue;
            std::set<int> opp;
            for (int fi : fa)
                for (int j = 0; j < 3; j++)
                    if (faces[fi][j]!=keep && faces[fi][j]!=rem) opp.insert(faces[fi][j]);
            neighbors(keep, na); neighbors(rem, nb);
            std::set<int> common;
            for (int x : na) if (nb.count(x)) common.insert(x);
            if (common != opp) continue;               // would create non-manifold/fold

            // Collapse rem -> keep.
            std::vector<int> rem_faces(vinc[rem].begin(), vinc[rem].end());
            for (int fi : rem_faces) {
                auto& f = faces[fi];
                bool edge_face = (f[0]==keep || f[1]==keep || f[2]==keep);
                if (edge_face) {                        // shared face: delete
                    face_dead[fi] = 1;
                    for (int j = 0; j < 3; j++) vinc[f[j]].erase(fi);
                } else {                                // re-point rem -> keep
                    for (int j = 0; j < 3; j++) if (f[j]==rem) f[j] = keep;
                    vinc[keep].insert(fi);
                }
            }
            vinc[rem].clear();
            alive[rem] = 0;
            merge_to[rem] = keep;
            level_of[rem] = level;
            removed_this_level.push_back(rem);
            alive_count--;

            // keep's neighbor set changed -> push its (possibly new) edges.
            // neighbors(keep, na);
            // for (int n : na) {
            //     int lo = std::min(keep, n), hi = std::max(keep, n);
            //     pq.push({dist(lo, hi), lo, hi});
            // }
        }

        // Predict each collapsed vertex barycentrically from the coarse 1-ring
        // of the survivor it merged into.
        std::vector<HierEntry> recs;
        recs.reserve(removed_this_level.size());
        for (int rem : removed_this_level) {
            int s = rem;
            while (!alive[s]) s = merge_to[s];          // resolve to a level survivor
            HierEntry e; e.global_id = rem; e.level = level;
            auto pv = pos(rem);
            double best_pen = std::numeric_limits<double>::max();
            std::array<int,3>    best_tri{};
            std::array<double,3> best_b{};
            for (int fi : vinc[s]) {
                const auto& f = faces[fi];
                auto b3 = detail::barycentric(pv, pos(f[0]), pos(f[1]), pos(f[2]));
                double pen = -std::min({ b3[0], b3[1], b3[2], 0.0 });
                if (pen < best_pen) { best_pen = pen; best_tri = f; best_b = b3; }
            }
            if (best_pen != std::numeric_limits<double>::max()) {
                e.pred_ids = { best_tri[0], best_tri[1], best_tri[2] };
                e.weights  = { best_b[0], best_b[1], best_b[2] };
            }                                            // else empty -> pred 0
            recs.push_back(std::move(e));
        }
        printf("  EC level %d: %d -> %d  (%d collapsed, keep %.1f%%)\n",
               level, before, alive_count, before - alive_count,
               100.0 * alive_count / before);
        level_recs.push_back(std::move(recs));
    }

    // ------------------------------------------------------------------
    // Base survivors: BPFS-ordered causal IDW (same as build_barycentric_hierarchy).
    // Compact the alive set to local indices for BPFS, then map back to global.
    // ------------------------------------------------------------------
    std::vector<int> base_globals;
    base_globals.reserve(alive_count);
    std::vector<int> local_of(N, -1);
    for (int v = 0; v < N; v++)
        if (alive[v]) { local_of[v] = (int)base_globals.size(); base_globals.push_back(v); }
    int nb_base = (int)base_globals.size();

    std::vector<std::set<int32_t>> base_adj(nb_base);
    for (int fi = 0; fi < (int)faces.size(); fi++) {
        if (face_dead[fi]) continue;
        int a = local_of[faces[fi][0]], b = local_of[faces[fi][1]], c = local_of[faces[fi][2]];
        if (a < 0 || b < 0 || c < 0) continue;          // touches a removed vertex
        base_adj[a].insert(b); base_adj[a].insert(c);
        base_adj[b].insert(a); base_adj[b].insert(c);
        base_adj[c].insert(a); base_adj[c].insert(b);
    }

    std::vector<HierEntry> plan;
    plan.reserve(N);
    if (nb_base > 0) {
        auto order = BPFS(base_adj);
        std::vector<int> rank_to_node(nb_base);
        for (int c = 0; c < nb_base; c++) rank_to_node[order[c]] = c;
        for (int r = 0; r < nb_base; r++) {
            int c = rank_to_node[r], g = base_globals[c];
            HierEntry e; e.global_id = g; e.level = 0;
            double wsum = 0.0;
            for (int32_t u : base_adj[c]) {
                if (order[u] >= order[c]) continue;     // not yet decoded
                int gu = base_globals[u];
                double w = 1.0 / std::max(dist(g, gu), 1e-14);
                e.pred_ids.push_back(gu);
                e.weights.push_back(w);
                wsum += w;
            }
            for (double& w : e.weights) w /= wsum;
            plan.push_back(std::move(e));
        }
    }
    for (int L = (int)level_recs.size() - 1; L >= 0; L--)
        for (auto& e : level_recs[L]) plan.push_back(std::move(e));

    return plan;
}


// Forest-collapse (PFS-style) variant of build_edge_collapse_hierarchy.
//
// Same mesh surgery / predictor / base layer, but a different per-level COLLAPSE
// SCHEDULE: instead of popping and collapsing edges one at a time down to a
// keep_ratio, each level greedily selects a whole BATCH of mutually independent
// collapsible edges and collapses them together (one level).  An edge joins the
// batch iff (a) it passes the link condition and (b) its closed 1-ring region
// {keep,rem} ∪ N(keep) ∪ N(rem) is disjoint from every already-selected edge's
// region — i.e. the selected edges form an edge independent set with
// non-overlapping influence, so the whole batch collapses safely in any order.
// No ratio knob; the level boundary is "one maximal independent batch".
//
// (NOT implemented, per spec: forest/simple-polygon connectivity ENCODING — the
// mesh is fixed and deterministically replayed, so no edge bitmaps / boundary
// stitching / topological reversibility tests are needed; only the simpler
// "1-ring disjoint + link condition" batch test.)
//
// # Not an authetic implementation
inline std::vector<HierEntry> build_forest_collapse_hierarchy(
    const std::vector<double>& V,
    const std::vector<int>&    faces_in,
    int                        N,
    int                        num_levels)
{
    auto pos = [&](int v) { return detail::get_pos3(V.data(), 3, v); };
    auto dist = [&](int a, int b) {
        double dx = V[a*3+0]-V[b*3+0], dy = V[a*3+1]-V[b*3+1], dz = V[a*3+2]-V[b*3+2];
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    };

    std::vector<std::array<int,3>> faces;
    faces.reserve(faces_in.size() / 3);
    for (size_t f = 0; f + 2 < faces_in.size(); f += 3)
        faces.push_back({faces_in[f], faces_in[f+1], faces_in[f+2]});
    std::vector<char> face_dead(faces.size(), 0);
    std::vector<std::set<int>> vinc(N);
    for (int fi = 0; fi < (int)faces.size(); fi++)
        for (int j = 0; j < 3; j++) vinc[faces[fi][j]].insert(fi);
    std::vector<char> alive(N, 1);
    std::vector<int>  level_of(N, 0);
    std::vector<int>  merge_to(N, -1);

    auto neighbors = [&](int v, std::set<int>& out) {
        out.clear();
        for (int fi : vinc[v])
            for (int j = 0; j < 3; j++)
                if (faces[fi][j] != v) out.insert(faces[fi][j]);
    };
    auto adjacent = [&](int a, int b) {
        for (int fi : vinc[a])
            if (faces[fi][0]==b || faces[fi][1]==b || faces[fi][2]==b) return true;
        return false;
    };

    std::vector<std::vector<HierEntry>> level_recs;
    int alive_count = N;

    for (int level = 1; level <= num_levels; level++) {
        if (alive_count < 4) break;

        // All current alive edges, shortest first.
        std::priority_queue<std::tuple<double,int,int>,
                            std::vector<std::tuple<double,int,int>>,
                            std::greater<std::tuple<double,int,int>>> pq;
        for (int fi = 0; fi < (int)faces.size(); fi++) {
            if (face_dead[fi]) continue;
            const auto& f = faces[fi];
            for (int j = 0; j < 3; j++) {
                int a = f[j], b = f[(j+1)%3];
                if (a > b) std::swap(a, b);
                pq.push({dist(a, b), a, b});
            }
        }

        // Greedily select a batch of edges with disjoint 1-ring regions.
        std::vector<char> occupied(N, 0);
        std::vector<std::pair<int,int>> batch;   // (keep, rem)
        std::set<int> na, nb;
        while (!pq.empty()) {
            auto [len, a, b] = pq.top(); pq.pop();
            if (!alive[a] || !alive[b] || !adjacent(a, b)) continue;
            int keep = std::min(a, b), rem = std::max(a, b);
            if (occupied[keep] || occupied[rem]) continue;

            neighbors(keep, na); neighbors(rem, nb);
            bool conflict = false;
            for (int x : na) if (occupied[x]) { conflict = true; break; }
            if (!conflict) for (int x : nb) if (occupied[x]) { conflict = true; break; }
            if (conflict) continue;

            // Link condition (closed manifold): edge in exactly 2 faces, and
            // keep/rem's common neighbors == those faces' opposite vertices.
            std::set<int> fa;
            for (int fi : vinc[keep])
                if (faces[fi][0]==rem || faces[fi][1]==rem || faces[fi][2]==rem) fa.insert(fi);
            if (fa.size() != 2) continue;
            std::set<int> opp;
            for (int fi : fa)
                for (int j = 0; j < 3; j++)
                    if (faces[fi][j]!=keep && faces[fi][j]!=rem) opp.insert(faces[fi][j]);
            std::set<int> common;
            for (int x : na) if (nb.count(x)) common.insert(x);
            if (common != opp) continue;

            // Accept: reserve its whole 1-ring region.
            occupied[keep] = 1; occupied[rem] = 1;
            for (int x : na) occupied[x] = 1;
            for (int x : nb) occupied[x] = 1;
            batch.push_back({keep, rem});
        }
        if (batch.empty()) break;

        // Collapse the whole batch (regions disjoint -> order-independent, safe).
        std::vector<int> removed_this;
        removed_this.reserve(batch.size());
        for (auto& kr : batch) {
            int keep = kr.first, rem = kr.second;
            for (int fi : std::vector<int>(vinc[rem].begin(), vinc[rem].end())) {
                auto& f = faces[fi];
                bool edge_face = (f[0]==keep || f[1]==keep || f[2]==keep);
                if (edge_face) {
                    face_dead[fi] = 1;
                    for (int j = 0; j < 3; j++) vinc[f[j]].erase(fi);
                } else {
                    for (int j = 0; j < 3; j++) if (f[j]==rem) f[j] = keep;
                    vinc[keep].insert(fi);
                }
            }
            vinc[rem].clear();
            alive[rem] = 0;
            merge_to[rem] = keep;
            level_of[rem] = level;
            removed_this.push_back(rem);
            alive_count--;
        }

        // Predict each collapsed vertex from its merge-target's coarse 1-ring.
        std::vector<HierEntry> recs;
        recs.reserve(removed_this.size());
        for (int rem : removed_this) {
            int s = rem;
            while (!alive[s]) s = merge_to[s];
            HierEntry e; e.global_id = rem; e.level = level;
            auto pv = pos(rem);
            double best_pen = std::numeric_limits<double>::max();
            std::array<int,3>    best_tri{};
            std::array<double,3> best_b{};
            for (int fi : vinc[s]) {
                const auto& f = faces[fi];
                auto b3 = detail::barycentric(pv, pos(f[0]), pos(f[1]), pos(f[2]));
                double pen = -std::min({ b3[0], b3[1], b3[2], 0.0 });
                if (pen < best_pen) { best_pen = pen; best_tri = f; best_b = b3; }
            }
            if (best_pen != std::numeric_limits<double>::max()) {
                e.pred_ids = { best_tri[0], best_tri[1], best_tri[2] };
                e.weights  = { best_b[0], best_b[1], best_b[2] };
            }
            recs.push_back(std::move(e));
        }
        printf("  FC level %d: %d collapsed, alive %d (keep %.1f%%)\n",
               level, (int)batch.size(), alive_count,
               100.0 * alive_count / (alive_count + (int)batch.size()));
        level_recs.push_back(std::move(recs));
    }

    // Base survivors: BPFS-ordered causal IDW (identical to the other hierarchies).
    std::vector<int> base_globals;
    base_globals.reserve(alive_count);
    std::vector<int> local_of(N, -1);
    for (int v = 0; v < N; v++)
        if (alive[v]) { local_of[v] = (int)base_globals.size(); base_globals.push_back(v); }
    int nb_base = (int)base_globals.size();

    std::vector<std::set<int32_t>> base_adj(nb_base);
    for (int fi = 0; fi < (int)faces.size(); fi++) {
        if (face_dead[fi]) continue;
        int a = local_of[faces[fi][0]], b = local_of[faces[fi][1]], c = local_of[faces[fi][2]];
        if (a < 0 || b < 0 || c < 0) continue;
        base_adj[a].insert(b); base_adj[a].insert(c);
        base_adj[b].insert(a); base_adj[b].insert(c);
        base_adj[c].insert(a); base_adj[c].insert(b);
    }

    std::vector<HierEntry> plan;
    plan.reserve(N);
    if (nb_base > 0) {
        auto order = BPFS(base_adj);
        std::vector<int> rank_to_node(nb_base);
        for (int c = 0; c < nb_base; c++) rank_to_node[order[c]] = c;
        for (int r = 0; r < nb_base; r++) {
            int c = rank_to_node[r], g = base_globals[c];
            HierEntry e; e.global_id = g; e.level = 0;
            double wsum = 0.0;
            for (int32_t u : base_adj[c]) {
                if (order[u] >= order[c]) continue;
                int gu = base_globals[u];
                double w = 1.0 / std::max(dist(g, gu), 1e-14);
                e.pred_ids.push_back(gu);
                e.weights.push_back(w);
                wsum += w;
            }
            for (double& w : e.weights) w /= wsum;
            plan.push_back(std::move(e));
        }
    }
    for (int L = (int)level_recs.size() - 1; L >= 0; L--)
        for (auto& e : level_recs[L]) plan.push_back(std::move(e));

    return plan;
}


// PFS-faithful (Taubin et al. Progressive Forest Split) level division — a
// CONTROLLED-EXPERIMENT variant of build_forest_collapse_hierarchy.
//
// Difference is ONLY the per-level batch semantics.  build_forest_collapse picks
// edges whose full 1-rings are globally disjoint -> every component degenerates
// to a single edge (tiny per-level removal).  PFS instead lets a component be a
// real multi-edge TREE of collapse edges (a tree in the mesh contracts to a
// single vertex = a simple polygon), so each level removes a large fraction
// (paper: ~T/2 triangles).  We borrow only PFS's forest-collapse TOPOLOGY rules
// to decide which edges form one level; we do NOT implement any PFS connectivity
// ENCODING (forest bitmaps / simple-polygon 2-bit codes / boundary stitching /
// reversible reconstruction) — the mesh is fixed and deterministically replayed.
//
// Per level, greedily grow a forest of collapse edges with:
//   - link condition per edge (manifold single collapse), checked at selection
//     AND re-checked at execution (sequential collapses can invalidate it);
//   - acyclic + vertex-disjoint trees: an edge is accepted only if at most one
//     endpoint already belongs to a tree (so it extends that tree as a leaf or
//     starts a new one) — this forbids both same-tree cycles (condition A) and
//     cross-tree merges (condition B / no shared vertices between components);
//   - triangle-exclusive: the edge's two collapsed triangles must be unused, so
//     distinct trees never share a triangle.
// Each tree is then contracted toward its root (collapse its edges, resolving
// the survivor through merge_to), removing ~all the tree's vertices.
//
// Predictor, mesh surgery, base layer (BPFS+IDW), and plan assembly are IDENTICAL
// to the other hierarchies (controlled experiment: only the level division changes).
//
// # Super slow
inline std::vector<HierEntry> build_pfs_collapse_hierarchy(
    const std::vector<double>& V,
    const std::vector<int>&    faces_in,
    int                        N,
    int                        num_levels)
{
    auto pos = [&](int v) { return detail::get_pos3(V.data(), 3, v); };
    auto dist = [&](int a, int b) {
        double dx = V[a*3+0]-V[b*3+0], dy = V[a*3+1]-V[b*3+1], dz = V[a*3+2]-V[b*3+2];
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    };

    std::vector<std::array<int,3>> faces;
    faces.reserve(faces_in.size() / 3);
    for (size_t f = 0; f + 2 < faces_in.size(); f += 3)
        faces.push_back({faces_in[f], faces_in[f+1], faces_in[f+2]});
    std::vector<char> face_dead(faces.size(), 0);
    std::vector<std::set<int>> vinc(N);
    for (int fi = 0; fi < (int)faces.size(); fi++)
        for (int j = 0; j < 3; j++) vinc[faces[fi][j]].insert(fi);
    std::vector<char> alive(N, 1);
    std::vector<int>  level_of(N, 0);
    std::vector<int>  merge_to(N, -1);

    auto neighbors = [&](int v, std::set<int>& out) {
        out.clear();
        for (int fi : vinc[v])
            for (int j = 0; j < 3; j++)
                if (faces[fi][j] != v) out.insert(faces[fi][j]);
    };
    auto adjacent = [&](int a, int b) {
        for (int fi : vinc[a])
            if (faces[fi][0]==b || faces[fi][1]==b || faces[fi][2]==b) return true;
        return false;
    };
    // Faces incident to both keep and rem (the 2 edge triangles); also returns
    // their opposite vertices.  Used for the link condition.
    auto edge_faces = [&](int keep, int rem, std::array<int,2>& ef, std::set<int>& opp) {
        int cnt = 0; opp.clear();
        for (int fi : vinc[keep]) {
            const auto& f = faces[fi];
            if (f[0]==rem || f[1]==rem || f[2]==rem) {
                if (cnt < 2) ef[cnt] = fi;
                cnt++;
                for (int j = 0; j < 3; j++)
                    if (f[j]!=keep && f[j]!=rem) opp.insert(f[j]);
            }
        }
        return cnt;
    };
    std::set<int> na, nb;
    auto link_ok = [&](int keep, int rem) {
        std::array<int,2> ef; std::set<int> opp;
        if (edge_faces(keep, rem, ef, opp) != 2) return false;
        neighbors(keep, na); neighbors(rem, nb);
        std::set<int> common;
        for (int x : na) if (nb.count(x)) common.insert(x);
        return common == opp;
    };

    // Dual-graph union-find over TRIANGLES (test A: a component is a simple
    // polygon iff its collapse edges form a TREE in the dual graph; a cycle = an
    // enclosed interior vertex = non-disk -> reject).
    std::vector<int> tparent(faces.size());
    auto tfind = [&](int x) { while (tparent[x] != x) { tparent[x] = tparent[tparent[x]]; x = tparent[x]; } return x; };
    // test B: which dual component owns each vertex (a triangle of that component,
    // resolved through tfind).  A vertex may be shared by edges of the SAME
    // component (so trees grow) but not across components (vertex-disjoint disks).
    std::vector<int> vowner(N, -1);
    auto rep = [&](int x) { while (!alive[x]) x = merge_to[x]; return x; };

    std::vector<std::vector<HierEntry>> level_recs;
    int alive_count = N;

    for (int level = 1; level <= num_levels; level++) {
        if (alive_count < 4) break;

        for (int i = 0; i < (int)faces.size(); i++) tparent[i] = i;
        std::fill(vowner.begin(), vowner.end(), -1);

        std::priority_queue<std::tuple<double,int,int>,
                            std::vector<std::tuple<double,int,int>>,
                            std::greater<std::tuple<double,int,int>>> pq;
        for (int fi = 0; fi < (int)faces.size(); fi++) {
            if (face_dead[fi]) continue;
            const auto& f = faces[fi];
            for (int j = 0; j < 3; j++) {
                int a = f[j], b = f[(j+1)%3];
                if (a > b) std::swap(a, b);
                pq.push({dist(a, b), a, b});
            }
        }

        // Grow a forest collapse: each accepted edge joins its two triangles in
        // the dual forest (must not close a cycle = disk constraint, test A) and
        // claims its two opposite vertices (must be unclaimed = vertex-disjoint
        // components, test B; also caps accepted edges at <= V/2 ~ T/2 tris).
        std::vector<std::pair<int,int>> batch;   // (keep, rem)
        std::vector<int> touched_tris;            // for component count
        while (!pq.empty()) {
            auto [len, a, b] = pq.top(); pq.pop();
            if (!alive[a] || !alive[b] || !adjacent(a, b)) continue;

            std::array<int,2> ef; std::set<int> opp;
            if (edge_faces(a, b, ef, opp) != 2) continue;       // boundary/non-manifold
            if (!link_ok(a, b)) continue;                       // manifold single collapse

            int r0 = tfind(ef[0]), r1 = tfind(ef[1]);
            if (r0 == r1) continue;                             // test A: dual cycle -> non-disk

            // test B (vertex-disjoint disks): the quad vertices (collapse endpoints
            // a,b plus the 2 opposite/boundary vertices) may only be owned by the
            // components being merged (r0/r1) or be free — never a third component.
            // Same-component sharing IS allowed, so trees grow into real multi-edge
            // simple polygons; only cross-component sharing is forbidden.
            bool blocked = false;
            int quad[4] = { a, b, -1, -1 };
            { int k = 2; for (int x : opp) quad[k++] = x; }
            for (int x : quad) {
                if (x < 0 || vowner[x] < 0) continue;
                int o = tfind(vowner[x]);
                if (o != r0 && o != r1) { blocked = true; break; }
            }
            if (blocked) continue;

            tparent[r0] = r1;                                   // merge dual component
            for (int x : quad) if (x >= 0) vowner[x] = ef[1];   // ef[1] resolves to merged root
            touched_tris.push_back(ef[0]); touched_tris.push_back(ef[1]);
            batch.push_back({ std::min(a,b), std::max(a,b) });
        }
        if (batch.empty()) break;

        // Component count = distinct dual roots among the collapsed triangles.
        std::set<int> comp_roots;
        for (int t : touched_tris) comp_roots.insert(tfind(t));
        int ncomp = (int)comp_roots.size();

        // Contract each tree toward its root: collapse edges in acceptance order,
        // resolving the survivor through merge_to, re-checking link condition now.
        std::vector<int> removed_this;
        removed_this.reserve(batch.size());
        for (auto& kr : batch) {
            int rem = kr.second;
            if (!alive[rem]) continue;
            int keep = rep(kr.first);
            if (keep == rem || !alive[keep] || !adjacent(keep, rem)) continue;
            if (!link_ok(keep, rem)) continue;

            for (int fi : std::vector<int>(vinc[rem].begin(), vinc[rem].end())) {
                auto& f = faces[fi];
                bool edge_face = (f[0]==keep || f[1]==keep || f[2]==keep);
                if (edge_face) {
                    face_dead[fi] = 1;
                    for (int j = 0; j < 3; j++) vinc[f[j]].erase(fi);
                } else {
                    for (int j = 0; j < 3; j++) if (f[j]==rem) f[j] = keep;
                    vinc[keep].insert(fi);
                }
            }
            vinc[rem].clear();
            alive[rem] = 0;
            merge_to[rem] = keep;
            level_of[rem] = level;
            removed_this.push_back(rem);
            alive_count--;
        }
        if (removed_this.empty()) break;

        // Barycentric prediction — IDENTICAL to the other hierarchies.
        std::vector<HierEntry> recs;
        recs.reserve(removed_this.size());
        for (int rem : removed_this) {
            int s = rem;
            while (!alive[s]) s = merge_to[s];
            HierEntry e; e.global_id = rem; e.level = level;
            auto pv = pos(rem);
            double best_pen = std::numeric_limits<double>::max();
            std::array<int,3>    best_tri{};
            std::array<double,3> best_b{};
            for (int fi : vinc[s]) {
                const auto& f = faces[fi];
                auto b3 = detail::barycentric(pv, pos(f[0]), pos(f[1]), pos(f[2]));
                double pen = -std::min({ b3[0], b3[1], b3[2], 0.0 });
                if (pen < best_pen) { best_pen = pen; best_tri = f; best_b = b3; }
            }
            if (best_pen != std::numeric_limits<double>::max()) {
                e.pred_ids = { best_tri[0], best_tri[1], best_tri[2] };
                e.weights  = { best_b[0], best_b[1], best_b[2] };
            }
            recs.push_back(std::move(e));
        }
        int collapsed = (int)removed_this.size();
        printf("  PFS level %d: %d collapsed (%d accepted edges in %d trees, "
               "avg %.2f edges/tree), ~%d tris, alive %d (keep %.1f%%)\n",
               level, collapsed, (int)batch.size(), ncomp,
               ncomp ? (double)batch.size()/ncomp : 0.0,
               2*collapsed, alive_count,
               100.0 * alive_count / (alive_count + collapsed));
        level_recs.push_back(std::move(recs));
    }

    // Base survivors: BPFS-ordered causal IDW (identical to the other hierarchies).
    std::vector<int> base_globals;
    base_globals.reserve(alive_count);
    std::vector<int> local_of(N, -1);
    for (int v = 0; v < N; v++)
        if (alive[v]) { local_of[v] = (int)base_globals.size(); base_globals.push_back(v); }
    int nb_base = (int)base_globals.size();

    std::vector<std::set<int32_t>> base_adj(nb_base);
    for (int fi = 0; fi < (int)faces.size(); fi++) {
        if (face_dead[fi]) continue;
        int a = local_of[faces[fi][0]], b = local_of[faces[fi][1]], c = local_of[faces[fi][2]];
        if (a < 0 || b < 0 || c < 0) continue;
        base_adj[a].insert(b); base_adj[a].insert(c);
        base_adj[b].insert(a); base_adj[b].insert(c);
        base_adj[c].insert(a); base_adj[c].insert(b);
    }

    std::vector<HierEntry> plan;
    plan.reserve(N);
    if (nb_base > 0) {
        auto order = BPFS(base_adj);
        std::vector<int> rank_to_node(nb_base);
        for (int c = 0; c < nb_base; c++) rank_to_node[order[c]] = c;
        for (int r = 0; r < nb_base; r++) {
            int c = rank_to_node[r], g = base_globals[c];
            HierEntry e; e.global_id = g; e.level = 0;
            double wsum = 0.0;
            for (int32_t u : base_adj[c]) {
                if (order[u] >= order[c]) continue;
                int gu = base_globals[u];
                double w = 1.0 / std::max(dist(g, gu), 1e-14);
                e.pred_ids.push_back(gu);
                e.weights.push_back(w);
                wsum += w;
            }
            for (double& w : e.weights) w /= wsum;
            plan.push_back(std::move(e));
        }
    }
    for (int L = (int)level_recs.size() - 1; L >= 0; L--)
        for (auto& e : level_recs[L]) plan.push_back(std::move(e));

    return plan;
}

} // namespace UMC
#endif
