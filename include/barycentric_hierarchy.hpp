#ifndef _UMC_BARYCENTRIC_HIERARCHY_HPP
#define _UMC_BARYCENTRIC_HIERARCHY_HPP

#include <vector>
#include <set>
#include <array>
#include <cstdint>
#include <numeric>
#include <algorithm>

#include "adjacent_prediction.hpp"
#include "parameterization.hpp"
#include "reorder.hpp"            // BPFS ordering for the base layer

namespace UMC {

// One entry of the multilevel processing plan.  Prediction is the generic
// weighted sum   pred = sum_i weights[i] * dec_data[pred_ids[i]] ,  where every
// pred_ids[i] is a global vertex id already decoded when this entry is reached:
//   - removed vertices : 3 coarse anchors  + barycentric weights
//   - base vertices    : earlier base neighbors + normalised IDW weights
//   - empty pred_ids   : no predictor (pred = 0)
struct HierEntry {
    int                 global_id;
    int                 level;        // 0 = base survivor; L = removed at level L
    std::vector<int>    pred_ids;
    std::vector<double> weights;
    int                 pred_kind = -1;  // diagnostics: 0=enclosed tet, 1=outside hull, 2=IDW, -1=base/unset
};

namespace detail {

// Greedy maximal independent set, prioritising low-valence vertices.
// Deterministic for identical inputs (std::sort on a fixed comparator), so the
// compressor and decompressor reconstruct the same set.
inline void greedy_mis(
    int                                   num_vertices,
    const std::vector<std::set<int32_t>>& adj,
    std::vector<bool>&                    removed_mask)
{
    std::vector<int> order(num_vertices);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return adj[a].size() < adj[b].size(); });

    std::vector<bool> blocked(num_vertices, false);
    removed_mask.assign(num_vertices, false);
    for (int v : order) {
        if (!blocked[v]) {
            removed_mask[v] = true;
            blocked[v]      = true;
            for (int32_t u : adj[v]) blocked[u] = true;
        }
    }
}

// Un-remove vertices whose cyclic 1-ring cannot be traced (boundary /
// non-manifold), so coarse holes can always be fan-retriangulated.
// Returns per-vertex incident-face lists (reused by build_coarse_triangulation).
inline std::vector<std::vector<int>> prune_unringed_removals(
    const std::vector<int>& faces,
    int                     num_faces,
    int                     num_vertices,
    std::vector<bool>&      removed_mask)
{
    std::vector<std::vector<int>> incident(num_vertices);
    for (int fi = 0; fi < num_faces; fi++)
        for (int j = 0; j < 3; j++)
            incident[faces[fi*3+j]].push_back(fi);

    for (int v = 0; v < num_vertices; v++) {
        if (!removed_mask[v]) continue;
        auto ring = ordered_link(v, incident[v], faces.data());
        if (ring.size() < 3) removed_mask[v] = false;
    }
    return incident;
}

// Coarse-mesh triangulation after removing an independent set: surviving faces
// (remapped) plus the fan-retriangulation of every removed vertex's 1-ring.
inline std::vector<int> build_coarse_triangulation(
    const std::vector<int>&              faces,
    int                                  num_faces,
    int                                  num_vertices,
    const std::vector<std::vector<int>>& incident,
    const std::vector<bool>&             removed_mask,
    const std::vector<int>&              old_to_new)
{
    std::vector<int> coarse;
    coarse.reserve(num_faces * 3);

    for (int fi = 0; fi < num_faces; fi++) {
        int a = faces[fi*3+0], b = faces[fi*3+1], c = faces[fi*3+2];
        if (removed_mask[a] || removed_mask[b] || removed_mask[c]) continue;
        coarse.push_back(old_to_new[a]);
        coarse.push_back(old_to_new[b]);
        coarse.push_back(old_to_new[c]);
    }
    for (int v = 0; v < num_vertices; v++) {
        if (!removed_mask[v]) continue;
        auto ring = ordered_link(v, incident[v], faces.data());
        if (ring.size() < 3) continue;
        for (size_t i = 1; i + 1 < ring.size(); i++) {
            coarse.push_back(old_to_new[ring[0]]);
            coarse.push_back(old_to_new[ring[i]]);
            coarse.push_back(old_to_new[ring[i+1]]);
        }
    }
    return coarse;
}

} // namespace detail

// Build the deterministic processing plan for multilevel barycentric prediction.
//
// V_in     flat (num_vertices*3) double positions (z padded to 0 for 2-D data)
// faces_in flat centroid triangulation, 3 ints per triangle
//
// Returns a plan of exactly num_vertices entries: base survivors first, then
// removed vertices ordered coarsest level -> finest level.  Processed in this
// order, every anchor of an entry has already been reconstructed, so the
// barycentric prediction is causal and reproducible on the decompressor.
// Removed vertices use barycentric prediction (3 coarse anchors); base
// survivors have no coarser parent and use causal same-level IDW.
inline std::vector<HierEntry> build_barycentric_hierarchy(
    const std::vector<double>& V_in,
    const std::vector<int>&    faces_in,
    int                        num_vertices,
    int                        num_levels,
    bool                       all_triples   = false,
    int                        anchor_metric = 0)
{
    std::vector<double> cur_V     = V_in;
    std::vector<int>    cur_faces = faces_in;
    int                 cur_nv    = num_vertices;
    std::vector<int>    cur_to_global(num_vertices);
    std::iota(cur_to_global.begin(), cur_to_global.end(), 0);

    std::vector<std::vector<HierEntry>> level_recs;  // removed entries per level

    for (int level = 1; level <= num_levels; level++) {
        int nf = (int)(cur_faces.size() / 3);
        if (cur_nv < 4 || nf < 1) break;

        auto adj = generate_adjacent_list<int>(cur_nv, 2, cur_faces, /*option=*/0);

        std::vector<bool> removed_mask;
        detail::greedy_mis(cur_nv, adj, removed_mask);
        auto incident = detail::prune_unringed_removals(cur_faces, nf, cur_nv, removed_mask);

        int n_coarse = 0;
        std::vector<int> old_to_new(cur_nv, -1);
        for (int i = 0; i < cur_nv; i++)
            if (!removed_mask[i]) old_to_new[i] = n_coarse++;

        std::vector<int> next_to_global(n_coarse);
        for (int i = 0; i < cur_nv; i++)
            if (!removed_mask[i]) next_to_global[old_to_new[i]] = cur_to_global[i];

        // Each removed vertex's 3 coarse anchors + barycentric weights.
        // all_triples + anchor_metric select how the anchor triangle is chosen
        // (fan/min-penalty vs full search by max-min-weight / smallest interior).
        auto param = build_parameterization(
            cur_V.data(), /*d=*/3, cur_nv,
            cur_faces.data(), nf, removed_mask, old_to_new, all_triples, anchor_metric);

        std::vector<HierEntry> recs;
        recs.reserve(cur_nv - n_coarse);
        for (int v = 0; v < cur_nv; v++) {
            if (!removed_mask[v]) continue;
            HierEntry e;
            e.global_id = cur_to_global[v];
            e.level     = level;
            auto it = param.find(v);
            if (it != param.end()) {              // empty -> pred 0 (boundary)
                const BarycentricParam& p = it->second;
                e.pred_ids = { next_to_global[p.tri_coarse[0]],
                               next_to_global[p.tri_coarse[1]],
                               next_to_global[p.tri_coarse[2]] };
                e.weights  = { p.bary[0], p.bary[1], p.bary[2] };
            }
            recs.push_back(std::move(e));
        }
        level_recs.push_back(std::move(recs));

        // Coarsen the mesh for the next level.
        auto coarse_faces = detail::build_coarse_triangulation(
            cur_faces, nf, cur_nv, incident, removed_mask, old_to_new);

        std::vector<double> next_V(n_coarse * 3);
        for (int i = 0; i < cur_nv; i++) {
            if (removed_mask[i]) continue;
            int ni = old_to_new[i];
            next_V[ni*3+0] = cur_V[i*3+0];
            next_V[ni*3+1] = cur_V[i*3+1];
            next_V[ni*3+2] = cur_V[i*3+2];
        }
        cur_V         = std::move(next_V);
        cur_faces     = std::move(coarse_faces);
        cur_to_global = std::move(next_to_global);
        cur_nv        = n_coarse;
    }

    // Assemble plan: base survivors first, then removed vertices coarsest ->
    // finest so anchors precede dependents.
    //
    // Base survivors have no coarser parent, but they form their own coarse mesh
    // and are spatially correlated.  Predict each from its already-decoded base
    // neighbors (causal IDW) instead of from 0, turning "encode absolute value"
    // into "encode local deviation" and sharply lowering the coarsest layer's
    // entropy.  The decode order is BPFS (breadth-first with priority): visit the
    // node with the most already-processed neighbors first, so every base point
    // has the maximum number of decoded neighbors available for its prediction —
    // far better than an arbitrary ascending-index sweep.
    std::vector<HierEntry> plan;
    plan.reserve(num_vertices);

    if (cur_nv > 0) {
        auto base_adj = generate_adjacent_list<int>(
            cur_nv, 2, cur_faces, /*option=*/0);
        auto order = DPFS(base_adj);              // order[c] = DPFS rank of base node c
        std::vector<int> rank_to_node(cur_nv);
        for (int c = 0; c < cur_nv; c++) rank_to_node[order[c]] = c;

        for (int r = 0; r < cur_nv; r++) {
            int c = rank_to_node[r];
            HierEntry e;
            e.global_id = cur_to_global[c];
            e.level     = 0;
            double wsum = 0.0;
            for (int32_t u : base_adj[c]) {
                if (order[u] >= order[c]) continue;   // not yet decoded in BPFS order
                double dx = cur_V[c*3+0] - cur_V[u*3+0];
                double dy = cur_V[c*3+1] - cur_V[u*3+1];
                double dz = cur_V[c*3+2] - cur_V[u*3+2];
                double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (dist < 1e-14) dist = 1e-14;
                double w = 1.0 / dist;
                e.pred_ids.push_back(cur_to_global[u]);
                e.weights.push_back(w);
                wsum += w;
            }
            for (double& w : e.weights) w /= wsum;     // normalise (wsum>0 here)
            plan.push_back(std::move(e));               // BPFS root: empty -> pred 0
        }
    }

    for (int L = (int)level_recs.size() - 1; L >= 0; L--)
        for (auto& e : level_recs[L]) plan.push_back(std::move(e));

    return plan;
}

} // namespace UMC
#endif
