#include <cstdio>
#include <cmath>
#include <vector>
#include <set>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <string>

#include "utils.hpp"
#include "adjacent_prediction.hpp"
#include "parameterization.hpp"

using namespace UMC;

// ---------------------------------------------------------------------------
// Greedy maximal independent set, prioritising low-valence vertices.
// Mirrors the Python _maximal_independent_set() in mesh_decimation.py.
// (Same routine as the single-level test.)
// ---------------------------------------------------------------------------
static void greedy_mis(
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

// ---------------------------------------------------------------------------
// Un-remove any vertex whose cyclic 1-ring cannot be traced (boundary or
// non-manifold). Those holes cannot be fan-retriangulated, so keeping them
// removed would leave gaps that break ordered_link at the next level.
// Returns the per-vertex incident-face lists (reused by the caller).
// ---------------------------------------------------------------------------
static std::vector<std::vector<int>> prune_unringed_removals(
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
        auto ring = detail::ordered_link(v, incident[v], faces.data());
        if (ring.size() < 3) removed_mask[v] = false;  // keep this vertex
    }
    return incident;
}

// ---------------------------------------------------------------------------
// Build the coarse-mesh triangulation after removing an independent set.
// Each face has at most one removed vertex (MIS property):
//   - 0 removed: face survives, remap its 3 indices through old_to_new
//   - 1 removed: face dropped; the removed vertex's 1-ring is fan-retriangulated
//                once (anchored at ring[0]) — same as Python independent_set_decimate.
// Returns coarse faces (flat, 3 ints per triangle) in COARSE indexing.
// ---------------------------------------------------------------------------
static std::vector<int> build_coarse_triangulation(
    const std::vector<int>&              faces,
    int                                  num_faces,
    int                                  num_vertices,
    const std::vector<std::vector<int>>& incident,
    const std::vector<bool>&             removed_mask,
    const std::vector<int>&              old_to_new)
{
    std::vector<int> coarse;
    coarse.reserve(num_faces * 3);

    // 1. Surviving faces (no removed vertex), remapped to coarse indices.
    for (int fi = 0; fi < num_faces; fi++) {
        int a = faces[fi*3+0], b = faces[fi*3+1], c = faces[fi*3+2];
        if (removed_mask[a] || removed_mask[b] || removed_mask[c]) continue;
        coarse.push_back(old_to_new[a]);
        coarse.push_back(old_to_new[b]);
        coarse.push_back(old_to_new[c]);
    }

    // 2. Fan-retriangulate each removed vertex's hole.
    for (int v = 0; v < num_vertices; v++) {
        if (!removed_mask[v]) continue;
        auto ring = detail::ordered_link(v, incident[v], faces.data());
        if (ring.size() < 3) continue;  // pruned earlier; defensive
        for (size_t i = 1; i + 1 < ring.size(); i++) {
            coarse.push_back(old_to_new[ring[0]]);
            coarse.push_back(old_to_new[ring[i]]);
            coarse.push_back(old_to_new[ring[i+1]]);
        }
    }
    return coarse;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_dir> [num_levels=1]\n", argv[0]);
        fprintf(stderr, "  Pre-compute centroid_triangulation.dat with:\n");
        fprintf(stderr, "    python python_hierarchy/generate_centroid_triangulation.py"
                        " <mesh_nc> <data_dir>\n");
        return 1;
    }
    std::string data_dir(argv[1]);
    if (data_dir.back() != '/') data_dir += '/';
    int num_levels = (argc >= 3) ? std::atoi(argv[2]) : 1;
    if (num_levels < 1) num_levels = 1;

    // ------------------------------------------------------------------
    // Load the finest (level-0) mesh.
    // ------------------------------------------------------------------
    size_t num;

    auto coords_f = readfile<float>((data_dir + "coordinates.dat").c_str(), num);
    int num_vertices = (int)(num / 2);            // float32, (N, 2)
    printf("Face centroids (vertices): %d\n", num_vertices);

    // Flat (N x 3) double positions, z padded to 0.
    std::vector<double> V(num_vertices * 3, 0.0);
    for (int i = 0; i < num_vertices; i++) {
        V[i*3+0] = coords_f[i*2+0];
        V[i*3+1] = coords_f[i*2+1];
    }

    auto faces_i = readfile<int>((data_dir + "centroid_triangulation.dat").c_str(), num);
    if (faces_i.empty()) {
        fprintf(stderr,
            "ERROR: %scentroid_triangulation.dat not found.\n"
            "Generate it with:\n"
            "  python python_hierarchy/generate_centroid_triangulation.py"
            " <mesh_nc> %s\n",
            data_dir.c_str(), data_dir.c_str());
        return 1;
    }
    std::vector<int> faces(faces_i.begin(), faces_i.end());
    int num_faces = (int)(faces.size() / 3);
    printf("Centroid triangles: %d\n", num_faces);

    auto attr_f = readfile<float>((data_dir + "height.dat.0").c_str(), num);
    std::vector<double> attr(num_vertices);
    for (int i = 0; i < num_vertices; i++) attr[i] = attr_f[i];

    double attr_min = *std::min_element(attr.begin(), attr.end());
    double attr_max = *std::max_element(attr.begin(), attr.end());
    double attr_range = attr_max - attr_min;   // fixed reference for all levels
    printf("Attr range: %.4f  (min=%.4f  max=%.4f)\n", attr_range, attr_min, attr_max);

    // ------------------------------------------------------------------
    // Current mesh state, coarsened in place each level. Level 0 = finest.
    // ------------------------------------------------------------------
    std::vector<double> cur_V    = std::move(V);
    std::vector<int>    cur_faces = std::move(faces);
    std::vector<double> cur_attr  = std::move(attr);
    int cur_nv = num_vertices;
    int n0     = num_vertices;

    struct LevelStat {
        int    level;
        int    n_in, n_out, n_removed, n_param;
        double bary_rms, bary_max;
        double idw_rms,  idw_max;
        double cumulative_ratio;
    };
    std::vector<LevelStat> stats;

    // All residuals concatenated across levels (coarse -> fine order).
    std::vector<float> all_bary_res;
    std::vector<float> all_idw_res;

    for (int level = 1; level <= num_levels; level++) {
        int    nf = (int)(cur_faces.size() / 3);
        printf("\n========== Level %d  (in: %d vertices, %d faces) ==========\n",
               level, cur_nv, nf);

        if (cur_nv < 4 || nf < 1) {
            printf("  Mesh too small to coarsen further; stopping.\n");
            break;
        }

        // Adjacency from the current triangulation (option=0, d=2 -> 3 nodes/elem).
        auto adj = generate_adjacent_list<int>(cur_nv, 2, cur_faces, /*option=*/0);

        // Maximal independent set, then prune unringed (boundary/non-manifold)
        // removals so every removed vertex has a clean cyclic 1-ring.
        std::vector<bool> removed_mask;
        greedy_mis(cur_nv, adj, removed_mask);
        auto incident = prune_unringed_removals(cur_faces, nf, cur_nv, removed_mask);

        int n_removed = 0;
        for (bool b : removed_mask) if (b) n_removed++;
        int n_coarse = cur_nv - n_removed;

        std::vector<int> old_to_new(cur_nv, -1);
        for (int i = 0, c = 0; i < cur_nv; i++)
            if (!removed_mask[i]) old_to_new[i] = c++;

        printf("Greedy MIS: %d removed (%.1f%%), %d surviving\n",
               n_removed, 100.0*n_removed/cur_nv, n_coarse);

        // Survivor (coarse-level) attribute array.
        std::vector<double> coarse_attr(n_coarse);
        for (int i = 0; i < cur_nv; i++)
            if (!removed_mask[i]) coarse_attr[old_to_new[i]] = cur_attr[i];

        // --- Barycentric parameterization + residuals ---
        auto param = build_parameterization(
            cur_V.data(), /*d=*/3, cur_nv,
            cur_faces.data(), nf, removed_mask, old_to_new);

        double sum_sq = 0.0, max_abs = 0.0;
        std::vector<float> bary_res; bary_res.reserve(param.size());
        for (auto& kv : param) {
            int v = kv.first;
            const BarycentricParam& p = kv.second;
            double pred = p.bary[0] * coarse_attr[p.tri_coarse[0]]
                        + p.bary[1] * coarse_attr[p.tri_coarse[1]]
                        + p.bary[2] * coarse_attr[p.tri_coarse[2]];
            double res = cur_attr[v] - pred;
            bary_res.push_back((float)res);
            sum_sq += res * res;
            if (std::abs(res) > max_abs) max_abs = std::abs(res);
        }
        double bary_rms = param.empty() ? 0.0 : std::sqrt(sum_sq / param.size());

        // --- IDW parameterization + residuals ---
        auto idw_param = build_idw_parameterization(
            cur_V.data(), /*d=*/3, cur_nv, adj, removed_mask, old_to_new);

        double idw_sum_sq = 0.0, idw_max_abs = 0.0;
        std::vector<float> idw_res; idw_res.reserve(idw_param.size());
        for (auto& kv : idw_param) {
            int v = kv.first;
            const IDWParam& p = kv.second;
            double pred = 0.0;
            for (int i = 0; i < (int)p.neighbors.size(); i++)
                pred += p.weights[i] * coarse_attr[p.neighbors[i]];
            double res = cur_attr[v] - pred;
            idw_res.push_back((float)res);
            idw_sum_sq += res * res;
            if (std::abs(res) > idw_max_abs) idw_max_abs = std::abs(res);
        }
        double idw_rms = idw_param.empty() ? 0.0 : std::sqrt(idw_sum_sq / idw_param.size());

        printf("Parameterized: bary %zu, idw %zu  / %d removed\n",
               param.size(), idw_param.size(), n_removed);
        printf("  Barycentric  RMS = %.4e (%.3f%%)   Max = %.4e (%.3f%%)\n",
               bary_rms, 100.0*bary_rms/attr_range, max_abs, 100.0*max_abs/attr_range);
        printf("  IDW          RMS = %.4e (%.3f%%)   Max = %.4e (%.3f%%)\n",
               idw_rms, 100.0*idw_rms/attr_range, idw_max_abs, 100.0*idw_max_abs/attr_range);

        // Accumulate residuals; all levels are written to one file each below.
        all_bary_res.insert(all_bary_res.end(), bary_res.begin(), bary_res.end());
        all_idw_res.insert(all_idw_res.end(),  idw_res.begin(),  idw_res.end());

        // --- Build the coarse triangulation and compact to the next level. ---
        auto coarse_faces = build_coarse_triangulation(
            cur_faces, nf, cur_nv, incident, removed_mask, old_to_new);

        std::vector<double> next_V(n_coarse * 3);
        for (int i = 0; i < cur_nv; i++) {
            if (removed_mask[i]) continue;
            int ni = old_to_new[i];
            next_V[ni*3+0] = cur_V[i*3+0];
            next_V[ni*3+1] = cur_V[i*3+1];
            next_V[ni*3+2] = cur_V[i*3+2];
        }

        stats.push_back({ level, cur_nv, n_coarse, n_removed,
                          (int)param.size(), bary_rms, max_abs,
                          idw_rms, idw_max_abs,
                          (double)(n0 - n_coarse) / n0 });

        cur_V     = std::move(next_V);
        cur_faces = std::move(coarse_faces);
        cur_attr  = std::move(coarse_attr);
        cur_nv    = n_coarse;
    }

    // ------------------------------------------------------------------
    // Write all residuals across levels into a single file per predictor.
    // ------------------------------------------------------------------
    writefile<float>((data_dir + "bary_res.dat").c_str(),
                     all_bary_res.data(), all_bary_res.size());
    writefile<float>((data_dir + "idw_res.dat").c_str(),
                     all_idw_res.data(), all_idw_res.size());
    printf("\nWrote %zu barycentric residuals -> %sbary_res.dat\n",
           all_bary_res.size(), data_dir.c_str());
    printf("Wrote %zu IDW residuals         -> %sidw_res.dat\n",
           all_idw_res.size(), data_dir.c_str());

    // ------------------------------------------------------------------
    // Hierarchy summary.
    // ------------------------------------------------------------------
    printf("\n========== Hierarchy summary (range = %.4f) ==========\n", attr_range);
    printf("%-5s %9s %9s %9s   %10s %10s   %10s %10s   %8s\n",
           "Lvl", "in", "out", "removed",
           "baryRMS%", "baryMax%", "idwRMS%", "idwMax%", "cumRem%");
    for (auto& s : stats) {
        printf("%-5d %9d %9d %9d   %10.3f %10.3f   %10.3f %10.3f   %8.1f\n",
               s.level, s.n_in, s.n_out, s.n_removed,
               100.0*s.bary_rms/attr_range, 100.0*s.bary_max/attr_range,
               100.0*s.idw_rms /attr_range, 100.0*s.idw_max /attr_range,
               100.0*s.cumulative_ratio);
    }
    return 0;
}
