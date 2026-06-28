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

// Greedy maximal independent set, prioritising low-valence vertices.
// Mirrors the Python _maximal_independent_set() in mesh_decimation.py.
static void greedy_mis(
    int                                         num_vertices,
    const std::vector<std::set<int32_t>>&       adj,
    std::vector<bool>&                          removed_mask,
    std::vector<int>&                           old_to_new)
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

    old_to_new.assign(num_vertices, -1);
    int count = 0;
    for (int i = 0; i < num_vertices; i++)
        if (!removed_mask[i]) old_to_new[i] = count++;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_dir>\n", argv[0]);
        fprintf(stderr, "  Pre-compute centroid_triangulation.dat with:\n");
        fprintf(stderr, "    python python_hierarchy/generate_centroid_triangulation.py"
                        " <mesh_nc> <data_dir>\n");
        return 1;
    }
    std::string data_dir(argv[1]);
    if (data_dir.back() != '/') data_dir += '/';

    // ------------------------------------------------------------------
    // Load mesh
    // ------------------------------------------------------------------
    size_t num;

    // Face centroids stored as float32, shape (N, 2).
    // Each "vertex" here is a face centroid of the original UGRID mesh.
    auto coords_f = readfile<float>((data_dir + "coordinates.dat").c_str(), num);
    const int d_stored = 2;
    int num_vertices = (int)(num / d_stored);
    printf("Face centroids (vertices): %d\n", num_vertices);

    // Promote to double and pad z=0 so V is flat (N x 3) for parameterization.
    std::vector<double> V(num_vertices * 3, 0.0);
    for (int i = 0; i < num_vertices; i++) {
        V[i*3+0] = coords_f[i*2+0];
        V[i*3+1] = coords_f[i*2+1];
        // V[i*3+2] stays 0.0
    }

    // Face-centroid triangulation: int32, shape (T, 3).
    // Generated once by generate_centroid_triangulation.py using the node-fan
    // method on mesh.nc — gives a proper triangulation in centroid space.
    auto faces = readfile<int>((data_dir + "centroid_triangulation.dat").c_str(), num);
    if (faces.empty()) {
        fprintf(stderr,
            "ERROR: %scentroid_triangulation.dat not found.\n"
            "Generate it with:\n"
            "  python python_hierarchy/generate_centroid_triangulation.py"
            " <mesh_nc> %s\n",
            data_dir.c_str(), data_dir.c_str());
        return 1;
    }
    int num_faces = (int)(num / 3);
    printf("Centroid triangles: %d\n", num_faces);

    // Height attribute stored as float32, one value per face centroid.
    auto attr_f = readfile<float>((data_dir + "height.dat.0").c_str(), num);
    std::vector<double> attr(num_vertices);
    for (int i = 0; i < num_vertices; i++) attr[i] = attr_f[i];

    double attr_min = *std::min_element(attr.begin(), attr.end());
    double attr_max = *std::max_element(attr.begin(), attr.end());
    double attr_range = attr_max - attr_min;
    printf("Attr range: %.4f  (min=%.4f  max=%.4f)\n",
           attr_range, attr_min, attr_max);

    // ------------------------------------------------------------------
    // Build vertex adjacency from the centroid triangulation (option=0,
    // d=2 -> 3 nodes per element) and use it for the greedy MIS.
    // Using the triangulation adjacency (degree ~6) rather than the
    // face-edge dual (degree 3) gives a better independent set.
    // ------------------------------------------------------------------
    auto adj = generate_adjacent_list<int>(num_vertices, d_stored, faces, /*option=*/0);

    // ------------------------------------------------------------------
    // Greedy maximal independent set -> removed_mask, old_to_new
    // ------------------------------------------------------------------
    std::vector<bool> removed_mask;
    std::vector<int>  old_to_new;
    greedy_mis(num_vertices, adj, removed_mask, old_to_new);

    int n_removed = 0;
    for (bool b : removed_mask) if (b) n_removed++;
    int n_coarse = num_vertices - n_removed;
    printf("\nGreedy MIS: %d removed (%.1f%%), %d surviving\n",
           n_removed, 100.0*n_removed/num_vertices, n_coarse);

    // ------------------------------------------------------------------
    // Build barycentric parameterization using the face-based overload.
    // The centroid triangulation provides proper triangular faces so
    // ordered_link can trace each removed vertex's cyclic 1-ring.
    // ------------------------------------------------------------------
    auto param = build_parameterization(
        V.data(), /*d=*/3,
        num_vertices,
        faces.data(), num_faces,
        removed_mask, old_to_new);

    printf("Parameterized: %zu / %d removed vertices (%.1f%%)\n",
           param.size(), n_removed, 100.0*(double)param.size()/n_removed);

    // ------------------------------------------------------------------
    // Evaluate barycentric prediction of height.dat.0
    // ------------------------------------------------------------------
    std::vector<double> coarse_attr(n_coarse);
    for (int i = 0; i < num_vertices; i++)
        if (!removed_mask[i]) coarse_attr[old_to_new[i]] = attr[i];

    double sum_sq = 0.0, max_abs = 0.0;
    // Split statistics by whether v falls INSIDE the anchor triangle
    // (penalty == 0, i.e. all barycentric weights >= 0) or OUTSIDE
    // (extrapolation, some weight < 0).
    int    n_outside = 0;
    double in_sum_sq = 0.0,  in_max_abs  = 0.0;
    double out_sum_sq = 0.0, out_max_abs = 0.0;
    double worst_neg_bary = 0.0;   // most-negative barycentric weight seen
    std::vector<float> coarse_level_barycentric_res(param.size());
    int coarse_barycentric_idx = 0;
    for (auto& kv : param) {
        int v = kv.first;
        const BarycentricParam& p = kv.second;
        double pred = p.bary[0] * coarse_attr[p.tri_coarse[0]]
                    + p.bary[1] * coarse_attr[p.tri_coarse[1]]
                    + p.bary[2] * coarse_attr[p.tri_coarse[2]];
        double res = attr[v] - pred;
        coarse_level_barycentric_res[coarse_barycentric_idx++] = static_cast<float>(res);
        double ares = std::abs(res);
        sum_sq += res * res;
        if (ares > max_abs) max_abs = ares;

        double min_b = std::min({ p.bary[0], p.bary[1], p.bary[2] });
        if (min_b < 0.0) {                 // outside: extrapolation
            n_outside++;
            out_sum_sq += res * res;
            if (ares > out_max_abs) out_max_abs = ares;
            if (min_b < worst_neg_bary) worst_neg_bary = min_b;
        } else {                            // inside: true interpolation
            in_sum_sq += res * res;
            if (ares > in_max_abs) in_max_abs = ares;
        }
    }
    writefile<float>((data_dir + "coarse_bary_res.dat").c_str(), coarse_level_barycentric_res.data(), coarse_level_barycentric_res.size());

    double bary_rms = 0.0;
    if (!param.empty()) {
        bary_rms = std::sqrt(sum_sq / (double)param.size());
        printf("\nBarycentric prediction residuals (height.dat.0, %zu vertices):\n", param.size());
        printf("  RMS = %.4e  (%.3f%% of range)\n", bary_rms,  100.0*bary_rms /attr_range);
        printf("  Max = %.4e  (%.3f%% of range)\n", max_abs,   100.0*max_abs  /attr_range);

        int n_inside = (int)param.size() - n_outside;
        printf("\n  Penalty != 0 (v OUTSIDE anchor triangle, extrapolation):\n");
        printf("    outside: %d / %zu  (%.2f%%)   worst negative bary = %.3f\n",
               n_outside, param.size(), 100.0*n_outside/param.size(), worst_neg_bary);
        if (n_inside > 0)
            printf("    inside  RMS = %.4e (%.3f%% range)   Max = %.4e (%.3f%% range)\n",
                   std::sqrt(in_sum_sq/n_inside), 100.0*std::sqrt(in_sum_sq/n_inside)/attr_range,
                   in_max_abs, 100.0*in_max_abs/attr_range);
        if (n_outside > 0)
            printf("    outside RMS = %.4e (%.3f%% range)   Max = %.4e (%.3f%% range)\n",
                   std::sqrt(out_sum_sq/n_outside), 100.0*std::sqrt(out_sum_sq/n_outside)/attr_range,
                   out_max_abs, 100.0*out_max_abs/attr_range);
    }

    // ------------------------------------------------------------------
    // IDW prediction: each removed vertex is predicted by the normalised
    // inverse-distance-weighted average of all adjacent surviving vertices.
    // ------------------------------------------------------------------
    auto idw_param = build_idw_parameterization(
        V.data(), /*d=*/3, num_vertices, adj, removed_mask, old_to_new);

    printf("\nIDW parameterized: %zu / %d removed vertices (%.1f%%)\n",
           idw_param.size(), n_removed, 100.0*(double)idw_param.size()/n_removed);

    std::vector<float> coarse_level_idw_res(idw_param.size());
    int coarse_idw_idx = 0;
    double idw_sum_sq = 0.0, idw_max_abs = 0.0;
    for (auto& kv : idw_param) {
        int v = kv.first;
        const IDWParam& p = kv.second;
        double pred = 0.0;
        for (int i = 0; i < (int)p.neighbors.size(); i++)
            pred += p.weights[i] * coarse_attr[p.neighbors[i]];
        double res = attr[v] - pred;
        coarse_level_idw_res[coarse_idw_idx++] = static_cast<float>(res);
        idw_sum_sq += res * res;
        if (std::abs(res) > idw_max_abs) idw_max_abs = std::abs(res);
    }
    writefile<float>((data_dir + "coarse_idw_res.dat").c_str(), coarse_level_idw_res.data(), coarse_level_idw_res.size());

    double idw_rms = 0.0;
    if (!idw_param.empty()) {
        idw_rms = std::sqrt(idw_sum_sq / (double)idw_param.size());
        printf("\nIDW prediction residuals (height.dat.0, %zu vertices):\n", idw_param.size());
        printf("  RMS = %.4e  (%.3f%% of range)\n", idw_rms,    100.0*idw_rms    /attr_range);
        printf("  Max = %.4e  (%.3f%% of range)\n", idw_max_abs, 100.0*idw_max_abs/attr_range);
    }

    // ------------------------------------------------------------------
    // Summary comparison
    // ------------------------------------------------------------------
    if (!param.empty() && !idw_param.empty()) {
        printf("\n%-12s  %12s  %12s\n", "Method", "RMS", "Max");
        printf("%-12s  %12.4e  %12.4e\n", "Barycentric", bary_rms,  max_abs);
        printf("%-12s  %12.4e  %12.4e\n", "IDW",         idw_rms,   idw_max_abs);
        printf("IDW/Bary RMS ratio: %.3f\n", idw_rms / bary_rms);
    }

    return 0;
}
