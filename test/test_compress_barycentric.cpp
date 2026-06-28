// Multilevel barycentric-prediction compressor (works on any triangulated
// surface: 2-D planar e.g. Tsunami, or 3-D-on-a-sphere e.g. Ocean).
//
// Mirrors the machinery of Compression Method 2 (NBP-IDW) in compress.hpp —
// LinearQuantizer -> HuffmanEncoder -> Lossless_zstd — but the predictor is the
// multilevel MIS + barycentric parameterization.  The prediction structure is
// derived purely from geometry (coords + triangulation), so it is NOT stored:
// the decompressor rebuilds the identical hierarchy and only the quantized
// residuals travel in the compressed stream.
//
// Usage:
//   test_compress_barycentric <data_dir> <coord_file> <dim> <tri_file> <var_file>
//                             <num_levels> <abs_eb_or_not> <eb> [hierarchy=0] [keep_ratio=0.5]
// Writes: <data_dir>/<var_file>.umc

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <chrono>

#include <SZ3/quantizer/IntegerQuantizer.hpp>
#include <SZ3/encoder/HuffmanEncoder.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>

#include "utils.hpp"
#include "barycentric_hierarchy.hpp"
#include "cluster_hierarchy.hpp"
#include "edge_collapse_hierarchy.hpp"

using namespace UMC;
using T = float;

int main(int argc, char** argv)
{
    if (argc < 9) {
        fprintf(stderr,
            "Usage: %s <data_dir> <coord_file> <dim> <tri_file> <var_file> "
            "<num_levels> <abs_eb_or_not> <eb> [hierarchy=0] [keep_ratio=0.5]\n"
            "  coord_file: vertex positions, float32, dim per vertex (e.g. coords.dat)\n"
            "  dim:        2 or 3 (z padded to 0 when 2)\n"
            "  tri_file:   triangle connectivity, int32, 3 per triangle\n"
            "  var_file:   scalar attribute, float32, one per vertex (e.g. temperature.dat.0)\n"
            "  hierarchy:  0 = MIS+barycentric, 1 = clustering+barycentric,\n"
            "              2 = edge-collapse+barycentric (keep_ratio per level, e.g. 0.25 for 2-D)\n"
            "  keep_ratio: survivors kept per level (clustering/edge-collapse)\n"
            "Example (Ocean):   %s data/Ocean coords.dat 3 triangulation.dat temperature.dat.0 4 0 1e-2\n"
            "Example (Tsunami): %s data/Tsunami coordinates.dat 2 centroid_triangulation.dat height.dat.0 4 0 1e-2\n",
            argv[0], argv[0], argv[0]);
        return 1;
    }
    std::string data_dir(argv[1]);
    if (data_dir.back() != '/') data_dir += '/';
    std::string coord_file = argv[2];
    int    dim         = std::atoi(argv[3]);
    std::string tri_file   = argv[4];
    std::string var_file   = argv[5];
    int    num_levels  = std::atoi(argv[6]);
    bool   use_abs_eb  = std::atoi(argv[7]) != 0;
    double eb          = std::atof(argv[8]);
    int    hierarchy   = (argc >= 10) ? std::atoi(argv[9]) : 0;
    double keep_ratio  = (argc >= 11) ? std::atof(argv[10]) : 0.5;
    // anchor_metric (MIS only): 0 = fan/min-penalty (legacy), 1 = all-triples
    // max-min-weight (most interior — best; default), 2 = all-triples smallest
    // strictly-interior.  Metric 1 maximizes min(w0,w1,w2) so the point is well
    // inside the triangle (all 3 anchors genuinely used, not edge-degenerate).
    int    anchor_metric = (argc >= 12) ? std::atoi(argv[11]) : 1;
    if (num_levels < 1) num_levels = 1;

    // Phase timer: lap() prints seconds since the previous lap and resets.
    using Clock = std::chrono::steady_clock;
    auto t_prev = Clock::now();
    auto lap = [&](const char* name) {
        auto t = Clock::now();
        printf("[time] %-22s %.4f s\n", name,
               std::chrono::duration<double>(t - t_prev).count());
        t_prev = t;
    };

    // ------------------------------------------------------------------
    // Load geometry + attribute.
    // ------------------------------------------------------------------
    size_t num = 0;
    auto coords_f = readfile<float>((data_dir + coord_file).c_str(), num);
    int num_vertices = (int)(num / dim);
    std::vector<double> V(num_vertices * 3, 0.0);   // always padded to N x 3
    for (int i = 0; i < num_vertices; i++)
        for (int c = 0; c < dim; c++)
            V[i*3+c] = coords_f[i*dim+c];

    auto faces_i = readfile<int>((data_dir + tri_file).c_str(), num);
    if (faces_i.empty()) {
        fprintf(stderr, "ERROR: %s%s not found.\n", data_dir.c_str(), tri_file.c_str());
        return 1;
    }
    std::vector<int> faces(faces_i.begin(), faces_i.end());

    std::string data_file = data_dir + var_file;
    size_t num_elements = 0;
    auto data = readfile<T>(data_file.c_str(), num_elements);
    if ((int)num_elements != num_vertices) {
        fprintf(stderr, "ERROR: attribute count %zu != vertex count %d\n",
                num_elements, num_vertices);
        return 1;
    }

    T max_val = *std::max_element(data.begin(), data.end());
    T min_val = *std::min_element(data.begin(), data.end());
    double valid_range = (double)(max_val - min_val);
    if (!use_abs_eb) eb *= valid_range;
    printf("vertices=%d  range=%.4f  abs_eb=%.6g  levels=%d\n",
           num_vertices, valid_range, eb, num_levels);
    lap("load geometry+attr");

    // ------------------------------------------------------------------
    // Build the prediction plan (identical on the decompressor).
    // ------------------------------------------------------------------
    const char* hname = hierarchy == 0 ? "MIS+barycentric"
                      : hierarchy == 1 ? "clustering+barycentric"
                      : hierarchy == 2 ? "edge-collapse+barycentric"
                      : hierarchy == 3 ? "forest-collapse+barycentric"
                      : "PFS-collapse+barycentric";
    printf("hierarchy = %s\n", hname);
    double spacing0 = median_edge_length(V, faces);
    std::vector<HierEntry> plan;
    if (hierarchy == 0)
        plan = build_barycentric_hierarchy(V, faces, num_vertices, num_levels,
                                           anchor_metric > 0, anchor_metric);
    else if (hierarchy == 1)
        plan = build_cluster_hierarchy(V, num_vertices, num_levels, keep_ratio, spacing0, /*verbose=*/true);
    else if (hierarchy == 2)
        plan = build_edge_collapse_hierarchy(V, faces, num_vertices, num_levels, keep_ratio);
    else if (hierarchy == 3)
        plan = build_forest_collapse_hierarchy(V, faces, num_vertices, num_levels);
    else
        plan = build_pfs_collapse_hierarchy(V, faces, num_vertices, num_levels);
    printf("plan entries = %zu (== vertices: %s)\n",
           plan.size(), plan.size() == (size_t)num_vertices ? "yes" : "NO");
    lap("build hierarchy");

    // ------------------------------------------------------------------
    // Quantize residuals along the plan (coarse -> fine, causal anchors).
    // ------------------------------------------------------------------
    std::vector<T> dec_data(data);                 // overwritten with reconstruction
    const int quant_radius = 32768;
    SZ3::LinearQuantizer<T> quantizer(eb, quant_radius);
    std::vector<int> quant_inds;
    quant_inds.reserve(num_elements);

    // Per-level prediction-residual accumulation (residual = truth - pred,
    // computed against the lossy anchors the decompressor will see).
    std::vector<double> lvl_sumsq(num_levels + 1, 0.0);
    std::vector<double> lvl_maxabs(num_levels + 1, 0.0);
    std::vector<size_t> lvl_count(num_levels + 1, 0);
    std::vector<size_t> lvl_neg(num_levels + 1, 0);   // entries with a negative weight (penalty!=0)
    // Base (level 0) IDW neighbor-count stats.
    size_t base_nbr_sum = 0, base_nbr_min = SIZE_MAX, base_nbr_max = 0, base_zero = 0;
    // Per-level quant-index histograms (for zero-order entropy / byte estimate).
    std::vector<std::map<int,size_t>> lvl_hist(num_levels + 1);

    for (const auto& e : plan) {
        double pred = 0;
        for (size_t k = 0; k < e.pred_ids.size(); k++)
            pred += e.weights[k] * dec_data[e.pred_ids[k]];

        double res = (double)data[e.global_id] - pred;
        lvl_sumsq[e.level] += res * res;
        lvl_maxabs[e.level] = std::max(lvl_maxabs[e.level], std::fabs(res));
        lvl_count[e.level]++;
        for (double w : e.weights) if (w < 0) { lvl_neg[e.level]++; break; }
        if (e.level == 0) {
            size_t nn = e.pred_ids.size();
            base_nbr_sum += nn;
            base_nbr_min  = std::min(base_nbr_min, nn);
            base_nbr_max  = std::max(base_nbr_max, nn);
            if (nn == 0) base_zero++;
        }

        int qind = quantizer.quantize_and_overwrite(dec_data[e.global_id], (T)pred);
        quant_inds.push_back(qind);
        lvl_hist[e.level][qind]++;
    }
    lap("predict+quantize");

    // ------------------------------------------------------------------
    // Per-level residual + estimated-byte report (level 0 = base, L =
    // removed-at-level-L, finest level last).  Bytes ~ zero-order entropy
    // of that level's quant indices (a Huffman lower bound).
    // ------------------------------------------------------------------
    printf("\n%-6s %9s %12s %12s %11s %7s %14s\n",
           "level", "count", "RMS(%range)", "Max(%range)", "est.bytes", "%total", "penalty!=0");
    double total_bytes = 0;
    std::vector<double> lvl_bytes(num_levels + 1, 0.0);
    for (int L = 0; L <= num_levels; L++) {
        if (lvl_count[L] == 0) continue;
        double H = 0;                                  // entropy in bits/symbol
        for (auto& kv : lvl_hist[L]) {
            double p = (double)kv.second / lvl_count[L];
            H -= p * std::log2(p);
        }
        lvl_bytes[L] = H * lvl_count[L] / 8.0;
        total_bytes += lvl_bytes[L];
    }
    for (int L = 0; L <= num_levels; L++) {
        if (lvl_count[L] == 0) continue;
        double rms = std::sqrt(lvl_sumsq[L] / lvl_count[L]);
        const char* tag = (L == 0) ? "base" : "";
        printf("%-2d%-4s %9zu %12.4f %12.4f %11.0f %6.1f%% %8zu (%4.1f%%)\n",
               L, tag, lvl_count[L],
               100.0 * rms / valid_range, 100.0 * lvl_maxabs[L] / valid_range,
               lvl_bytes[L], 100.0 * lvl_bytes[L] / total_bytes,
               lvl_neg[L], 100.0 * lvl_neg[L] / lvl_count[L]);
    }
    if (lvl_count[0] > 0)
        printf("base IDW neighbors: min=%zu max=%zu mean=%.2f  (zero-neighbor nodes: %zu)\n",
               base_nbr_min, base_nbr_max,
               (double)base_nbr_sum / lvl_count[0], base_zero);
    printf("\n");

    // Report reconstruction error (within eb by construction).
    double sq = 0, max_err = 0;
    for (size_t i = 0; i < num_elements; i++) {
        double err = (double)dec_data[i] - (double)data[i];
        sq += err * err;
        if (fabs(err) > max_err) max_err = fabs(err);
    }

    // ------------------------------------------------------------------
    // Serialize: header + quantizer + Huffman + zstd  (same order as Method 2).
    // ------------------------------------------------------------------
    t_prev = Clock::now();                 // exclude the diagnostic report above
    size_t cap = (size_t)num_elements * sizeof(T) * 4 + 4096;
    unsigned char* compressed = (unsigned char*)malloc(cap);
    unsigned char* pos = compressed;

    write(num_levels, pos);
    write(hierarchy, pos);
    write(anchor_metric, pos);
    write(keep_ratio, pos);
    write(eb, pos);
    size_t ne = num_elements;
    write(ne, pos);
    size_t num_quant_inds = quant_inds.size();
    write(num_quant_inds, pos);
    quantizer.save(pos);

    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.preprocess_encode(quant_inds, 0);
    encoder.save(pos);
    encoder.encode(quant_inds, pos);
    encoder.postprocess_encode();
    lap("huffman encode");

    auto lossless = SZ3::Lossless_zstd();
    size_t before_lossless = pos - compressed;
    size_t compressed_size = 0;
    unsigned char* lossless_data = lossless.compress(compressed, before_lossless, compressed_size);
    lossless.postcompress_data(compressed);
    lap("zstd lossless");

    writefile((data_file + ".umc").c_str(), lossless_data, compressed_size);
    free(lossless_data);

    // ------------------------------------------------------------------
    // Report.
    // ------------------------------------------------------------------
    double mse = sq / num_elements;
    printf("size before lossless = %zu\n", before_lossless);
    printf("compressed_size = %zu\n", compressed_size);
    printf("compression_ratio = %.4f\n",
           (double)num_elements * sizeof(T) / compressed_size);
    printf("Max error = %.6g  (%.4f%% of range)\n",
           max_err, 100.0 * max_err / valid_range);
    printf("MSE = %.6g  PSNR = %.4f\n", mse, 20 * log10(valid_range / sqrt(mse)));
    return 0;
}
