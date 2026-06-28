// Multilevel barycentric-prediction decompressor (any triangulated surface).
//
// Counterpart to test_compress_barycentric.  Rebuilds the identical multilevel
// MIS + barycentric hierarchy from geometry, then recovers each vertex value in
// the same coarse->fine plan order using the stored quantized residuals.
//
// Usage:
//   test_decompress_barycentric <data_dir> <coord_file> <dim> <tri_file> <var_file>
//   (geometry args must match the compressor)
// Reads : <data_dir>/<var_file>.umc (+ <var_file> original, for error stats)
// Writes: <data_dir>/<var_file>.umc.out

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

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
    if (argc < 6) {
        fprintf(stderr,
            "Usage: %s <data_dir> <coord_file> <dim> <tri_file> <var_file>\n"
            "  (geometry args must match the compressor so the plan is rebuilt identically)\n",
            argv[0]);
        return 1;
    }
    std::string data_dir(argv[1]);
    if (data_dir.back() != '/') data_dir += '/';
    std::string coord_file = argv[2];
    int    dim         = std::atoi(argv[3]);
    std::string tri_file   = argv[4];
    std::string var_file   = argv[5];
    std::string data_file       = data_dir + var_file;
    std::string compressed_file = data_file + ".umc";

    // ------------------------------------------------------------------
    // Load geometry (identical to the compressor).
    // ------------------------------------------------------------------
    size_t num = 0;
    auto coords_f = readfile<float>((data_dir + coord_file).c_str(), num);
    int num_vertices = (int)(num / dim);
    std::vector<double> V(num_vertices * 3, 0.0);
    for (int i = 0; i < num_vertices; i++)
        for (int c = 0; c < dim; c++)
            V[i*3+c] = coords_f[i*dim+c];
    auto faces_i = readfile<int>((data_dir + tri_file).c_str(), num);
    std::vector<int> faces(faces_i.begin(), faces_i.end());

    // ------------------------------------------------------------------
    // Decode stream: zstd -> header -> quantizer -> Huffman.
    // ------------------------------------------------------------------
    size_t input_size = 0;
    auto input = readfile<unsigned char>(compressed_file.c_str(), input_size);
    auto lossless = SZ3::Lossless_zstd();
    size_t remaining_length = input_size;
    auto compressed_data = lossless.decompress(input.data(), input_size);
    const unsigned char* pos = compressed_data;

    int num_levels = 0;
    read(num_levels, pos);
    int hierarchy = 0;
    read(hierarchy, pos);
    int anchor_metric = 0;
    read(anchor_metric, pos);
    double keep_ratio = 0.5;
    read(keep_ratio, pos);
    double eb = 0;
    read(eb, pos);
    size_t num_elements = 0;
    read(num_elements, pos);
    size_t num_quant_inds = 0;
    read(num_quant_inds, pos);

    const int quant_radius = 32768;
    SZ3::LinearQuantizer<T> quantizer(eb, quant_radius);
    quantizer.load(pos, remaining_length);
    auto encoder = SZ3::HuffmanEncoder<int>();
    encoder.load(pos, remaining_length);
    std::vector<int> quant_inds = encoder.decode(pos, num_quant_inds);
    encoder.postprocess_decode();
    lossless.postdecompress_data(compressed_data);

    const char* hname = hierarchy == 0 ? "MIS+barycentric"
                      : hierarchy == 1 ? "clustering+barycentric"
                      : hierarchy == 2 ? "edge-collapse+barycentric"
                      : hierarchy == 3 ? "forest-collapse+barycentric"
                      : "PFS-collapse+barycentric";
    printf("vertices=%d  levels=%d  hierarchy=%s  abs_eb=%.6g  quant_inds=%zu\n",
           num_vertices, num_levels, hname, eb, num_quant_inds);

    // ------------------------------------------------------------------
    // Rebuild the plan and recover values (coarse -> fine).
    // ------------------------------------------------------------------
    std::vector<HierEntry> plan;
    if (hierarchy == 0)
        plan = build_barycentric_hierarchy(V, faces, num_vertices, num_levels,
                                           anchor_metric > 0, anchor_metric);
    else if (hierarchy == 1)
        plan = build_cluster_hierarchy(V, num_vertices, num_levels, keep_ratio,
                                       median_edge_length(V, faces), false);
    else if (hierarchy == 2)
        plan = build_edge_collapse_hierarchy(V, faces, num_vertices, num_levels, keep_ratio);
    else if (hierarchy == 3)
        plan = build_forest_collapse_hierarchy(V, faces, num_vertices, num_levels);
    else
        plan = build_pfs_collapse_hierarchy(V, faces, num_vertices, num_levels);
    if (plan.size() != num_quant_inds) {
        fprintf(stderr, "ERROR: plan size %zu != quant_inds %zu (hierarchy mismatch)\n",
                plan.size(), num_quant_inds);
        return 1;
    }

    std::vector<T> dec_data(num_elements, 0);
    const int* qi = quant_inds.data();
    for (const auto& e : plan) {
        double pred = 0;
        for (size_t k = 0; k < e.pred_ids.size(); k++)
            pred += e.weights[k] * dec_data[e.pred_ids[k]];
        dec_data[e.global_id] = quantizer.recover((T)pred, *(qi++));
    }

    writefile((compressed_file + ".out").c_str(), dec_data.data(), num_elements);

    // ------------------------------------------------------------------
    // Error statistics against the original.
    // ------------------------------------------------------------------
    size_t ori_num = 0;
    auto data_ori = readfile<T>(data_file.c_str(), ori_num);
    if (ori_num == num_elements)
        print_statistics(data_ori.data(), dec_data.data(), num_elements);

    return 0;
}
