// 3-D tetrahedral-mesh compressor: multilevel edge-collapse + tet-barycentric
// prediction (LES velocity).  Counterpart of test_compress_barycentric for
// volumetric tet meshes.  Quantize -> Huffman -> zstd; the hierarchy is rebuilt
// deterministically by the decompressor, so only residuals are stored.
//
// Usage: test_compress_tet <data_dir> <coord_file> <conn_file> <var_file>
//                          <num_levels> <abs_eb_or_not> <eb> [keep_ratio=0.5]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

#include <SZ3/quantizer/IntegerQuantizer.hpp>
#include <SZ3/encoder/HuffmanEncoder.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>

#include "utils.hpp"
#include "tet_collapse_hierarchy.hpp"

using namespace UMC;
using T = float;

int main(int argc, char** argv)
{
    if (argc < 8) {
        fprintf(stderr, "Usage: %s <data_dir> <coord_file> <conn_file> <var_file> "
                "<num_levels> <abs_eb_or_not> <eb> [keep_ratio=0.5] [hierarchy=2] [dim=3]\n"
                "  hierarchy: 0 = MIS, 2 = edge-collapse (keep_ratio)\n"
                "  dim:       3 = tet barycentric, 2 = triangle barycentric (IDW fallback)\n", argv[0]);
        return 1;
    }
    int argv_id = 1;
    std::string data_dir(argv[argv_id++]); if (data_dir.back()!='/') data_dir += '/';
    std::string coord_file=argv[argv_id++], conn_file=argv[argv_id++], var_file=argv[argv_id++];
    int    from_edge  = std::atoi(argv[argv_id++]);
    int    dim        = std::atoi(argv[argv_id++]);
    int    num_levels = std::atoi(argv[argv_id++]);
    bool   use_abs_eb = std::atoi(argv[argv_id++]) != 0;
    double eb         = std::atof(argv[argv_id++]);
    if (num_levels < 1) num_levels = 1;
    if (dim != 2 && dim != 3) { fprintf(stderr,"ERROR: dim must be 2 or 3\n"); return 1; }

    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    auto lap = [&](const char* n){ auto t=Clock::now();
        printf("[time] %-18s %.3f s\n", n, std::chrono::duration<double>(t-t0).count()); t0=t; };

    std::string data_file = data_dir+var_file;
    size_t num_elements=0;
    auto data = readfile<T>(data_file.c_str(), num_elements);
    int num_vertices = (int)num_elements;        // one scalar per vertex

    // Coordinates: the file's stride (floats per vertex) is the geometric
    // embedding dimension (2 or 3), derived from the data — independent of the
    // mesh dimension `dim` (2=triangle, 3=tet).  Always store V 3-wide, padding
    // z=0 for planar meshes, so the predictor's geometry is dimension-uniform.
    size_t ncoord=0;
    auto coords_f = readfile<float>((data_dir+coord_file).c_str(), ncoord);
    int cdim = (int)(ncoord / num_vertices);
    if (cdim < 2 || cdim > 3 || (size_t)num_vertices*cdim != ncoord) {
        fprintf(stderr,"ERROR: coords %zu not 2 or 3 per vert %d\n", ncoord, num_vertices); return 1;
    }
    std::vector<double> V(num_vertices*3, 0.0);
    for (int i=0;i<num_vertices;i++) for (int c=0;c<cdim;c++) V[i*3+c]=coords_f[i*cdim+c];

    size_t nconn=0;
    auto conn_i = readfile<int>((data_dir+conn_file).c_str(), nconn);
    std::vector<int> tets(conn_i.begin(), conn_i.end());

    T max_val=*std::max_element(data.begin(),data.end());
    T min_val=*std::min_element(data.begin(),data.end());
    double valid_range=(double)(max_val-min_val);
    if (!use_abs_eb) eb *= valid_range;
    printf("verts=%d  cells=%zu  mesh_dim=%d  coord_dim=%d  range=%.4f  abs_eb=%.6g  levels=%d\n",
           num_vertices, tets.size()/4, dim, cdim, valid_range, eb, num_levels);
    lap("load");

    auto plan = build_tet_mis_hierarchy(V, tets, num_vertices, num_levels, dim, from_edge);
    printf("plan entries = %zu (== verts: %s)\n", plan.size(),
           plan.size()==(size_t)num_vertices ? "yes":"NO");
    lap("build hierarchy");

    std::vector<T> dec_data(data);
    SZ3::LinearQuantizer<T> quantizer(eb, 32768);
    std::vector<int> quant_inds; quant_inds.reserve(num_elements);
    // per-level prediction-residual stats, split by predictor kind:
    //   k=0 enclosed tet, k=1 outside hull, k=2 IDW, k=3 base layer (level 0)
    auto kind_idx=[&](const HierEntry& e){
        if (e.level==0) return 3;                     // base layer
        return (e.pred_kind>=0 && e.pred_kind<=2) ? e.pred_kind : 1;
    };
    std::vector<std::array<double,4>> lvl_sq(num_levels+1, {0,0,0,0});  // sum sq residual
    std::vector<std::array<size_t,4>> lvl_cnt(num_levels+1, {0,0,0,0}); // count
    std::vector<std::array<double,4>> lvl_max(num_levels+1, {0,0,0,0}); // max |residual|
    for (const auto& e : plan) {
        double pred=0;
        for (size_t k=0;k<e.pred_ids.size();k++) pred += e.weights[k]*dec_data[e.pred_ids[k]];
        double resid=(double)dec_data[e.global_id]-pred;     // pre-quantization prediction error
        int L = (e.level<=num_levels) ? e.level : num_levels;
        int kd = kind_idx(e);
        lvl_sq[L][kd]+=resid*resid; lvl_cnt[L][kd]++;
        if (std::fabs(resid)>lvl_max[L][kd]) lvl_max[L][kd]=std::fabs(resid);
        quant_inds.push_back(quantizer.quantize_and_overwrite(dec_data[e.global_id],(T)pred));
    }
    double sq=0,max_err=0;
    for (size_t i=0;i<num_elements;i++){ double er=(double)dec_data[i]-(double)data[i]; sq+=er*er; if(std::fabs(er)>max_err)max_err=std::fabs(er);}
    lap("predict+quantize");

    size_t cap=(size_t)num_elements*sizeof(T)*4+4096;
    unsigned char* compressed=(unsigned char*)malloc(cap); unsigned char* pos=compressed;
    write(from_edge, pos); write(dim,pos); write(num_levels,pos); write(eb,pos);
    size_t ne=num_elements; write(ne,pos);
    size_t nqi=quant_inds.size(); write(nqi,pos);
    quantizer.save(pos);
    auto encoder=SZ3::HuffmanEncoder<int>();
    encoder.preprocess_encode(quant_inds,0); encoder.save(pos); encoder.encode(quant_inds,pos); encoder.postprocess_encode();
    auto lossless=SZ3::Lossless_zstd();
    size_t before=pos-compressed, compressed_size=0;
    unsigned char* out=lossless.compress(compressed, before, compressed_size);
    lossless.postcompress_data(compressed);
    writefile((data_file+".umc").c_str(), out, compressed_size);
    free(out);
    lap("encode+zstd");

    // per-level prediction-residual breakdown, split by predictor kind: tells
    // whether error comes from the base IDW, the enclosing-tet predictor, the
    // outside-hull (zero-prediction) cases, or the <4-candidate IDW fallback.
    const char* knm[4] = {"enclosed", "outside ", "idw     ", "base    "};
    printf("---- per-level prediction residual RMSE (pre-quantization) ----\n");
    for (int L=0; L<=num_levels; L++) {
        for (int k=0;k<4;k++) {
            if (lvl_cnt[L][k]==0) continue;
            double rmse = std::sqrt(lvl_sq[L][k]/lvl_cnt[L][k]);
            printf("  level %d  %s: n=%-8zu  RMSE=%.6g  max=%.6g  (%.4f%% range)\n",
                   L, knm[k], lvl_cnt[L][k], rmse, lvl_max[L][k],
                   100.0*lvl_max[L][k]/valid_range);
        }
    }
    printf("---------------------------------------------------------------\n");

    double mse=sq/num_elements;
    printf("compressed_size = %zu\n", compressed_size);
    printf("compression_ratio = %.4f\n", (double)num_elements*sizeof(T)/compressed_size);
    printf("Max error = %.6g  (%.4f%% of range)\n", max_err, 100.0*max_err/valid_range);
    printf("MSE = %.6g  PSNR = %.4f\n", mse, 20*log10(valid_range/sqrt(mse)));
    return 0;
}
