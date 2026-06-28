// 3-D tetrahedral-mesh decompressor: rebuilds the identical edge-collapse + tet-
// barycentric hierarchy and recovers each vertex value from the stored residuals.
//
// Usage: test_decompress_tet <data_dir> <coord_file> <conn_file> <var_file>
//        (geometry args must match the compressor)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#include <SZ3/quantizer/IntegerQuantizer.hpp>
#include <SZ3/encoder/HuffmanEncoder.hpp>
#include <SZ3/lossless/Lossless_zstd.hpp>

#include "utils.hpp"
#include "tet_collapse_hierarchy.hpp"

using namespace UMC;
using T = float;

int main(int argc, char** argv)
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <data_dir> <coord_file> <conn_file> <var_file>\n", argv[0]);
        return 1;
    }
    std::string data_dir(argv[1]); if (data_dir.back()!='/') data_dir += '/';
    std::string coord_file=argv[2], conn_file=argv[3], var_file=argv[4];
    std::string data_file=data_dir+var_file, comp_file=data_file+".umc";

    // Read the compressed header first: dim (spatial dimension) is stored there,
    // so the decompressor reconstructs geometry exactly as the compressor did.
    size_t input_size=0;
    auto input = readfile<unsigned char>(comp_file.c_str(), input_size);
    auto lossless=SZ3::Lossless_zstd();
    size_t remaining=input_size;
    auto cdata=lossless.decompress(input.data(), input_size);
    const unsigned char* pos=cdata;

    int from_edge=0; read(from_edge, pos);
    int dim=3; read(dim,pos);
    int num_levels=0; read(num_levels,pos);
    double eb=0; read(eb,pos);
    size_t num_elements=0; read(num_elements,pos);
    size_t nqi=0; read(nqi,pos);

    int num_vertices=(int)num_elements;          // one scalar per vertex
    size_t ncoord=0;
    auto coords_f = readfile<float>((data_dir+coord_file).c_str(), ncoord);
    int cdim=(int)(ncoord/num_vertices);         // coord stride (embedding dim), padded to 3
    std::vector<double> V(num_vertices*3, 0.0);
    for (int i=0;i<num_vertices;i++) for (int c=0;c<cdim;c++) V[i*3+c]=coords_f[i*cdim+c];
    size_t nconn=0;
    auto conn_i = readfile<int>((data_dir+conn_file).c_str(), nconn);
    std::vector<int> tets(conn_i.begin(), conn_i.end());

    SZ3::LinearQuantizer<T> quantizer(eb,32768);
    quantizer.load(pos,remaining);
    auto encoder=SZ3::HuffmanEncoder<int>();
    encoder.load(pos,remaining);
    std::vector<int> quant_inds=encoder.decode(pos,nqi);
    encoder.postprocess_decode();
    lossless.postdecompress_data(cdata);

    printf("verts=%d  dim=%d  levels=%d abs_eb=%.6g  quant_inds=%zu\n",
           num_vertices, dim, num_levels, eb, nqi);

    auto plan = build_tet_mis_hierarchy(V, tets, num_vertices, num_levels, dim, from_edge);
    if (plan.size()!=nqi){ fprintf(stderr,"ERROR plan %zu != qi %zu\n", plan.size(), nqi); return 1; }

    std::vector<T> dec_data(num_elements,0);
    const int* qi=quant_inds.data();
    for (const auto& e : plan) {
        double pred=0;
        for (size_t k=0;k<e.pred_ids.size();k++) pred += e.weights[k]*dec_data[e.pred_ids[k]];
        dec_data[e.global_id]=quantizer.recover((T)pred,*(qi++));
    }
    writefile((comp_file+".out").c_str(), dec_data.data(), num_elements);

    size_t on=0; auto ori=readfile<T>(data_file.c_str(), on);
    if (on==num_elements) print_statistics(ori.data(), dec_data.data(), num_elements);
    return 0;
}
