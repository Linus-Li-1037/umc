#ifndef _UMC_PARAMETERIZATION_HPP
#define _UMC_PARAMETERIZATION_HPP

#include <vector>
#include <unordered_map>
#include <array>
#include <cmath>
#include <limits>
#include <algorithm>

namespace UMC {

// Parameterization entry for one removed vertex:
//   tri_coarse: the 3 coarse-mesh vertex indices of the anchor triangle
//   bary:       corresponding barycentric weights
struct BarycentricParam {
    std::array<int, 3>    tri_coarse;
    std::array<double, 3> bary;
};

namespace detail {

// Returns vertex idx as a 3-D point; pads z=0 when d==2.
inline std::array<double, 3> get_pos3(const double* V, int d, int idx)
{
    return { V[idx*d], V[idx*d+1], d > 2 ? V[idx*d+2] : 0.0 };
}

// Barycentric coordinates of p projected into the plane of triangle (a,b,c).
// Returns {1,0,0} for degenerate (near-zero-area) triangles.
inline std::array<double, 3> barycentric(
    const std::array<double,3>& p,
    const std::array<double,3>& a,
    const std::array<double,3>& b,
    const std::array<double,3>& c)
{
    double v0[3] = { b[0]-a[0], b[1]-a[1], b[2]-a[2] };
    double v1[3] = { c[0]-a[0], c[1]-a[1], c[2]-a[2] };
    double v2[3] = { p[0]-a[0], p[1]-a[1], p[2]-a[2] };
    double d00 = v0[0]*v0[0] + v0[1]*v0[1] + v0[2]*v0[2];
    double d01 = v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2];
    double d11 = v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2];
    double d20 = v2[0]*v0[0] + v2[1]*v0[1] + v2[2]*v0[2];
    double d21 = v2[0]*v1[0] + v2[1]*v1[1] + v2[2]*v1[2];
    double denom = d00*d11 - d01*d01;
    if (std::abs(denom) < 1e-14) return { 1.0, 0.0, 0.0 };
    double beta  = (d11*d20 - d01*d21) / denom;
    double gamma = (d00*d21 - d01*d20) / denom;
    return { 1.0 - beta - gamma, beta, gamma };
}

// Cyclic one-ring of the neighbors of vertex v in the triangulation.
// conn: flat face array, 3 ints per triangle.
// Returns an empty vector for boundary or non-manifold vertices.
inline std::vector<int> ordered_link(
    int                        v,
    const std::vector<int>&    incident_faces,
    const int*                 conn)
{
    // For each incident face, identify the two other vertices (n1, n2) in
    // winding order after v, and record next_in_link[n1] = n2.
    std::unordered_map<int,int> next_in_link;
    next_in_link.reserve(incident_faces.size());

    for (int fi : incident_faces) {
        int tv[3] = { conn[fi*3], conn[fi*3+1], conn[fi*3+2] };
        int n1, n2;
        if      (tv[0] == v) { n1 = tv[1]; n2 = tv[2]; }
        else if (tv[1] == v) { n1 = tv[2]; n2 = tv[0]; }
        else                  { n1 = tv[0]; n2 = tv[1]; }
        if (next_in_link.count(n1)) return {};  // non-manifold edge
        next_in_link[n1] = n2;
    }
    if (next_in_link.empty()) return {};

    int start = next_in_link.begin()->first;
    std::vector<int> ring;
    ring.reserve(next_in_link.size());
    ring.push_back(start);
    int cur = start;
    while (true) {
        auto it = next_in_link.find(cur);
        if (it == next_in_link.end()) return {};  // boundary vertex
        int nxt = it->second;
        if (nxt == start) break;
        ring.push_back(nxt);
        cur = nxt;
        if (ring.size() > next_in_link.size()) return {};  // cycle guard
    }
    if (ring.size() != next_in_link.size()) return {};
    return ring;
}

} // namespace detail

// For every removed vertex, find the fan triangle (in the CURRENT fine mesh)
// whose barycentric coordinates best contain it, then map the anchor vertices
// through old_to_new to obtain coarse-mesh indices.
//
// V            flat vertex positions: num_vertices * d doubles
// d            spatial dimension stored in V (2 or 3; z is padded to 0 when d==2)
// num_vertices N
// conn         flat triangle connectivity: num_faces * 3 ints
// num_faces    M
// removed_mask true for each vertex that has been decimated away
// old_to_new   coarse-mesh index for each surviving vertex (-1 for removed ones)
//
// Returns: map  fine_vertex_index -> BarycentricParam
// all_triples: if false, only the fan triangles (ring[0], ring[i], ring[i+1])
//   are considered (O(k) per vertex).  If true, every triple of ring vertices
//   is tried (O(k^3)) — a strictly larger candidate set.
// metric (only used when all_triples): how to pick among the candidate triangles
//   0 = min outside-penalty, tie-break smaller area (legacy).
//   1 = MAX min-weight: maximize min(w0,w1,w2).  A single criterion that prefers
//       the most-interior point (all 3 anchors genuinely used, weights far from 0)
//       > edge points (a weight = 0, degenerate to 2-point linear) > extrapolation
//       (a weight < 0).  Tie-break smaller area.
//   2 = smallest area among STRICTLY-interior triangles (min-weight > eps, i.e.
//       true 3-point barycentric); falls back to max min-weight if none qualify.
inline std::unordered_map<int, BarycentricParam> build_parameterization(
    const double*             V,
    int                       d,
    int                       num_vertices,
    const int*                conn,
    int                       num_faces,
    const std::vector<bool>&  removed_mask,
    const std::vector<int>&   old_to_new,
    bool                      all_triples = false,
    int                       metric      = 0)
{
    // Per-vertex list of incident face indices.
    std::vector<std::vector<int>> incident(num_vertices);
    for (int fi = 0; fi < num_faces; fi++)
        for (int j = 0; j < 3; j++)
            incident[conn[fi*3+j]].push_back(fi);

    std::unordered_map<int, BarycentricParam> result;
    result.reserve(num_vertices / 4);

    for (int v = 0; v < num_vertices; v++) {
        if (!removed_mask[v]) continue;

        auto ring = detail::ordered_link(v, incident[v], conn);
        int  k = (int)ring.size();
        if (k < 3) continue;

        auto pv = detail::get_pos3(V, d, v);

        int    ra = -1, rb = -1, rc = -1;     // winning ring indices
        std::array<double,3> best_b{};

        if (all_triples) {
            const double EPS = 1e-9;
            double best_minw = -std::numeric_limits<double>::max();
            double best_area =  std::numeric_limits<double>::max();
            for (int i = 0; i < k; i++)
                for (int j = i+1; j < k; j++)
                    for (int m = j+1; m < k; m++) {
                        auto A = detail::get_pos3(V, d, ring[i]);
                        auto B = detail::get_pos3(V, d, ring[j]);
                        auto C = detail::get_pos3(V, d, ring[m]);
                        auto b3 = detail::barycentric(pv, A, B, C);
                        double minw = std::min({ b3[0], b3[1], b3[2] });
                        double ux=B[0]-A[0], uy=B[1]-A[1], vx=C[0]-A[0], vy=C[1]-A[1];
                        double area = 0.5 * std::fabs(ux*vy - uy*vx);

                        bool better;
                        if (metric == 2) {                  // smallest area, strictly interior
                            bool ci = minw > EPS, bi = best_minw > EPS;
                            if      (ci && !bi)  better = true;
                            else if (ci &&  bi)  better = area < best_area;
                            else if (!ci && !bi) better = minw > best_minw;   // least extrapolating
                            else                 better = false;
                        } else {                            // metric 0/1: maximize a key, tie-break area
                            double key  = (metric == 1) ? minw : std::min(minw, 0.0);
                            double bkey = (metric == 1) ? best_minw : std::min(best_minw, 0.0);
                            better = (key > bkey + 1e-12) ||
                                     (key > bkey - 1e-12 && area < best_area);
                        }
                        if (better) {
                            best_minw = minw; best_area = area;
                            ra = i; rb = j; rc = m; best_b = b3;
                        }
                    }
        } else {
            auto a = detail::get_pos3(V, d, ring[0]);
            double best_pen = std::numeric_limits<double>::max();
            for (int i = 1; i + 1 < k; i++) {
                auto b3 = detail::barycentric(
                    pv, a,
                    detail::get_pos3(V, d, ring[i]),
                    detail::get_pos3(V, d, ring[i+1]));
                double pen = -std::min({ b3[0], b3[1], b3[2], 0.0 });
                if (pen < best_pen) {
                    best_pen = pen; ra = 0; rb = i; rc = i+1; best_b = b3;
                }
            }
        }
        if (ra < 0) continue;

        BarycentricParam param;
        param.tri_coarse = {
            old_to_new[ring[ra]],
            old_to_new[ring[rb]],
            old_to_new[ring[rc]]
        };
        param.bary = best_b;
        result[v]  = param;
    }
    return result;
}

// Adjacency-based overload for meshes whose connectivity is stored as an
// edge list rather than explicit triangle faces (e.g. the Tsunami dataset).
//
// For each removed vertex v, the edge-neighbors in adj[v] serve as the ring.
// Because v was chosen by a maximal independent set, all its edge-neighbors
// survive (old_to_new[u] >= 0 for every u in adj[v]).  We try every triple
// of those neighbors and pick the one whose barycentric triangle best contains
// v's position (minimum outside-penalty), exactly as the face-based version
// does over the fan triangles.
//
// V            flat vertex positions: num_vertices * d doubles
// d            spatial dimension stored in V (2 or 3; z is padded to 0 when d==2)
// num_vertices N
// adj          symmetric vertex adjacency built from the edge list
// removed_mask true for each vertex that has been decimated away
// old_to_new   coarse-mesh index for each surviving vertex (-1 for removed ones)
//
// Returns: map  fine_vertex_index -> BarycentricParam
inline std::unordered_map<int, BarycentricParam> build_parameterization(
    const double*                         V,
    int                                   d,
    int                                   num_vertices,
    const std::vector<std::set<int32_t>>& adj,
    const std::vector<bool>&              removed_mask,
    const std::vector<int>&               old_to_new)
{
    std::unordered_map<int, BarycentricParam> result;
    result.reserve(num_vertices / 4);

    for (int v = 0; v < num_vertices; v++) {
        if (!removed_mask[v]) continue;

        // Collect surviving neighbors (all survive by the MIS property).
        std::vector<int> nbrs;
        nbrs.reserve(adj[v].size());
        for (int32_t u : adj[v])
            if (!removed_mask[u]) nbrs.push_back(u);
        if ((int)nbrs.size() < 3) continue;

        auto pv = detail::get_pos3(V, d, v);

        int    best_i = -1, best_j = -1, best_k = -1;
        double best_pen = std::numeric_limits<double>::max();
        std::array<double,3> best_b{};

        for (int i = 0; i < (int)nbrs.size(); i++) {
            auto pi = detail::get_pos3(V, d, nbrs[i]);
            for (int j = i+1; j < (int)nbrs.size(); j++) {
                auto pj = detail::get_pos3(V, d, nbrs[j]);
                for (int k = j+1; k < (int)nbrs.size(); k++) {
                    auto b3 = detail::barycentric(
                        pv, pi, pj, detail::get_pos3(V, d, nbrs[k]));
                    double pen = -std::min({ b3[0], b3[1], b3[2], 0.0 });
                    if (pen < best_pen) {
                        best_pen = pen;
                        best_i = i; best_j = j; best_k = k;
                        best_b  = b3;
                    }
                }
            }
        }
        if (best_i < 0) continue;

        BarycentricParam param;
        param.tri_coarse = {
            old_to_new[nbrs[best_i]],
            old_to_new[nbrs[best_j]],
            old_to_new[nbrs[best_k]]
        };
        param.bary = best_b;
        result[v]  = param;
    }
    return result;
}

// ---------------------------------------------------------------------------
// IDW (Inverse Distance Weighted) parameterization
// ---------------------------------------------------------------------------

// Parameterization entry for one removed vertex using IDW:
//   neighbors: coarse-mesh indices of all adjacent surviving vertices
//   weights:   1/distance weights, normalised to sum = 1
struct IDWParam {
    std::vector<int>    neighbors;
    std::vector<double> weights;
};

// For every removed vertex v, collect its adjacent surviving vertices from
// adj[v] and assign normalised 1/distance weights.
//
// Because the MIS was computed on the same adjacency, every u in adj[v] is
// guaranteed to be a surviving vertex (old_to_new[u] >= 0).
//
// V            flat vertex positions: num_vertices * d doubles
// d            spatial dimension (2 or 3)
// num_vertices N
// adj          symmetric adjacency used for the MIS
// removed_mask true for each decimated vertex
// old_to_new   coarse-mesh index per surviving vertex (-1 for removed ones)
//
// Returns: map  fine_vertex_index -> IDWParam
inline std::unordered_map<int, IDWParam> build_idw_parameterization(
    const double*                         V,
    int                                   d,
    int                                   num_vertices,
    const std::vector<std::set<int32_t>>& adj,
    const std::vector<bool>&              removed_mask,
    const std::vector<int>&               old_to_new)
{
    std::unordered_map<int, IDWParam> result;
    result.reserve(num_vertices / 4);

    for (int v = 0; v < num_vertices; v++) {
        if (!removed_mask[v]) continue;

        auto pv = detail::get_pos3(V, d, v);
        IDWParam param;
        double weight_sum = 0.0;

        for (int32_t u : adj[v]) {
            if (removed_mask[u]) continue;  // safety — should not occur in MIS
            auto pu = detail::get_pos3(V, d, u);
            double dx = pv[0]-pu[0], dy = pv[1]-pu[1], dz = pv[2]-pu[2];
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist < 1e-14) dist = 1e-14;  // guard against coincident points
            double w = 1.0 / dist;
            param.neighbors.push_back(old_to_new[u]);
            param.weights.push_back(w);
            weight_sum += w;
        }
        if (param.neighbors.empty()) continue;

        // Normalise so weights sum to 1.
        for (double& w : param.weights) w /= weight_sum;

        result[v] = std::move(param);
    }
    return result;
}

} // namespace UMC
#endif
