#ifndef _UMC_TET_COLLAPSE_HIERARCHY_HPP
#define _UMC_TET_COLLAPSE_HIERARCHY_HPP

#include <vector>
#include <set>
#include <queue>
#include <tuple>
#include <array>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstdio>

#include "parameterization.hpp"        // detail::get_pos3
#include "barycentric_hierarchy.hpp"   // HierEntry
#include "reorder.hpp"                  // DPFS
#include "adjacent_prediction.hpp"

namespace UMC {

namespace detail {

// 3-D barycentric weights of p in tet (a,b,c,d): p = sum wi*vi, sum wi = 1.
// All wi >= 0 iff p is inside the tet.
inline std::array<double,4> barycentric_tet(
    const std::array<double,3>& p, const std::array<double,3>& a,
    const std::array<double,3>& b, const std::array<double,3>& c,
    const std::array<double,3>& d)
{
    double m00=b[0]-a[0], m01=c[0]-a[0], m02=d[0]-a[0];
    double m10=b[1]-a[1], m11=c[1]-a[1], m12=d[1]-a[1];
    double m20=b[2]-a[2], m21=c[2]-a[2], m22=d[2]-a[2];
    double r0=p[0]-a[0], r1=p[1]-a[1], r2=p[2]-a[2];
    double det = m00*(m11*m22-m12*m21) - m01*(m10*m22-m12*m20) + m02*(m10*m21-m11*m20);
    if (std::fabs(det) < 1e-30) return {1.0,0.0,0.0,0.0};
    double id = 1.0/det;
    double beta  = (r0*(m11*m22-m12*m21) - m01*(r1*m22-m12*r2) + m02*(r1*m21-m11*r2))*id;
    double gamma = (m00*(r1*m22-m12*r2) - r0*(m10*m22-m12*m20) + m02*(m10*r2-r1*m20))*id;
    double delta = (m00*(m11*r2-r1*m21) - m01*(m10*r2-r1*m20) + r0*(m10*m21-m11*m20))*id;
    return {1.0-beta-gamma-delta, beta, gamma, delta};
}

// Vertex-vertex adjacency from a cell list, verts_per_cell ints per cell (4 for a
// tet, also 4 for the Katrina 2-D triangle format (v0,v0,v1,v2) where the leading
// vertex is duplicated).  All distinct vertex pairs of a cell become edges; the
// id_i != id_j guard drops the self-loop produced by the duplicated vertex, so a
// degenerate "tet" collapses to exactly its triangle's three edges.
inline std::vector<std::set<int>> tet_vertex_adjacency(
    const std::vector<int>& tets, int N, int verts_per_cell = 4)
{
    std::vector<std::set<int>> adj(N);
    int vc = verts_per_cell;
    for (size_t t=0; t+vc<=tets.size(); t+=vc) {
        for (int i=0;i<vc;i++) for (int j=i+1;j<vc;j++) {
            int a=tets[t+i], b=tets[t+j];
            if (a==b) continue;                       // skip duplicated-vertex self loop
            adj[a].insert(b); adj[b].insert(a);
        }
    }
    return adj;
}

// Predict vertex v from its candidate surviving neighbors, selecting the
// interpolation kind automatically from the mesh dimension mdim:
//   mdim==3 -> enclosing TET   (4 anchors, detail::barycentric_tet)
//   mdim==2 -> enclosing TRIANGLE (3 anchors, detail::barycentric, which
//              projects v into the anchor-triangle plane, so it is correct even
//              for a 2-D surface embedded in 3-D, e.g. Katrina lon/lat/elev).
// Positions are always read 3-wide (z padded to 0 for planar data), matching the
// V layout convention.  Among the candidates, pick the (mdim+1)-
// simplex maximizing min(barycentric weight) > 0 — the one that best encloses v
// (genuine (mdim+1)-point interpolation).  Falls back to normalised IDW over all
// candidates when fewer than mdim+1 exist, or when no simplex encloses v
// (v outside the convex hull of its neighbors).
// Return status (diagnostics): 0 = enclosing simplex found (true interpolation),
// 2 = IDW fallback (too few candidates or outside hull).
inline int best_simplex_pred(const std::vector<double>& V, int mdim, int v,
                             std::vector<int>& cand, HierEntry& e)
{
    auto pos=[&](int x){ return detail::get_pos3(V.data(),3,x); };
    auto pv = pos(v);
    auto d2 = [&](int x){ auto px=pos(x); double s=0;
        for (int t=0;t<3;t++){ double dd=pv[t]-px[t]; s+=dd*dd; } return s; };
    // auto idw = [&]()->int {                        // normalised inverse-distance over nearest 4
    //     std::vector<std::pair<double,int>> nd;     // (dist^2, id); pair<> gives a
    //     nd.reserve(cand.size());                   // deterministic id tie-break
    //     for (int u : cand) nd.push_back({d2(u), u});
    //     int num = (4 < (int)nd.size()) ? 4 : (int)nd.size();
    //     std::partial_sort(nd.begin(), nd.begin()+num, nd.end());
    //     double wsum=0;
    //     for (int j=0;j<num;j++) { double w=1.0/std::sqrt(nd[j].first); e.pred_ids.push_back(nd[j].second); e.weights.push_back(w); wsum+=w; }
    //     for (double& w : e.weights) w/=wsum;
    //     return 2;
    // };
    auto idw = [&]()->int {                        // normalised inverse-distance
        double wsum=0;
        for (int u : cand) { double w=1.0/std::sqrt(d2(u)); e.pred_ids.push_back(u); e.weights.push_back(w); wsum+=w; }
        for (double& w : e.weights) w/=wsum;
        return 2;
    };

    const int S = mdim + 1;                        // anchors per simplex
    int m = (int)cand.size();
    if (m < S) return idw();

    double best_minw = 0.0;                         // require strictly enclosing (minw>0)
    bool found=false;
    std::vector<int> bt; std::vector<double> bw;
    if (mdim == 2) {
        for (int i=0;i<m;i++) {
            for (int j=i+1;j<m;j++) {
                for (int k=j+1;k<m;k++) {
                    auto w = detail::barycentric(pv, pos(cand[i]),pos(cand[j]),pos(cand[k]));
                    double minw = std::min({w[0],w[1],w[2]});
                    if (minw > best_minw) { 
                        best_minw=minw;
                        bt={cand[i],cand[j],cand[k]}; 
                        bw={w[0],w[1],w[2]}; 
                        if(minw > 0) found=true; 
                    }
                }
            }
        }    
    } else {  // mdim == 3
        for (int i=0;i<m;i++) {
            for (int j=i+1;j<m;j++) {
                for (int k=j+1;k<m;k++) {
                    for (int l=k+1;l<m;l++) {
                        auto w = detail::barycentric_tet(pv, pos(cand[i]),pos(cand[j]),pos(cand[k]),pos(cand[l]));
                        double minw = std::min({w[0],w[1],w[2],w[3]});
                        if (minw > best_minw) { 
                            best_minw=minw;
                            bt={cand[i],cand[j],cand[k],cand[l]}; 
                            bw={w[0],w[1],w[2],w[3]}; 
                            if(minw > 0) found=true;
                        }
                    }
                }
            }
        }
    }
    if (!found) return idw();                       // outside hull -> IDW
    // if (!found) {
    //     e.pred_ids.assign(bt.begin(), bt.end());
    //     e.weights.assign(bw.begin(), bw.end());
    //     return 1;
    // }
    e.pred_ids.assign(bt.begin(), bt.end());
    e.weights.assign(bw.begin(), bw.end());
    return 0;
}

// Base layer (final survivors): DPFS-ordered causal IDW on the adjacency graph,
// + assemble the full plan (base first, then removed levels coarsest->finest).
inline std::vector<HierEntry> assemble_tet_plan(
    const std::vector<double>& V, int N,
    const std::vector<std::set<int>>& adj, const std::vector<char>& alive,
    std::vector<std::vector<HierEntry>>& level_recs)
{
    auto dist=[&](int a,int b){ double dx=V[a*3]-V[b*3],dy=V[a*3+1]-V[b*3+1],dz=V[a*3+2]-V[b*3+2]; return std::sqrt(dx*dx+dy*dy+dz*dz); };
    std::vector<int> base_globals; std::vector<int> local_of(N,-1);
    for (int v=0; v<N; v++) if (alive[v]) { local_of[v]=(int)base_globals.size(); base_globals.push_back(v); }
    int nb=(int)base_globals.size();
    std::vector<std::set<int32_t>> badj(nb);
    for (int v=0; v<N; v++) if (alive[v]) for (int u : adj[v]) if (alive[u]) badj[local_of[v]].insert(local_of[u]);

    std::vector<HierEntry> plan; plan.reserve(N);
    if (nb>0) {
        auto order=DPFS(badj);
        std::vector<int> rank_to_node(nb);
        for (int c=0;c<nb;c++) rank_to_node[order[c]]=c;
        for (int r=0;r<nb;r++) {
            int c=rank_to_node[r], g=base_globals[c];
            HierEntry e; e.global_id=g; e.level=0; double wsum=0;
            std::vector<std::pair<int, double>> neighbors;
            for (int32_t u : badj[c]) { 
                if (order[u]>=order[c]) continue; 
                int gu=base_globals[u];
                double w=1.0/dist(g,gu); 
                neighbors.push_back(std::make_pair(gu, w));
                std::sort(neighbors.begin(), neighbors.end(), sortByWeight<double>);
                // e.pred_ids.push_back(gu); 
                // e.weights.push_back(w); 
                // wsum+=w; 
            }
            int num = (4 < (int)neighbors.size()) ? 4 : (int)neighbors.size();
            for(int j=0; j<num; j++){
                wsum += neighbors[j].second;
            }
            for(int j=0; j<num; j++){
                e.pred_ids.push_back(neighbors[j].first);
                e.weights.push_back(neighbors[j].second / wsum);
            }
            // for (double& w : e.weights) w/=wsum;
            plan.push_back(std::move(e));
        }
    }
    for (int L=(int)level_recs.size()-1; L>=0; L--)
        for (auto& e : level_recs[L]) plan.push_back(std::move(e));
    return plan;
}

// Base layer (final survivors): per-CLUSTER first-point prediction.  After
// splitting the base adjacency into connected clusters, each cluster is encoded
// relative to its FIRST point: that first point is a root (empty pred -> pred 0,
// stores its absolute value); every other point in the cluster predicts from
// that single first point — pred = first-point value, resid = real - first_point.
// No IDW.  (cluster seed = lowest-index node, matching the DPFS component order.)
inline std::vector<HierEntry> assemble_tet_plan_avg(
    const std::vector<double>& V, int N,
    const std::vector<std::set<int>>& adj, const std::vector<char>& alive,
    std::vector<std::vector<HierEntry>>& level_recs)
{
    std::vector<int> base_globals; std::vector<int> local_of(N,-1);
    for (int v=0; v<N; v++) if (alive[v]) { local_of[v]=(int)base_globals.size(); base_globals.push_back(v); }
    int nb=(int)base_globals.size();
    std::vector<std::set<int32_t>> badj(nb);
    for (int v=0; v<N; v++) if (alive[v]) for (int u : adj[v]) if (alive[u]) badj[local_of[v]].insert(local_of[u]);

    std::vector<HierEntry> plan; plan.reserve(N);
    std::vector<char> seen(nb, 0);
    int n_cluster = 0;
    for (int s=0; s<nb; s++) {
        if (seen[s]) continue;
        n_cluster++;
        int g0 = base_globals[s];                  // cluster's first point (representative)
        seen[s]=1;
        HierEntry root; root.global_id=g0; root.level=0;   // root: empty pred -> pred 0
        plan.push_back(std::move(root));
        std::vector<int> stack{ s };               // flood the rest of the cluster
        while(!stack.empty()){
            int c=stack.back(); stack.pop_back();
            for (int32_t u : badj[c]) if (!seen[u]) {
                seen[u]=1;
                HierEntry e; e.global_id=base_globals[u]; e.level=0;
                e.pred_ids.push_back(g0);          // predict from the first point only
                e.weights.push_back(1.0);          // pred = first-point value
                plan.push_back(std::move(e));
                stack.push_back(u);
            }
        }
        // // Flood-fill the whole connected cluster first (deterministic).
        // std::vector<int> members; members.push_back(s); seen[s]=1;
        // std::vector<int> stack{ s };
        // while(!stack.empty()){
        //     int c=stack.back(); stack.pop_back();
        //     for (int32_t u : badj[c]) if (!seen[u]) { seen[u]=1; members.push_back(u); stack.push_back(u); }
        // }
        // // Representative = member closest to the cluster's spatial centroid, so the
        // // single first-point predictor sits centrally and minimises residuals,
        // // while the prediction depth stays 1 (everyone predicts from it directly).
        // double cx=0,cy=0,cz=0;
        // for (int m : members) { int g=base_globals[m]; cx+=V[g*3]; cy+=V[g*3+1]; cz+=V[g*3+2]; }
        // double inv=1.0/(double)members.size(); cx*=inv; cy*=inv; cz*=inv;
        // int rep=members[0]; double best=std::numeric_limits<double>::max();
        // for (int m : members) {
        //     int g=base_globals[m];
        //     double dx=V[g*3]-cx, dy=V[g*3+1]-cy, dz=V[g*3+2]-cz;
        //     double d=dx*dx+dy*dy+dz*dz;
        //     if (d<best) { best=d; rep=m; }
        // }
        // int g0 = base_globals[rep];
        // HierEntry root; root.global_id=g0; root.level=0;   // root: empty pred -> pred 0
        // plan.push_back(std::move(root));
        // for (int m : members) {
        //     if (m==rep) continue;
        //     HierEntry e; e.global_id=base_globals[m]; e.level=0;
        //     e.pred_ids.push_back(g0);              // predict from the centroid point only
        //     e.weights.push_back(1.0);              // pred = representative value
        //     plan.push_back(std::move(e));
        // }
    }
    printf("base: %d clusters over %d points\n", n_cluster, nb);

    for (int L=(int)level_recs.size()-1; L>=0; L--)
        for (auto& e : level_recs[L]) plan.push_back(std::move(e));
    return plan;
}

} // namespace detail


// 3-D MIS hierarchy on a LIGHTWEIGHT ADJACENCY GRAPH (no tet mesh maintained).
// Per level: greedy independent set (low-degree first) on the current adjacency;
// each removed vertex is predicted by the best enclosing tet of 4 of its
// surviving neighbors; then the vertex is dropped and its neighbors are cliqued
// (vertex-elimination fill) so they stay mutually discoverable next level.
inline std::vector<HierEntry> build_tet_mis_hierarchy(
    const std::vector<double>& V, const std::vector<int>& tets_in,
    int N, int num_levels, int mdim, int from_edge)
{
    // auto adj = detail::tet_vertex_adjacency(tets_in, N);
    auto adj = UMC::generate_adjacent_list(N, mdim, tets_in, from_edge);
    std::vector<char> alive(N,1);
    std::vector<std::vector<HierEntry>> level_recs;
    int alive_count = N;

    for (int level=1; level<=num_levels; level++) {
        if (alive_count < 8) break;
        // greedy MIS, low-degree first
        std::vector<int> order; order.reserve(alive_count);
        for (int v=0;v<N;v++) if (alive[v]) order.push_back(v);
        std::sort(order.begin(), order.end(), [&](int a,int b){ return adj[a].size()<adj[b].size(); });
        std::vector<char> blocked(N,0), removed(N,0);
        for (int v : order) {
            if (!blocked[v]) {
                if((int)adj[v].size() < mdim+1) continue;  // need >= mdim+1 anchors
                removed[v]=1; blocked[v]=1;
                for (int u : adj[v]) blocked[u]=1;
            }
        }

        std::vector<HierEntry> recs;
        int n_enc=0, n_out=0, n_idw=0;
        for (int v=0; v<N; v++) {
            if (!removed[v]) continue;
            std::vector<int> cand(adj[v].begin(), adj[v].end());   // all survive (independent set)
            HierEntry e; e.global_id=v; e.level=level;
            int st=detail::best_simplex_pred(V, mdim, v, cand, e);
            e.pred_kind=st;
            if(st==0)n_enc++; else if(st==1)n_out++; else n_idw++;
            recs.push_back(std::move(e));
        }
        if (recs.empty()) break;
        printf("    [pred] level %d: enclosed %d, OUTSIDE %d (%.1f%%), idw %d\n",
               level, n_enc, n_out, 100.0*n_out/(n_enc+n_out+n_idw), n_idw);
        // update adjacency: clique each removed vertex's neighbors, then drop it
        int nrem=0;
        for (int v=0; v<N; v++) {
            if (!removed[v]) continue;
            std::vector<int> nb(adj[v].begin(), adj[v].end());
            if(mdim == 2) for (size_t i=0;i<nb.size();i++) for (size_t j=i+1;j<nb.size();j++) { adj[nb[i]].insert(nb[j]); adj[nb[j]].insert(nb[i]); }
            // if(level == num_levels) for (size_t i=0;i<nb.size();i++) for (size_t j=i+1;j<nb.size();j++) { adj[nb[i]].insert(nb[j]); adj[nb[j]].insert(nb[i]); }
            for (int u : nb) adj[u].erase(v);
            adj[v].clear(); alive[v]=0; alive_count--; nrem++;
        }
        printf("  TET-MIS level %d: removed %d, alive %d (keep %.1f%%)\n",
               level, nrem, alive_count, 100.0*alive_count/(alive_count+nrem));
        level_recs.push_back(std::move(recs));
    }
    return detail::assemble_tet_plan(V, N, adj, alive, level_recs);
}


// 3-D edge-collapse hierarchy on a LIGHTWEIGHT ADJACENCY GRAPH (no tet mesh, no
// link condition, no signed-volume checks — none needed since we never maintain
// a conforming tetrahedralization).  Per level: collapse shortest edges (b->a,
// transferring b's adjacency to a) until alive_count <= keep_ratio*before; each
// collapsed vertex is predicted by the best enclosing tet of 4 of its merge
// target's surviving neighbors.  keep_ratio gives exact per-level ratio control.
inline std::vector<HierEntry> build_tet_collapse_hierarchy(
    const std::vector<double>& V, const std::vector<int>& tets_in,
    int N, int num_levels, double keep_ratio, int mdim = 3)
{
    auto adj = detail::tet_vertex_adjacency(tets_in, N);
    auto dist=[&](int a,int b){ double dx=V[a*3]-V[b*3],dy=V[a*3+1]-V[b*3+1],dz=V[a*3+2]-V[b*3+2]; return std::sqrt(dx*dx+dy*dy+dz*dz); };
    std::vector<char> alive(N,1);
    std::vector<int>  merge_to(N,-1);
    auto rep=[&](int x){ while(!alive[x]) x=merge_to[x]; return x; };
    std::vector<std::vector<HierEntry>> level_recs;
    int alive_count = N;

    for (int level=1; level<=num_levels; level++) {
        int before=alive_count;
        int target=std::max(8,(int)std::ceil(before*keep_ratio));
        if (before<=target) break;

        std::priority_queue<std::tuple<double,int,int>, std::vector<std::tuple<double,int,int>>,
                            std::greater<std::tuple<double,int,int>>> pq;
        for (int v=0; v<N; v++) if (alive[v]) for (int u : adj[v]) if (v<u) pq.push({dist(v,u),v,u});

        std::vector<int> removed_this;
        while (alive_count>target && !pq.empty()) {
            auto [len,a,b]=pq.top(); pq.pop();
            if (!alive[a]||!alive[b]||!adj[a].count(b)) continue;
            int keep=std::min(a,b), rem=std::max(a,b);
            // collapse rem -> keep: transfer adjacency, drop rem (no mesh upkeep)
            for (int u : adj[rem]) { if (u!=keep) { adj[keep].insert(u); adj[u].insert(keep); } adj[u].erase(rem); }
            adj[rem].clear(); alive[rem]=0; merge_to[rem]=keep;
            removed_this.push_back(rem); alive_count--;
            for (int u : adj[keep]) pq.push({dist(keep,u), std::min(keep,u), std::max(keep,u)});
        }
        if (removed_this.empty()) break;

        std::vector<HierEntry> recs; recs.reserve(removed_this.size());
        int n_enc=0, n_out=0, n_idw=0;
        for (int rem : removed_this) {
            int s=rep(rem);
            std::vector<int> cand; cand.reserve(adj[s].size());
            for (int u : adj[s]) if (alive[u] && u!=rem) cand.push_back(u);
            HierEntry e; e.global_id=rem; e.level=level;
            int st=detail::best_simplex_pred(V, mdim, rem, cand, e);
            e.pred_kind=st;
            if(st==0)n_enc++; else if(st==1)n_out++; else n_idw++;
            recs.push_back(std::move(e));
        }
        printf("  TET-EC level %d: %d -> %d (%d collapsed, keep %.1f%%)\n",
               level, before, alive_count, before-alive_count, 100.0*alive_count/before);
        printf("    [pred] level %d: enclosed %d, OUTSIDE %d (%.1f%%), idw %d\n",
               level, n_enc, n_out, 100.0*n_out/(n_enc+n_out+n_idw), n_idw);
        level_recs.push_back(std::move(recs));
    }
    return detail::assemble_tet_plan(V, N, adj, alive, level_recs);
}

} // namespace UMC
#endif
