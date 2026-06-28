#ifndef _UMC_REORDER_HPP
#define _UMC_REORDER_HPP

#include <vector>
#include <queue>
#include <set>
#include <stack>
#include <cassert>
#include <queue>
#include "partition.hpp"

namespace UMC{

// Reordering based on Fast and Effective Lossy Compression Algorithms for Scientific Datasets
template<typename T>
std::vector<T> reorder(const std::vector<T>& data, const std::vector<int32_t>& index_map, int n, int d){
	assert(data.size() == n*d);
	std::vector<T> reordered_data(data.size());
	for(int i=0; i<index_map.size(); i++){
		auto new_i = index_map[i];
		for(int j=0; j<d; j++){
			reordered_data[new_i*d + j] = data[i*d + j];
		}
	}
	return reordered_data;
}

template<typename T>
std::vector<T> reorder_conn(const std::vector<T>& conn, const std::vector<int32_t>& index_map, int d){
	std::vector<T> reordered_conn(conn.size());
	for(int i=0; i<conn.size(); i++){
		reordered_conn[i] = index_map[conn[i]];
	}
	return reordered_conn;
}

// breath-first search
std::vector<int32_t> BFS(const std::vector<std::set<int32_t>>& adj_list){
    int n = adj_list.size();
    std::vector<int32_t> index_map(n, 0);
    std::vector<bool> visited(n, false);
    // TODO: consider multiple connected components
    std::queue<int32_t> node_queue;
    visited[0] = true;
    node_queue.push(0);
    int count = 0;
    while(!node_queue.empty()){
    	auto index = node_queue.front();
    	node_queue.pop();
    	index_map[index] = count ++;
    	for(const auto& iter:adj_list[index]){
    		if(!visited[iter]){
    			node_queue.push(iter);
    			visited[iter] = true;
    		}
    	}
    }
    printf("BFS: #mapped_index = %d, #total_index = %d\n", count, n);
    return index_map;
}

// breath-first search with priority
// Visits the frontier node with the most already-processed (placed) neighbors
// first.  A lazy max-priority-queue keyed by (processed_count, -node) makes this
// O((V+E) log V) instead of O(V^2); the -node tie-break reproduces the original
// "smallest index among the max" choice.  Every processed_count increment pushes
// a fresh entry, so the true current maximum is always represented; stale entries
// (lower than the node's current count, or already placed) are skipped on pop.
// The outer loop reseeds at each unvisited node, covering disconnected components
// (the original left those at index 0, colliding).
std::vector<int32_t> BPFS(const std::vector<std::set<int32_t>>& adj_list){
    int n = adj_list.size();
    std::vector<int32_t> index_map(n, 0);
    std::vector<int32_t> processed_nodes(n, 0);
    std::vector<bool> placed(n, false);
    // max-heap of (processed_count, -node)
    std::priority_queue<std::pair<int32_t,int32_t>> pq;
    int count = 0;
    for(int s = 0; s < n; s++){
        if(placed[s]) continue;
        pq.push({0, -s});
        while(!pq.empty()){
            auto top = pq.top(); pq.pop();
            int node = -top.second;
            if(placed[node]) continue;                 // already emitted
            if(top.first != processed_nodes[node]) continue;  // stale priority
            index_map[node] = count++;
            placed[node] = true;
            for(const auto& iter : adj_list[node]){
                if(placed[iter]) continue;
                processed_nodes[iter]++;
                pq.push({processed_nodes[iter], -(int32_t)iter});  // fresh entry
            }
        }
    }
    printf("BPFS: #mapped_index = %d, #total_index = %d\n", count, n);
    return index_map;
}

// depth-first search
std::vector<int32_t> DFS(const std::vector<std::set<int32_t>>& adj_list){
    int n = adj_list.size();
    std::vector<int32_t> index_map(n, -1);
    std::vector<bool> visited(n, false);
    std::stack<int32_t> node_stack;
    int count = 0;
    node_stack.push(0);
    while(!node_stack.empty()){
        auto index = node_stack.top();
        node_stack.pop();
        if(!visited[index]){
            visited[index] = true;
            index_map[index] = count++;
        }
        // if(count % 10000 == 0){
        //     std::cout << count << " / " << n << "\n";
        // }
        for(const auto& i : adj_list[index]){
            if(!visited[i]){
                node_stack.push(i);
            }
        }
    }
    printf("DFS: #mapped_index = %d, #total_index = %d\n", count, n);
    return index_map;
}

// depth-first search with priority
std::vector<int32_t> DPFS(const std::vector<std::set<int32_t>>& adj_list){
    int n = adj_list.size();
    std::vector<int32_t> index_map(n, 0);
    std::vector<int32_t> processed_nodes(n, 0);
    std::vector<bool> visited(n, false);
    std::stack<int32_t> node_stack;
    int count = 0;
    // Restart from the next unvisited node when a component is exhausted, so a
    // DISCONNECTED graph (e.g. a fragmented MIS base layer) still yields a full
    // permutation covering all n nodes — otherwise unvisited nodes keep index 0,
    // breaking the causal (order[u] < order[c]) base-layer prediction.
	int cluster = 0;
    for(int start=0; start<n; start++){
		if(visited[start]) continue;
		cluster++;
		visited[start] = true;
		index_map[start] = count ++;
		node_stack.push(start);
		while(!node_stack.empty()){
			auto index = node_stack.top();
			if(!visited[index]){
				index_map[index] = count ++;
				visited[index] = true;
			}
			// if(count % 10000 == 0){
			// 	std::cout << count << " / " << n << "\n";
			// }
			if(count > n){
				std::cerr << "exceed limits\n";
				exit(-1);
			}
			// update processed neighbors
			for(const auto& iter:adj_list[index]){
				processed_nodes[iter] ++;
			}
			// pick the node with the most processed neighbors
			int max_num_neighbor = 0;
			int max_index = 0;
			bool traverse_flag = false;
			for(const auto& i : adj_list[index]){
				if(visited[i]) continue;
				if(max_num_neighbor < processed_nodes[i]){
					max_num_neighbor = processed_nodes[i];
					max_index = i;
				} 
				traverse_flag = true;
			}
			if(traverse_flag){
				node_stack.push(max_index);
			}
			else{
				// nothing to traverse for current node, remove from stack
				node_stack.pop();
			}
		}
    }
    printf("DPFS: #mapped_index = %d, #total_index = %d, #cluster = %d\n", count, n, cluster);
    return index_map;
}

std::vector<int32_t> generate_reorder_index_map(int n, const std::vector<std::set<int32_t>>& adj_list, int opt){
	assert(n == adj_list.size());
	std::vector<int32_t> index_map(n, -1);
	switch(opt){
        case 1:            
            index_map = DPFS(adj_list);
            break;
        case 2:            
            index_map = DFS(adj_list);
            break;
        case 3:
            index_map = BFS(adj_list);
            break;
        case 4:
            index_map = BPFS(adj_list);
            break;
        default:
            printf("Original order\n");
            break;
	};
	return index_map;
}

// depth-first search with priority
std::vector<uint32_t> blockDPFS(const std::vector<std::set<int32_t>>& adj_list, const std::vector<idx_t>& part, const std::vector<uint32_t>& part_map_inv, const std::vector<uint32_t>& num_part_nodes){
    int n = adj_list.size();
	int num_blocks = num_part_nodes.size();
    std::vector<uint32_t> index_map(n, 0);
    std::vector<uint32_t> processed_nodes(n, 0);
    std::vector<bool> visited(n, false);
    // TODO: consider multiple connected components
    int count = 0;
	uint32_t offset = 0;
	uint32_t expected_count = 0;
	for(int b=0; b<num_blocks; b++){
		expected_count += num_part_nodes[b];
		if(num_part_nodes[b] == 0) continue;
		while(expected_count != count){
			int start_id = -1;
			for(int i=0; i<num_part_nodes[b]; i++){
				int original_id = part_map_inv[offset + i];
				if(!visited[original_id]){
					start_id = original_id;
					break;
				}
			}
			int block_id = b;
			std::stack<int32_t> node_stack;
			node_stack.push(start_id);
			while(!node_stack.empty()){
				auto index = node_stack.top();
				if(!visited[index]){
					index_map[index] = count ++;
					visited[index] = true;
				}
				if(count > n){
					std::cerr << "exceed limits\n";
					exit(-1);
				}
				// update processed neighbors
				for(const auto& iter:adj_list[index]){
					if((part[iter] > block_id) ||((part[iter] == block_id) &&(!visited[iter]))) processed_nodes[iter] ++;
				}
				// pick the node with the most processed neighbors
				int max_num_neighbor = 0;
				int max_index = 0;
				bool traverse_flag = false;
				for(const auto& i : adj_list[index]){
					if(visited[i]) continue;
					if(part[i] != block_id) continue;
					if(max_num_neighbor < processed_nodes[i]){
						max_num_neighbor = processed_nodes[i];
						max_index = i;
					} 
					traverse_flag = true;
				}
				if(traverse_flag){
					node_stack.push(max_index);
				}
				else{
					// nothing to traverse for current node, remove from stack
					node_stack.pop();
				}
			}
		}
		offset += num_part_nodes[b];
	}
    printf("DPFS: #mapped_index = %d, #total_index = %d\n", count, n);
    return index_map;
}

}
#endif