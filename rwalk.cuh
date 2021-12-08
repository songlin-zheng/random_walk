// #ifndef RWALKCU_H_
// #define RWALKCU_H_

#include <stdio.h>
#include <assert.h>


// void __global__ singleRandomWalk(int num_of_node, int num_of_walk, int max_walk_length, int32_t* node_idx, float* timestamp, int32_t* start_idx_dev, int32_t* rand_walk);

void cuda_rwalk(int max_walk_length, int num_walks_per_node, int32_t num_nodes, int32_t num_edges, unsigned long long random_number);

// #endif /* RWALKCU_H_ */