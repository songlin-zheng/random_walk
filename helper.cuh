

#include <stdio.h>
#include <assert.h>


extern int32_t *start_idx_host; // rwalk.h
extern int32_t *node_idx_host; // rwalk.h
extern float *timestamp_host; // rwalk.h
extern int32_t *random_walk_host; //rwalk.h

void cuda_helper(int max_walk_length, int num_walks_per_node, int32_t num_nodes, int32_t num_edges);