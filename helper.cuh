

#include <stdio.h>
#include <assert.h>
// #include <cuda.h>
// #include <cuda_runtime.h>

// decleration
extern int32_t* dev_node_idx;
extern float* dev_timestamp;
extern int32_t* dev_start_idx;

extern int32_t *start_idx_host;
extern int32_t *node_idx_host;
extern float *timestamp_host;
extern int32_t *random_walk_host;


extern int dev_count;

void cuda_helper_test(int max_walk_length, int num_walks_per_node, int32_t num_nodes, int32_t num_edges, unsigned long long random_number);