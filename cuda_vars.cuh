#ifndef CUDA_VARS_H_
#define CUDA_VARS_H_

#include <cuda.h>
#include <cuda_runtime.h>

// declaration
// dev
extern int32_t* random_walk_dev; // rwalk.cu
extern int32_t* node_idx_dev; // rwalk.cu
extern float* timestamp_dev; // rwalk.cu
extern int32_t* start_idx_dev; // rwalk.cu
extern int32_t* node_idx_host_sorted; // rwalk.cu
extern float* timestamp_host_sorted;  // rwalk.cu
extern float* cdf_buffer_host; // rwalk.cu
extern int32_t* mapping_host; // rwalk.cu

// host
extern float* timestamp_dev_sorted; // rwalk.cu
extern int32_t* node_idx_dev_sorted; // rwalk.cu
extern float* cdf_buffer_dev; // rwalk.cu
extern int32_t* mapping_dev; // rwalk.cu

// misc
extern int threadBlockSize; // rwalk.cu
extern int count_dev;
extern cudaDeviceProp prop; // rwalk.cu
extern float extend_ratio; // rwalk.cu

#endif