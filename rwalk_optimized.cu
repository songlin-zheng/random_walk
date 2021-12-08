#include "rwalk.cuh"
#include "helper.cuh"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <limits>

int32_t* random_walk_dev;
int32_t* node_idx_dev;
float* timestamp_dev;
int32_t* start_idx_dev;

float *cdf_buffer_host;
float *cdf_buffer_dev;

int32_t *node_idx_host_sorted;
float *timestamp_host_sorted;

float *timestamp_dev_sorted;
int32_t *node_idx_dev_sorted;

int32_t *mapping_host;
int32_t *mapping_dev;

float extend_ratio = 0.1;

int threadBlockSize;
cudaDeviceProp prop;
int count_dev;
bool preprocessing = true;


// assert(err == cudaSuccess);

#define cudaCheck(err) { \
	if (err != cudaSuccess) { \
		printf("CUDA error: %s: %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
	} \
}

// rand_walk -> [num_of_node, num_of_walk, max_walk_length]
void __global__ singleRandomWalk(int num_of_node, int num_of_walk, int max_walk_length, int32_t* node_idx, float* timestamp, int32_t* start_idx, int32_t* rand_walk, unsigned long long rnumber){
    // assuming grid = 1
    int32_t i =  (blockDim.x * blockIdx.x) + threadIdx.x;
    rnumber = i * (unsigned long long) rnumber + 11;
    if(i >= num_of_node * num_of_walk){
        return;
    }

    int32_t src_node = i / (int32_t) num_of_walk;
    float curr_timestamp = .0f;
    rand_walk[i * max_walk_length + 0] = src_node;

    // printf("start : %lld ; end : %lld; src_node: %lld; num_of_walk : %d; max_walk_length: %d; i : %lld\n", (long long int)start, (long long int)end, (long long int)src_node, num_of_walk, max_walk_length, (long long int)i);
    int32_t start;
    int32_t end;

    int walk_cnt;
    for(walk_cnt = 1; walk_cnt < max_walk_length; walk_cnt ++){
        // ! can be improved
        start = start_idx[src_node];
        end = start_idx[src_node + 1];
        // printf("start: %lld end: %lld\n", (long long int) start, (long long int)end);

        // control divergence
        // range should be [start, end)
        if(start < end){
            float* valid_timestamp = (float*) malloc((end - start) * sizeof(float));
            int32_t* valid_node = (int32_t*) malloc((end - start) * sizeof(int32_t));
            int idx = 0;
            // float cdf[end - start];
            float max_timestamp = timestamp[start];
            float min_timestamp = timestamp[start];
            // ! parallizable
            for(int32_t j = start; j < end; j ++){
                // printf("idx: %lld, timestamp: %f node_idx: %lld\n", (long long int) j, timestamp[j], (long long int) node_idx);
                if(timestamp[j] > curr_timestamp){
                    valid_node[idx] = node_idx[j];
                    valid_timestamp[idx++] = timestamp[j];
                }
                max_timestamp = max(max_timestamp, timestamp[j]);
                min_timestamp = min(min_timestamp, timestamp[j]);
            }
            // printf("idx: %d\n", idx);
            if(!idx){
                free(valid_node);
                free(valid_timestamp);
                break;
            }

            // every timestamp is the same
            // printf("max: %f ; min : %f\n", max_timestamp, min_timestamp);
            if(max_timestamp - min_timestamp >= - 0.0000001 && max_timestamp - min_timestamp <= 0.0000001){
                // printf("valid node: %lld\n", (long long int)valid_node[0]);
                // printf("Time interval too small\n");
                rand_walk[i * max_walk_length + walk_cnt] = valid_node[0];
                src_node = valid_node[0];
                curr_timestamp = valid_timestamp[0];
                free(valid_node);
                free(valid_timestamp);
                continue;
            }

            float* cdf = (float*) malloc(idx * sizeof(float));

            // ! need to determine how to get prob
            float prob = rnumber * 1.0 / ULLONG_MAX;

            // refresh rnumber
            rnumber = rnumber * (unsigned long long)25214903917 + 11;
            bool fall_through = true;

            // ! reduction tree here (kernel in kernel)
            float denom = .0f;
            for(int j = 0; j < idx; j ++){
                cdf[j] =  expf((valid_timestamp[j] - curr_timestamp) / (max_timestamp - min_timestamp));
                denom += cdf[j];
            }
            float curr_cdf = .0f,  next_cdf = .0f;
            for(int j = 0; j < idx; j ++){
                next_cdf += cdf[j] / denom;
                if(prob >= curr_cdf && prob <= next_cdf){
                    // printf("valid node: %lld\n", (long long int)valid_node[j]);
                    rand_walk[i * max_walk_length + walk_cnt] = valid_node[j];
                    src_node = valid_node[j];
                    curr_timestamp = valid_timestamp[j];
                    fall_through = false;
                    break;
                }
                curr_cdf = next_cdf;
            }

            // fall through should never happen
            if(fall_through){
                // printf("valid node: %lld\n", (long long int)valid_node[0]);
                rand_walk[i * max_walk_length + walk_cnt] = valid_node[0];
                src_node = valid_node[0];
                curr_timestamp = valid_timestamp[0];
            }

            free(valid_node);
            free(valid_timestamp);
            free(cdf);
        }
        else{
            break;
        }
    }

    if(walk_cnt < max_walk_length){
        // signal the rest is invalid and there is no descending node
        rand_walk[i * max_walk_length + walk_cnt] = -1;
    }
}


void cuda_rwalk(int max_walk_length, int num_walks_per_node, int32_t num_nodes, int32_t num_edges, unsigned long long random_number){

#if defined(DEBUG)
    size_t free_memory;
    size_t total_memory;

    cudaCheck(cudaMemGetInfo(&free_memory, &total_memory));
    // printf("free memory : %zu ; total memory : %zu\n", free_memory, total_memory);
#endif

    // malloc GPU memory
    cudaCheck(cudaMalloc((void **)&start_idx_dev, sizeof(int32_t) * (num_nodes + 1)));
    cudaCheck(cudaMalloc((void **)&node_idx_dev, sizeof(int32_t) * num_edges));
    cudaCheck(cudaMalloc((void **)&timestamp_dev, sizeof(float) * num_edges));
    cudaCheck(cudaMalloc((void **)&random_walk_dev, sizeof(int32_t) * num_nodes * max_walk_length * num_walks_per_node));

    // memcpy
    cudaCheck(cudaMemcpy(start_idx_dev, start_idx_host, sizeof(int32_t) * (num_nodes + 1), cudaMemcpyHostToDevice));

    cudaGetDeviceProperties(&prop, 0);
    threadBlockSize = prop.maxThreadsPerBlock;

    if(preprocessing){
        cdf_buffer_host = new float[num_edges];
        node_idx_host_sorted = new int32_t[num_edges];
        timestamp_host_sorted = new float[num_edges];
        mapping_host = new int32_t[num_edges];

        cudaCheck(cudaMalloc((void **)&cdf_buffer_dev, sizeof(float) * num_edges));
        cudaCheck(cudaMalloc((void **)&mapping_dev, sizeof(int32_t) * num_edges));
        cudaCheck(cudaMalloc((void **)&node_idx_dev_sorted, sizeof(int32_t) * num_edges));
        cudaCheck(cudaMalloc((void **)&timestamp_dev_sorted, sizeof(float) * num_edges));
        cuda_helper(max_walk_length, num_walks_per_node, num_nodes, num_edges);
    }
    else{
        cudaCheck(cudaMemcpy(node_idx_dev, node_idx_host, sizeof(int32_t) * num_edges, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(timestamp_dev, timestamp_host, sizeof(float) * num_edges, cudaMemcpyHostToDevice));
    }

#if defined(DEBUG)
    cudaGetDeviceCount(&count_dev);
    for(int i = 0; i < count_dev; i ++){
        printf("total_global_Mem: %zu MB\n shared_mem_per_block: %zu\n max_threads_per_block: %d\n max_thread_dim: [%d, %d, %d]\n max_grid_size: [%d, %d, %d]",
        prop.totalGlobalMem / 1024 / 1024, prop.sharedMemPerBlock, prop.maxThreadsPerBlock, prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2], prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    }
#endif


    // start training
    int grid_size = (num_nodes * num_walks_per_node - 1) / 32 + 1;
    dim3 gridDim(grid_size);
    dim3 blockDim(32);

    singleRandomWalk<<<gridDim, blockDim>>>(num_nodes, num_walks_per_node, max_walk_length, node_idx_dev, timestamp_dev, start_idx_dev, random_walk_dev, random_number);

#if defined(DEBUG)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
#endif

    // cudaCheck(cudaMemGetInfo(&free_memory, &total_memory));
    // printf("free memory : %zu ; total memory : %zu\n", free_memory, total_memory);

    // get result
    cudaDeviceSynchronize();
    cudaCheck(cudaMemcpy(random_walk_host, random_walk_dev, sizeof(int32_t) * num_nodes * max_walk_length * num_walks_per_node, cudaMemcpyDeviceToHost));

    if(preprocessing){
        cudaCheck(cudaFree(cdf_buffer_dev));
        cudaCheck(cudaFree(mapping_dev));
        cudaCheck(cudaFree(node_idx_dev_sorted));
        cudaCheck(cudaFree(timestamp_dev_sorted));
        delete[] mapping_host;
        delete[] timestamp_host_sorted;
        delete[] node_idx_host_sorted;
        delete[] cdf_buffer_host;
    }
    // clean arrays
    cudaCheck(cudaFree(start_idx_dev));
    cudaCheck(cudaFree(node_idx_dev));
    cudaCheck(cudaFree(timestamp_dev));
    cudaCheck(cudaFree(random_walk_dev));
}

