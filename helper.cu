#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <limits>
#include "helper.cuh"

int32_t* host_node_idx_out;
float* host_timestamp_idx_out;

int32_t* dev_node_idx;
float* dev_timestamp;
int32_t* dev_start_idx;

float* dev_timestamp_out;
int32_t* dev_node_idx_out;

int threadBlockSize;
cudaDeviceProp prop;


// assert(err == cudaSuccess);

#define cudaCheck(err) { \
	if (err != cudaSuccess) { \
		printf("CUDA error: %s: %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
	} \
}

// template <class int_t>
// void getMaximalTimestampPerNode();
template <class int_t, class float_t>
void __global__ sortTimestamp(int_t start, int_t end, float_t* src_ts, int_t* src_node, float_t* dest_ts, int_t* dest_node){
    extern __shared__ float smem[];
    // address missalignment
    float_t* local_ts = smem;
    int_t* idx_1 = (int_t*)&smem[end - start];
    int_t* idx_2 = (int_t*)&idx_1[end - start];
    int tid = threadIdx.x;

    // recursively load elements into local_arr
    for(int_t i = tid; i < end - start; i += blockDim.x){
        local_ts[i] = src_ts[start + i];
        idx_1[i] = i;
    }
    __syncthreads();

    int_t width = 1;
    int num_of_sort = 0;
    while(width < end - start){
        for(int_t i = tid; i < end - start; i += blockDim.x){
            int_t start_i = i * (width << 1);
            int_t mid_i = min(start_i + width, end - start);
            int_t end_i = min(end - start, start_i + (width << 1));
            if(start_i < end - start){
                // printf("start, mid, end: %lld, %lld, %lld\n", start_i, mid_i, end_i);

                if(num_of_sort % 2 == 0){
                    bottomUpSort(local_ts, idx_1, idx_2, start_i, mid_i, end_i);
                }
                else{
                    bottomUpSort(local_ts, idx_2, idx_1, start_i, mid_i, end_i);
                }
            }
        }
        // width x 2
        width <<= 1;
        num_of_sort ++;
        __syncthreads();
    }

    if(tid == 0){
        for(int i = 0; i < end - start; i ++){
            int_t idx = num_of_sort % 2 == 0 ? idx_1[i] : idx_2[i];
            printf("[i, idx, node_idx, ts]: [%d, %d, %d, %f] ", i, idx, src_node[start + idx], local_ts[idx]);
        }
    }
    printf("\n");

    for(int i = tid; i < end - start; i += blockDim.x){
        int_t idx = num_of_sort % 2 == 0 ? idx_1[i] : idx_2[i];
        dest_ts[start + i] = local_ts[idx];
        dest_node[start + i] = src_node[start + idx];
    }
}

// [start, end)
template <class int_t, class float_t>
void __device__ inline bottomUpSort(float_t* ts, int_t* src_idx, int_t* dest_idx, int_t start, int_t mid, int_t end){
    // ovid overflow
    int_t i = start, j = mid;
    for(int_t k = start; k < end; k ++){
        // printf("src_idx[i], src_idx[j], ts[src_idx[i]], ts[src_idx[j]]: %lld, %lld, %f, %f\n", src_idx[i], src_idx[j], ts[src_idx[i]], ts[src_idx[j]]);
        if(i < mid && (j == end || ts[src_idx[i]] < ts[src_idx[j]])){
            dest_idx[k] = src_idx[i];
            i ++;
        }
        else{
            dest_idx[k] = src_idx[j];
            j ++;
        }
    }
}

template <class int_t, class float_t>
void __global__ getOutEdgeTimestampCorrespondence(int_t* start_idx, float_t* timestamp, int_t* node_idx, int_t* outIdx, int_t curr_node){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //  wrap up for edge > blockDim
    int_t start = start_idx[curr_node];
    int_t end = start_idx[curr_node + 1];
    for(int i = idx; i < end - start; i += blockDim.x){
        int_t next_node = node_idx[start + i];
        float next_ts = timestamp[start + i];
        int_t next_start = start_idx[next_node];
        int_t next_end = start_idx[next_node + 1];
        // binary search to find the timestamp position in all out edges
        int_t l = 0, r = next_end - 1 - next_start;
        while(l < r){
            int_t mid = l + (r - l) / 2;
            if(timestamp[next_start + mid] >= next_ts){
                r = mid - 1;
            }
            else{
                l = mid + 1;
            }
        }
        // 1. l = next_end next_ts is larger than all its out edge ts
        // 2. l = [0, next_end)
        // 3. l = -1 next_ts is smaller than all its out edge ts
        if(l == 0 && next_ts < timestamp[next_start]){
            outIdx[start + i] = -1;
        }
        else{
            outIdx[start + i] = l;
        }
    }
}

template <class int_t, class float_t>
void __global__ prefixSumLinear(int_t* start_idx, int_t* node_idx, float_t* timestamp, float_t* buffer, int_t curr_node){
    assert(blockDim.x == 1 && "linear prefix sum must have only 1 thread in each block");
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int_t start = start_idx[curr_node];
    int_t end = start_idx[curr_node + 1];

    // wrap up
    for(int i = idx; i < end - start; i ++){
        buffer[start + i] = timestamp[start + i] + (i == 0 ? 0 : buffer[start + i - 1]);
    }
}

template <class int_t, class float_t>
void __global__ prefixSumConcurrent(int_t* start_idx, int_t* node_idx, float_t* timestamp, float_t* buffer, int_t curr_node){
    // printf("I'm in\n");
    extern __shared__ float_t smem[];

    int_t start = start_idx[curr_node];
    int_t end = start_idx[curr_node + 1];
    // printf("s, e: %d %d \n", start, end);
    int tid = threadIdx.x;

    //  wrap up
    for(int i = tid; i < end - start; i += blockDim.x){
        smem[i] = timestamp[end - 1] - timestamp[start + i];
        // printf("[i, val]: [%d, %f] ", i, smem[i]);
    }
    // printf("\n");
    __syncthreads();

    // reduction
    int stride = 1;
    while(stride  < end - start){
        for(int i = tid; i < end - start; i += blockDim.x){
            int idx = stride * (i + 1) * 2 - 1;
            if(idx < end - start){
                smem[idx] += smem[idx - stride];
            }
        }
        stride <<= 1;
        __syncthreads();
    }

    // post scan
    stride >>= 1;
    while(stride > 0){
        for(int i = tid; i < end - start; i += blockDim.x){
            int idx = stride * (i + 1) * 2 - 1;
            if(idx < end - start){
                smem[idx + stride] += smem[idx];
            }
        }
        stride >>= 1;
        __syncthreads();
    }

    // put into buffer
    for(int i = tid; i < end - start; i += blockDim.x){
        buffer[start + i] = smem[i];
    }
}




void cuda_helper_test(int max_walk_length, int num_walks_per_node, int32_t num_nodes, int32_t num_edges, unsigned long long random_number){
    host_node_idx_out = new int32_t[num_edges];
    host_timestamp_idx_out = new float[num_edges];
    cudaCheck(cudaMalloc((void **)&dev_start_idx, sizeof(int32_t) * (num_nodes + 1)));
    cudaCheck(cudaMalloc((void **)&dev_node_idx, sizeof(int32_t) * num_edges));
    cudaCheck(cudaMalloc((void **)&dev_timestamp, sizeof(float) * num_edges));
    cudaCheck(cudaMalloc((void **)&dev_node_idx_out, sizeof(int32_t) * num_edges));
    cudaCheck(cudaMalloc((void **)&dev_timestamp_out, sizeof(float) * num_edges));

    // memcpy
    cudaCheck(cudaMemcpy(dev_start_idx, start_idx_host, sizeof(int32_t) * (num_nodes + 1), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_node_idx, node_idx_host, sizeof(int32_t) * num_edges, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_timestamp, timestamp_host, sizeof(float) * num_edges, cudaMemcpyHostToDevice));

    // cudaGetDeviceCount(&dev_count);
    // for(int i = 0; i < dev_count; i ++){
    //     printf("total_global_Mem: %zu\n shared_mem_per_block: %zu\n max_threads_per_block: %d\n max_thread_dim: %d\n max_grid_size: %d",
    //     prop.totalGlobalMem, prop.sharedMemPerBlock, prop.maxThreadsPerBlock, prop.maxThreadsDim, prop.maxGridSize);

    // }
    cudaGetDeviceProperties(&prop, 0);
    threadBlockSize = prop.maxThreadsPerBlock;
    // start training
    // int grid_size = (num_nodes * num_walks_per_node - 1 ) / 32 + 1;
    // ?? header file
    int count = 10;
    cudaStream_t* streams = new cudaStream_t[count];
    for(int i = 0; i < count; i ++){
        cudaStreamCreate(&streams[i]);
    }

    for(int i = 0; i < num_nodes; i ++){
        int32_t start = start_idx_host[i];
        int32_t end = start_idx_host[i + 1];
        if(end - start > 0){
            dim3 grid(1);
            dim3 block(min((int)threadBlockSize, (int)ceil((end - start) / 2.0)));
            int stream_id = i % count;
            // wait previous job to finish
            cudaStreamSynchronize(streams[stream_id]);

            // memcpy
            cudaCheck(cudaMemcpyAsync((void**) &dev_node_idx[start], (void**) &node_idx_host[start], sizeof(int32_t) * (end - start), cudaMemcpyHostToDevice, streams[stream_id]));
            cudaCheck(cudaMemcpyAsync((void**) &dev_timestamp[start], (void**) &timestamp_host[start], sizeof(float) * (end - start), cudaMemcpyHostToDevice, streams[stream_id]));
            sortTimestamp<int32_t, float><<<grid, block, sizeof(float) * (end - start) + 2 * sizeof(int32_t) * (end - start), streams[stream_id]>>>(start_idx_host[i], start_idx_host[i + 1], dev_timestamp, dev_node_idx, dev_timestamp_out, dev_node_idx_out);
            cudaCheck(cudaMemcpyAsync((void**) &host_node_idx_out[start], (void**) &dev_node_idx_out[start], sizeof(int32_t) * (end - start), cudaMemcpyDeviceToHost, streams[stream_id]));
            cudaCheck(cudaMemcpyAsync((void**) &host_timestamp_idx_out[start], (void**) &dev_timestamp_out[start], sizeof(float) * (end - start), cudaMemcpyDeviceToHost, streams[stream_id]));
        }
    }

    for(int i = 0; i < count; i ++){
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    delete [] streams;

    // get result
    for(int i = 0; i < num_nodes; i ++){
        printf("before:\n [node_idx, timeStamp]:\n");
        for(int j = start_idx_host[i]; j < start_idx_host[i + 1]; j ++){
            printf("[%d,%f] ", node_idx_host[j], timestamp_host[j]);
        }
        printf("\n");
        printf("after:\n");
        for(int j = start_idx_host[i]; j < start_idx_host[i + 1]; j ++){
            printf("[%d,%f] ", host_node_idx_out[j], host_timestamp_idx_out[j]);
        }
        printf("\n");
    }

    // test CDF buffer
    float* buffer_host = new float[num_edges];
    float* buffer_dev;
    cudaMalloc((void**) & buffer_dev, sizeof(float) * num_edges);
    for(int i = 0; i < num_nodes; i ++){
        int32_t start = start_idx_host[i];
        int32_t end = start_idx_host[i + 1];
        if(end - start > 0){
            dim3 grid(1);
            dim3 block(min(threadBlockSize, (end - start)));
            prefixSumConcurrent<int32_t, float><<<grid, block, sizeof(float) * (end - start)>>>(dev_start_idx, dev_node_idx_out, dev_timestamp_out, buffer_dev, i);
        }
    }
    cudaDeviceSynchronize();

    cudaMemcpy(buffer_host, buffer_dev, sizeof(float) * num_edges, cudaMemcpyDeviceToHost);

    for(int i = 0; i < num_nodes; i ++){
        for(int j = start_idx_host[i]; j < start_idx_host[i + 1]; j ++){
            printf("[%d,%f] ", host_node_idx_out[j], buffer_host[j]);
        }
        printf("\n");
    }

    int32_t* mapping_host = new int32_t[num_edges];
    int32_t* mapping_dev;
    cudaMalloc((void**) & mapping_dev, sizeof(int32_t) * num_edges);
    for(int32_t i = 0; i < num_nodes; i ++){
        int32_t start = start_idx_host[i];
        int32_t end = start_idx_host[i + 1];
        if(end - start > 0){
            dim3 grid(1);
            dim3 block(min(threadBlockSize, (end - start)));
            getOutEdgeTimestampCorrespondence<int32_t, float><<<grid, block>>>(dev_start_idx, dev_timestamp_out, dev_node_idx_out, mapping_dev, i);
        }
    }
    cudaDeviceSynchronize();

    cudaMemcpy(mapping_host, mapping_dev, sizeof(float) * num_edges, cudaMemcpyDeviceToHost);

    for(int i = 0; i < num_nodes; i ++){
        for(int j = start_idx_host[i]; j < start_idx_host[i + 1]; j ++){
            int32_t nbr = host_node_idx_out[j];
            printf("node: %d ts: %f pos: %d\n neighbor %d\n ts: ", i, host_timestamp_idx_out[j], mapping_host[j], host_node_idx_out[j]);
            for(int k = start_idx_host[nbr]; k < start_idx_host[nbr + 1];  k ++){
                printf("%f ", host_timestamp_idx_out[k]);
            }
            printf("\n");
        }
        printf("\n");
    }



    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    // cudaCheck(cudaMemGetInfo(&free_memory, &total_memory));
    // printf("free memory : %zu ; total memory : %zu\n", free_memory, total_memory);

    // get result
    // for(int i = 0; i < num_nodes; i ++){
    //     printf("before:\n [node_idx, timeStamp]:\n");
    //     for(int j = start_idx_host[i]; j < start_idx_host[i + 1]; j ++){
    //         printf("[%d,%f] ", node_idx_host[j], timestamp_host[j]);
    //     }
    //     printf("\n");
    //     printf("after:\n");
    //     for(int j = start_idx_host[i]; j < start_idx_host[i + 1]; j ++){
    //         printf("[%d,%f] ", host_node_idx_out[j], host_timestamp_idx_out[j]);
    //     }
    //     printf("\n");
    // }

    // clean arrays
    cudaCheck(cudaFree(dev_start_idx));
    cudaCheck(cudaFree(dev_node_idx));
    cudaCheck(cudaFree(dev_timestamp));
}