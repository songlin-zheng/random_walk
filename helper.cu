#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <limits>
#include "helper.cuh"
#include "cuda_vars.cuh"

#define cudaCheck(err)                                                                            \
    {                                                                                             \
        if (err != cudaSuccess)                                                                   \
        {                                                                                         \
            printf("CUDA error: %s: %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        }                                                                                         \
    }

// [start, end)
template <class int_t, class w_t>
void __device__ inline bottomUpSort(w_t *weight, int_t *src_idx, int_t *dest_idx, int_t start, int_t mid, int_t end)
{
    // ovid overflow
    int_t i = start, j = mid;
    for (int_t k = start; k < end; k++)
    {
        // printf("src_idx[i], src_idx[j], weight[src_idx[i]], weight[src_idx[j]]: %lld, %lld, %f, %f\n", src_idx[i], src_idx[j], weight[src_idx[i]], weight[src_idx[j]]);
        if (i < mid && (j == end || weight[src_idx[i]] < weight[src_idx[j]]))
        {
            dest_idx[k] = src_idx[i];
            i++;
        }
        else
        {
            dest_idx[k] = src_idx[j];
            j++;
        }
    }
}

template <class int_t, class w_t>
void __device__ mergeSort(int_t start, int_t end, int_t stride, int_t* idx_1, int_t* idx_2, w_t* weight, int& num_of_sort){
    while (stride < end - start)
    {
        for (int_t i = tid; i < end - start; i += blockDim.x)
        {
            int_t start_i = i * (stride << 1);
            int_t mid_i = min(start_i + stride, end - start);
            int_t end_i = min(end - start, start_i + (stride << 1));
            if (start_i < end - start)
            {

                if (num_of_sort % 2 == 0)
                {
                    bottomUpSort(weight, idx_1, idx_2, start_i, mid_i, end_i);
                }
                else
                {
                    bottomUpSort(weight, idx_2, idx_1, start_i, mid_i, end_i);
                }
            }
        }
        // stride x 2
        stride <<= 1;
        num_of_sort++;
        __syncthreads();
    }
}
template <class int_t, class w_t>
void __global__ sortByWeight(int_t start, int_t end, w_t *src_weight, int_t *src_node, w_t *dest_weight, int_t *dest_node)
{
    extern __shared__ float smem[];
    // address missalignment
    w_t *local_weight = smem;
    int_t *idx_1 = (int_t *)&smem[end - start];
    int_t *idx_2 = (int_t *)&idx_1[end - start];
    int tid = threadIdx.x;

    // recursively load elements into local_arr
    for (int_t i = tid; i < end - start; i += blockDim.x)
    {
        local_weight[i] = src_weight[start + i];
        idx_1[i] = i;
    }
    __syncthreads();

    int num_of_sort = 0;
    int stride = 1;
    mergeSort(start, end, stride, idx_1, idx_2, local_weight, num_of_sort);

#if defined(DEBUG)
    if (tid == 0)
    {
        for (int i = 0; i < end - start; i++)
        {
            int_t idx = num_of_sort % 2 == 0 ? idx_1[i] : idx_2[i];
            printf("[i, idx, node_idx, ts]: [%d, %d, %d, %f] ", i, idx, src_node[start + idx], local_weight[idx]);
        }
        printf("\n");
    }
#endif

    for (int i = tid; i < end - start; i += blockDim.x)
    {
        int_t idx = num_of_sort % 2 == 0 ? idx_1[i] : idx_2[i];
        dest_weight[start + i] = local_weight[idx];
        dest_node[start + i] = src_node[start + idx];
    }
}

template <class int_t, class w_t, bool use_shared_memory>
void __global__ argsort(w_t* weight, int_t* argsort_idx, int_t* argsort_idx_buffer, int_t width, int_t stride, int_t arr_size){
    int num_of_sort = 0;
    if(use_shared_memory){
        extern __shared__ w_t smem[];
        // address missalignment
        w_t* local_weight = smem;
        int_t* idx_1 = (int_t *)&smem[blockDim.x];
        int_t* idx_2 = (int_t *)&idx_1[blockDim.x];
        int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
        int tid = threadIdx.x

        // recursively load elements into local_arr
        for(int i = tid; i < width; i += blockDim.x){
            if(blockIdx.x * width + i < arr_size){
                local_weight[i] = weight[blockIdx.x * width + i];
                idx_1[i] = argsort_idx[blockIdx.x * width + i];
            }
        }
        __syncthreads();
        mergeSort(0, end, stride, idx_1, idx_2, local_weight, num_of_sort);

        for (int i = tid; i < width; i += blockDim.x)
        {
            if(blockIdx.x * width + i < arr_size){
                int_t idx = num_of_sort % 2 == 0 ? idx_1[i] : idx_2[i];
                argsort_idx[blockIdx.x * width + i] = idx;
            }
        }
    }
    else{
        local_weight = weight;
        idx_1 = argsort_idx;
        idx_2 = argsort_idx_buffer;
        int_t start = blockDim.x * width;
        int_t end = min(arr_size, blockDim.x * (width + 1));
        mergeSort(start, end, stride, argsort_idx, argsort_idx_buffer, weight, num_of_sort);
        if(num_of_sort % 2 != 0){
            for(int_t i = start; i < end; i += blockDim.x){
                argsort_idx[i] = argsort_idx_buffer[i];
            }
        }
    }
}

template <class int_t, class float_t>
void __global__ getOutEdgeTimestampCorrespondence(int_t *start_idx, float_t *timestamp, int_t *node_idx, int_t *outIdx, int_t curr_node)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    //  wrap up for edge > blockDim
    int_t start = start_idx[curr_node];
    int_t end = start_idx[curr_node + 1];
    for (int i = idx; i < end - start; i += blockDim.x)
    {
        int_t next_node = node_idx[start + i];
        float next_ts = timestamp[start + i];
        int_t next_start = start_idx[next_node];
        int_t next_end = start_idx[next_node + 1];

        // no neighbors
        if (next_end == next_start)
        {
            outIdx[start + i] = -1;
        }
        else
        {
            // binary search to find the timestamp position in all out edges
            int_t l = 0, r = next_end - 1 - next_start;
            while (l < r)
            {
                int_t mid = l + (r - l) / 2;
                if (timestamp[next_start + mid] >= next_ts)
                {
                    r = mid;
                }
                else
                {
                    l = mid + 1;
                }
            }
            // 1. l = next_end next_ts is larger than all its out edge ts
            // 2. l = [0, next_end)
            // 3. l = -1 next_ts is smaller than all its out edge ts
            if (next_ts <= timestamp[next_start + l])
            {
                outIdx[start + i] = l - 1;
            }
            else
            {
                outIdx[start + i] = l;
            }
        }
    }
}

template <class int_t, class float_t>
void __global__ prefixSumLinear(int_t *start_idx, int_t *node_idx, float_t *timestamp, float_t *buffer, int_t curr_node)
{
    assert(blockDim.x == 1 && "linear prefix sum must have only 1 thread in each block");
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int_t start = start_idx[curr_node];
    int_t end = start_idx[curr_node + 1];

    // wrap up
    for (int i = idx; i < end - start; i++)
    {
        buffer[start + i] = timestamp[start + i] + (i == 0 ? 0 : buffer[start + i - 1]);
    }
}

template <class int_t, class float_t>
void __global__ prefixSumConcurrent(int_t *start_idx, int_t *node_idx, float_t *timestamp, float_t *buffer, int_t curr_node, float ratio)
{
    // printf("I'm in\n");
    extern __shared__ float_t smem[];

    int_t start = start_idx[curr_node];
    int_t end = start_idx[curr_node + 1];
    // printf("s, e: %d %d \n", start, end);
    int tid = threadIdx.x;

    //  wrap up
    for (int i = tid; i < end - start; i += blockDim.x)
    {
        smem[i] = timestamp[end - 1] + ratio * (timestamp[end - 1] - timestamp[start]) - timestamp[start + i];
        // printf("[i, val]: [%d, %f] ", i, smem[i]);
    }
    // printf("\n");
    __syncthreads();

    // reduction
    int stride = 1;
    while (stride < end - start)
    {
        for (int i = tid; i < end - start; i += blockDim.x)
        {
            int idx = stride * (i + 1) * 2 - 1;
            if (idx < end - start)
            {
                smem[idx] += smem[idx - stride];
            }
        }
        stride <<= 1;
        __syncthreads();
    }

    // post scan
    stride >>= 1;
    while (stride > 0)
    {
        for (int i = tid; i < end - start; i += blockDim.x)
        {
            int idx = stride * (i + 1) * 2 - 1;
            if (idx < end - start)
            {
                smem[idx + stride] += smem[idx];
            }
        }
        stride >>= 1;
        __syncthreads();
    }

    // put into buffer
    for (int i = tid; i < end - start; i += blockDim.x)
    {
        buffer[start + i] = smem[i];
    }
}

void cuda_helper(int max_walk_length, int num_walks_per_node, int32_t num_nodes, int32_t num_edges)
{
    int width = 1024;
    int stride = 1;
    in32_t* node_argsort_idx_dev_buffer = nullptr;
    while(width < num_edges){
        int blockSize = min((int) threadBlockSize, width / 2);
        int gridSize = ceil(num_nodes / (float) width);
        int smem_size_per_block = sizeof(int32_t) * 3 * width;
        dim3 grid(gridSize);
        dim3 block(blockSize);
        if(smem_size_per_block > prop.sharedMemPerBlock){
            if(!node_argsort_idx_dev_buffer){
                cudaCheck(cudaMalloc((void**) &node_argsort_idx_dev_buffer, sizeof(int32_t) * num_nodes));
            }
            argsort<int32_t, int32_t, false><<<gridSize, blockSize>>>(node_argsort_idx_dev);
        }
        else{

        }
    }
    if(node_argsort_idx_dev_buffer){
        cudaCheck(cudaFree(node_argsort_idx_dev_buffer));
    }

    int count = 10;
    cudaStream_t *streams = new cudaStream_t[count];
    for (int i = 0; i < count; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    // sort timestamp and node idx
    for (int i = 0; i < num_nodes; i++)
    {
        int32_t start = start_idx_host[i];
        int32_t end = start_idx_host[i + 1];
        if (end - start > 0)
        {
            dim3 grid(1);
            dim3 block(min((int)threadBlockSize, (int)ceil((end - start) / 2.0)));
            int stream_id = i % count;
            // wait previous job to finish
            cudaStreamSynchronize(streams[stream_id]);

            // memcpy
            cudaCheck(cudaMemcpyAsync((void **)&node_idx_dev[start], (void **)&node_idx_host[start], sizeof(int32_t) * (end - start), cudaMemcpyHostToDevice, streams[stream_id]));
            cudaCheck(cudaMemcpyAsync((void **)&timestamp_dev[start], (void **)&timestamp_host[start], sizeof(float) * (end - start), cudaMemcpyHostToDevice, streams[stream_id]));
            sortByWeight<int32_t, float><<<grid, block, sizeof(float) * (end - start) + 2 * sizeof(int32_t) * (end - start), streams[stream_id]>>>(start_idx_host[i], start_idx_host[i + 1], timestamp_dev, node_idx_dev, timestamp_dev_sorted, node_idx_dev_sorted);
            cudaCheck(cudaMemcpyAsync((void **)&node_idx_host_sorted[start], (void **)&node_idx_dev_sorted[start], sizeof(int32_t) * (end - start), cudaMemcpyDeviceToHost, streams[stream_id]));
            cudaCheck(cudaMemcpyAsync((void **)&timestamp_host_sorted[start], (void **)&timestamp_dev_sorted[start], sizeof(float) * (end - start), cudaMemcpyDeviceToHost, streams[stream_id]));
        }
    }

    for (int i = 0; i < count; i++)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;

#if defined(DEBUG)
    printf("----------------sort timestamp--------------------\n");
    // get result
    for (int i = 0; i < num_nodes; i++)
    {
        printf("before:\n [node_idx, timeStamp]:\n");
        for (int j = start_idx_host[i]; j < start_idx_host[i + 1]; j++)
        {
            printf("[%d,%f] ", node_idx_host[j], timestamp_host[j]);
        }
        printf("\n");
        printf("after:\n");
        for (int j = start_idx_host[i]; j < start_idx_host[i + 1]; j++)
        {
            printf("[%d,%f] ", node_idx_host_sorted[j], timestamp_host_sorted[j]);
        }
        printf("\n");
    }
#endif

    // get prefix buffer
    for (int i = 0; i < num_nodes; i++)
    {
        int32_t start = start_idx_host[i];
        int32_t end = start_idx_host[i + 1];
        if (end - start > 0)
        {
            dim3 grid(1);
            dim3 block(min(threadBlockSize, (end - start)));
            prefixSumConcurrent<int32_t, float><<<grid, block, sizeof(float) * (end - start)>>>(start_idx_dev, node_idx_dev_sorted, timestamp_dev_sorted, cdf_buffer_dev, i, extend_ratio);
        }
    }
    cudaDeviceSynchronize();

    cudaMemcpy(cdf_buffer_host, cdf_buffer_dev, sizeof(float) * num_edges, cudaMemcpyDeviceToHost);

#if defined(DEBUG)
    printf("----------------prefix sum--------------------\n");
    for (int i = 0; i < num_nodes; i++)
    {
        float tmp = 0;
        for (int j = start_idx_host[i]; j < start_idx_host[i + 1]; j++)
        {
            tmp += timestamp_host_sorted[start_idx_host[i + 1] - 1] + extend_ratio * (timestamp_host_sorted[start_idx_host[i + 1] - 1] - timestamp_host_sorted[start_idx_host[i]]) - timestamp_host_sorted[j];
            printf("[%d, cuda: %f, gt: %f] ", node_idx_host_sorted[j], cdf_buffer_host[j], tmp);
        }
        printf("\n");
    }
#endif

    // get timestamp correspondence
    for (int32_t i = 0; i < num_nodes; i++)
    {
        int32_t start = start_idx_host[i];
        int32_t end = start_idx_host[i + 1];
        if (end - start > 0)
        {
            dim3 grid(1);
            dim3 block(min(threadBlockSize, (end - start)));
            getOutEdgeTimestampCorrespondence<int32_t, float><<<grid, block>>>(start_idx_dev, timestamp_dev_sorted, node_idx_dev_sorted, mapping_dev, i);
        }
    }
    cudaDeviceSynchronize();

    cudaMemcpy(mapping_host, mapping_dev, sizeof(float) * num_edges, cudaMemcpyDeviceToHost);

#if defined(DEBUG)
    printf("----------------edge correspondence--------------------\n");
    for (int i = 0; i < num_nodes; i++)
    {
        for (int j = start_idx_host[i]; j < start_idx_host[i + 1]; j++)
        {
            int32_t nbr = node_idx_host_sorted[j];
            printf("node: %d ts: %f pos: %d\n neighbor %d\n ts: ", i, timestamp_host_sorted[j], mapping_host[j], node_idx_host_sorted[j]);
            for (int k = start_idx_host[nbr]; k < start_idx_host[nbr + 1]; k++)
            {
                printf("%f ", timestamp_host_sorted[k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
#endif
}