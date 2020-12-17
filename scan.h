#pragma once
#include <math.h> 
#include <stdint.h> 
#include <algorithm>
#include <assert.h>

__global__ void scan_kernel(
    const unsigned int num_samples_per_thread, 
    const unsigned int *const bin_count, 
    unsigned int *const prefix_idx, 
    unsigned int *const block_sum)
{
    const unsigned int tid = (blockIdx.x*blockDim.x) + threadIdx.x; 
    const unsigned int tid_offset = tid*num_samples_per_thread; 

    unsigned int prefix_sum; 

    if (tid == 0) prefix_sum = 0;
    else prefix_sum = bin_count[tid_offset-1]; 

    for (unsigned int i = 0; i < num_samples_per_thread; i++)
    {
        prefix_idx[i+tid_offset] = prefix_sum; 
        prefix_sum += bin_count[i+tid_offset];
    }

    // store the block prefixsum as the value from the last element
    block_sum[tid] = prefix_idx[(num_samples_per_thread-1) + tid_offset];
}

__global__ void add_scan_total_kernel(
    unsigned int *const prefix_idx, 
    const unsigned int *const total_count)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    prefix_idx[tid] += total_count[blockIdx.x];
}

__global__ void scan_kernel_single(
    const unsigned int num_sample, 
    const unsigned int *const bin_count,
    unsigned int *const dest_bin_idx)
{
    unsigned int prefix_sum = 0; 

    for (unsigned int i = 0; i < num_samples; i++)
    {
        dest_bin_idx[i] = prefix_sum; 
        prefix_sum += bin_count[i];
    }
}

