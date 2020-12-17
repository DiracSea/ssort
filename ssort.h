#pragma once
#include <math.h> 
#include <stdint.h> 
#include <algorithm>
#include <assert.h>
#include "scan.h"

// I. select sample
__global__ void select_samples_gpu_kernel(unsigned int *const sample_data, const unsigned int sample_interval, const unsigned int *const src_data)
{
    const unsigned int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
    sample_data[tid] = src_data[tid*sample_interval]; 
}


__host__ void select_samples_gpu(
    unsigned int *const sample_data, 
    const unsigned int sample_interval, 
    const unsigned int num_element, 
    const unsigned int num_sample, 
    const unsigned int *const src_data, 
    const unsigned int num_threads_per_block)
{
    // one block of N threads per sample
    const unsigned int num_block = num_sample/num_threads_per_block; 

    assert((num_block*num_threads_per_block) == num_sample); 

    select_samples_gpu_kernel<<<num_block, num_threads_per_block>>>(
        sample_data, sample_interval, src_data
    );
}

// II. sorting the samples


int cmp(const void* a , const void* b)
{
    unsigned int x = *(unsigned int*) a;
    unsigned int y = *(unsigned int*) b;
    if (x > y) return 1;
    else if (x < y) return -1;
    else return 0;
}


__host__ void sort_sample_cpu(
    unsigned int *const sample_data, 
    const unsigned int num_sample)
{
    
    std::qsort(sample_data, num_sample, sizeof(unsigned int), cmp); 
}

// III. counting the sample bins

__host__ __device__ unsigned int bin_search(
    const unsigned int *const src_data, 
    const unsigned int search_value, 
    const unsigned int num_element)
{
    // take the middle of two sections
    unsigned int size = (num_element >> 1); 
    unsigned int start_idx = 0; 
    bool found = false;

    do
    {
        const unsigned int src_idx = (start_idx+size); 
        const unsigned int test_value = src_data[src_idx];

        if (test_value == search_value) found = true;
        else
        {
            if (search_value > test_value) start_idx = (start_idx+size);
        }

        if (found == false) size >>= 1;
               
    } while ((found == false) && (size != 0));

    return (start_idx + size);
}

// single data point, atomic add
__global__ void count_bin_gpu_kernel(
    const unsigned int num_sample,
    const unsigned int *const src_data,
    const unsigned int *const sample_data,
    unsigned int *const bin_count
)
{
    const unsigned int tid = (blockIdx.x*blockDim.x) + threadIdx.x;

    // read the sample point
    const unsigned int data_to_find = src_data[tid];

    // gain the idx of the elem in the search list
    const unsigned int idx = bin_search(sample_data, data_to_find, num_sample);
    atomicAdd(&bin_count[idx],1);
}

__host__ void count_bin_gpu(
    const unsigned int num_sample,
    const unsigned int num_element,
    const unsigned int *const src_data,
    const unsigned int *const sample_data,
    unsigned int *const bin_count,
    const unsigned int num_thread)
{
    const unsigned int num_block = num_element/num_thread;
    count_bin_gpu_kernel<<<num_block, num_thread>>>(num_sample, src_data, sample_data, bin_count);
}

// IV prefix sum (scan)
__host__ void calc_bin_idx_gpu(
    const unsigned int num_element, 
    const unsigned int *const bin_count, 
    unsigned int *const dest_bin_idx, 
    const unsigned int num_threads_per_block, 
    unsigned int num_block, 
    unsigned int *const block_sum, 
    unsigned int *const block_sum_prefix)
{   
    //add_scan_total_kernel
    if (num_element >= 4096)
    {
        const unsigned int num_thread_total = num_threads_per_block*num_block;
        const unsigned int num_elements_per_thread = num_element/num_thread_total;

        assert((num_elements_per_thread*num_thread_total) == num_element); 

        scan_kernel<<<num_block, num_elements_per_thread>>>(num_elements_per_thread, bin_count, dest_bin_idx, block_sum);
        
        // calc prefix for the block sums
        // single thread

        scan_kernel_single<<<1,1>>>(num_thread_total, block_sum, block_sum_prefix);

        // add prefix sum total back into the original prefix blocks
        // switch to N threads per block
        num_block = num_element/num_elements_per_thread;
        add_scan_total_kernel<<<num_block, num_elements_per_thread>>>(dest_bin_idx, block_sum_prefix);
    }
    else
    {
        // calc prefix for the block sums
        // single thread

        scan_kernel_single<<<1,1>>>(num_element, bin_count, dest_bin_idx);
    }
}

// V sorting into bins

__global__ void sort_to_bin_gpu_kernel(
    const unsigned int num_sample,
    const unsigned int *const src_data,
    const unsigned int *const sample_data,
    unsigned int *const dest_bin_idx_tmp,
    unsigned int *const dest_data)
{
    const unsigned int tid = (blockIdx.x*blockDim.x) + threadIdx.x;

    // read sample point
    const unsigned int data = src_data[tid];

    // bin in src data 
    const unsigned int bin = bin_search(sample_data, data, num_sample);

    // increment the current idx for that bin
    const unsigned int dest_idx = atomicAdd(&dest_bin_idx_tmp[bin],1);

    // write data using the current idx of the correct bin
    dest_data[dest_idx] = data;
}


// VI sorting the bins

__device__ void radix_sort(
    unsigned int *const data, 
    const unsigned int start_idx, 
    const unsigned int end_idx, 
    unsigned int *const sort_tmp_1)
{
    // sort into num_list
    for (unsigned int bit = 0; bit < 32; bit++)
    {
        // mask off all 
        const unsigned int bit_mask = (1 << bit);

        unsigned int base_cnt_0 = start_idx; 
        unsigned int base_cnt_1 = start_idx; 

        for (unsigned int i = start_idx; i < end_idx; i++)
        {
            // fetch test data elem
            const unsigned int elem = data[i];

            // if the elem in the one list
            if ((elem&bit_mask) > 0)
                // copy to the one list
                sort_tmp_1[base_cnt_1++] = elem;
            else 
                data[base_cnt_0++] = elem;
        }
    
    // copy data back to the src from the one's list
    for (unsigned int i = start_idx; i < base_cnt_1; i++)
        data[base_cnt_0++] = sort_tmp_1[i];

    }
}

__global__ void sort_bin_gpu_kernel(
    const unsigned int num_sample,
    const unsigned int num_element, 
    unsigned int *const data, 
    const unsigned int *const sample_data,
    const unsigned int *const bin_count,
    const unsigned int *const dest_bin_idx,
    unsigned int *const sort_tmp)
{
    const unsigned int tid = (blockIdx.x*blockDim.x) + threadIdx.x;

    if (tid != (num_sample-1))
        radix_sort(data, dest_bin_idx[tid], dest_bin_idx[tid+1], sort_tmp);
    else
        radix_sort(data, dest_bin_idx[tid], num_element, sort_tmp); 
}


__host__ void sort_bin_gpu(
    const unsigned int num_sample,
    const unsigned int num_element,
    unsigned int *const data,
    const unsigned int *const sample_data,
    const unsigned int *const bin_count,
    const unsigned int *const dest_bin_idx,
    unsigned int *const sort_tmp,
    const unsigned int num_thread)
{
    const unsigned int num_block = num_sample/num_thread;

    sort_bin_gpu_kernel<<<num_block, num_thread>>>(num_sample, num_element, data,
    sample_data, bin_count, dest_bin_idx, sort_tmp);
}
