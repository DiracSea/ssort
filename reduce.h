#pragma once
#include <math.h> 
#include <stdint.h> 
#include <algorithm>

#define MAX_NUM_LIST 1024

__device__ void reduce_gpu(
    const unsigned int *const src_array, 
    unsigned int *const dest_array, 
    const unsigned int num_list,
    const unsigned int num_element, 
    const unsigned int tid
)
{
    const unsigned int num_element_per_list = (num_element/num_list);

    __shared__ unsigned int list_index[MAX_NUM_LIST]; 
    __shared__ unsigned int reduce_val[MAX_NUM_LIST];
    __shared__ unsigned int reduce_idx[MAX_NUM_LIST];

    list_index[tid] = 0; 
    reduce_val[tid] = 0; 
    reduce_idx[tid] = 0; 
    __syncthreads(); 

    for(unsigned int i = 0; i < num_element; i++)
    {
        unsigned int tid_max = (num_list>>1); 
        unsigned int data;

        if (list_index[tid] < num_element_per_list)
        {
            const unsigned int src_idx = tid + (list_index[tid]*num_list); 

            // data from list for givn thread
            data = src_array[src_idx];
        }
        else
        {
            data = -1;
        }
        
        // store the current value and idx
        reduce_val[tid] = data;
        reduce_idx[tid] = tid;

        __syncthreads(); 

        while(tid_max != 0)
        {
            // reduce tid_max from num_list to 0
            if (tid < tid_max)
            {
                const unsigned int val2_idx = tid + tid_max;

                const unsigned int val2 = reduce_val[val2_idx];

                if (reduce_val[tid] > val2)
                {
                    // store smaller value
                    reduce_val[tid] = val2;
                    reduce_idx[tid] = reduce_idx[val2_idx];
                }
            }
            tid_max >>= 1;
            __syncthreads();
        }
        if (tid == 0)
        {
            list_index[reduce_idx[0]]++; 
            dest_array[i] = reduce_val[0];
        }
        __syncthreads();
    }
}