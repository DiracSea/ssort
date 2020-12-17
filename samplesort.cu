#include <math.h> 
#include <stdint.h> 
#include <stdio.h> 

#include <algorithm>
#include <assert.h>

#include "ssort.h"
#include "support.h"

using namespace std;

#define MAX_SIZE 1048576
#define MIN_THREAD 32
#define MAX_BLOCK 16
/*
I
__host__ void select_samples_gpu(
    int32_t *const sample_data, 
    const int32_t sample_interval, 
    const int32_t num_elements, 
    const int32_t num_samples, 
    const int32_t *const src_data, 
    const int32_t num_threads_per_block)
II
__host__ void sort_sample_cpu(
    int32_t *const sample_data, 
    const int32_t sample_num)
III
__host__ void count_bin_gpu(
    const int32_t num_sample,
    const int32_t num_element,
    const int32_t *const src_data,
    const int32_t *const sample_data,
    int32_t *const bin_count,
    const int32_t num_thread)
IV
__host__ void calc_bin_idx_gpu(
    const int32_t num_element, 
    const int32_t *const bin_count, 
    int32_t *const dest_bin_idx, 
    const int32_t num_threads_per_block, 
    int32_t num_block, 
    int32_t *const block_sum, 
    int32_t *const block_sum_prefix)
V
__global__ void sort_to_bin_gpu_kernel(
    const int32_t num_sample,
    const int32_t *const src_data,
    const int32_t *const sample_data,
    int32_t *const dest_bin_idx_tmp,
    int32_t *const dest_data)
VI
__host__ void sort_bin_gpu(
    const int32_t num_sample,
    const int32_t num_element,
    int32_t *const data,
    const int32_t *const sample_data,
    const int32_t *const bin_count,
    const int32_t *const dest_bin_idx,
    int32_t *const sort_tmp,
    const int32_t num_thread)
*/
int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int sample_interval, num_element, num_sample, num_threads_per_block, num_thread, num_block;// num_block = block_size
    unsigned int* sample_data; unsigned int* tmp_sample;
    unsigned int* src_data; unsigned int* tmp_data;
    unsigned int* bin_count;
    unsigned int* dest_bin_idx;//dest_bin_idx_tmp;
    unsigned int* dest_bin_idx_tmp; 

    unsigned int* block_sum;
    unsigned int* block_sum_prefix;

    // unsigned int* dest_bin_idx_tmp;
    unsigned int* dest_data; //SORT_TMP
 
    unsigned int* data;unsigned int* sort_tmp;

    cudaError_t cuda_ret;
    

    if(argc == 1) {

        num_threads_per_block = MIN_THREAD;
        num_block = MAX_BLOCK;

    } else if(argc == 2) {

        num_threads_per_block = MIN_THREAD;
        num_block = atoi(argv[1]);

    } else if(argc == 3) {
        
        num_threads_per_block = atoi(argv[1]);
        num_block = atoi(argv[2]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./histogram            # block num: 16, threads/block 32, Input: 1000000"
           "\n    Usage: ./histogram <m>        # block num: m, threads/block 32, Input: 1000000"
           "\n    Usage: ./histogram <m> <n>    # block num: m, threads/block n, Input: 1000000"
           "\n");
        exit(0);
    }

    num_element = MAX_SIZE;

    num_thread = num_threads_per_block*num_block;
    unsigned int msqrt = (unsigned int)sqrt(num_element);
    unsigned int mlog = 16;

    num_sample = msqrt*mlog;
    sample_interval = num_element/num_sample;


    malloc(tmp_data, num_element * sizeof(unsigned int));

    initVector(&tmp_data, num_element);
    // bins_h = (unsigned int*) malloc(num_bins*sizeof(unsigned int));

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n    Number of block num = %u\n, number of thread/block = %u\n", num_element,
        num_block,num_threads_per_block);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**)&src_data, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&sample_data, num_sample * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    // cuda_ret = cudaMalloc((void**)&tmp_sample, num_sample * sizeof(unsigned int));
    // if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&bin_count, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&dest_bin_idx, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&block_sum, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&block_sum_prefix, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&dest_data, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&data, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&dest_bin_idx_tmp, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    // cuda_ret = cudaMalloc((void**)&sort_tmp, num_element * sizeof(unsigned int));
    // if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    malloc(sort_tmp, num_element * sizeof(unsigned int));
    malloc(tmp_sample, num_sample * sizeof(unsigned int));

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(src_data, tmp_data, num_element * sizeof(unsigned int),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy tmp_data to the device");

    cuda_ret = cudaMemset(sample_data, 0, num_sample * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set sample_data memory");

    cuda_ret = cudaMemset(bin_count, 0, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set bin_count memory");

    cuda_ret = cudaMemset(dest_bin_idx, 0, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set bin_count memory");
    cuda_ret = cudaMemset(dest_bin_idx_tmp, 0, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set bin_count memory");

    cuda_ret = cudaMemset(block_sum, 0, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set bin_count memory");
    cuda_ret = cudaMemset(block_sum_prefix, 0, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set bin_count memory");

    cuda_ret = cudaMemset(dest_data, 0, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set bin_count memory");

    cuda_ret = cudaMemset(data, 0, num_element * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set bin_count memory");

    // cuda_ret = cudaMemset(sort_tmp, 0, num_element * sizeof(unsigned int));
    // if(cuda_ret != cudaSuccess) FATAL("Unable to set bin_count memory");

    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch  select sample ----------------------------------------------------------
    printf("Launching select sample..."); fflush(stdout);
    startTime(&timer);

    select_samples_gpu(sample_data, sample_interval, num_element, num_sample, src_data, num_threads_per_block);
    // histogram(in_d, bins_d, num_elements, num_bins);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    cuda_ret = cudaMemcpy(tmp_sample, sample_data, num_sample * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy sample to host");
    cudaDeviceSynchronize();


    // II sorting the samples
    printf("Launching sorting the samples..."); fflush(stdout);
    startTime(&timer);
    sort_sample_cpu(tmp_sample, num_sample);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    cuda_ret = cudaMemcpy(sample_data, tmp_sample, num_sample * sizeof(unsigned int),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy sample_tmp to the device");

    // III counting the sample bins

    printf("Launching counting the sample bins..."); fflush(stdout);
    startTime(&timer);
    count_bin_gpu(num_sample,num_element,src_data,sample_data,bin_count,num_thread*4);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    // IV prefix sum (scan)

    printf("Launching prefix sum..."); fflush(stdout);
    startTime(&timer);
    calc_bin_idx_gpu(num_element,bin_count,dest_bin_idx,num_threads_per_block,num_block,block_sum,block_sum_prefix);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // V sorting into bins
    cuda_ret = cudaMemcpy(dest_bin_idx_tmp, dest_bin_idx, num_element * sizeof(unsigned int),
        cudaMemcpyDeviceToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy dest_bin_idx to dest_bin_idx_tmp");
    cudaDeviceSynchronize();

    printf("Launching sorting into bins..."); fflush(stdout);
    startTime(&timer);

    stopTime(&timer);
    sort_to_bin_gpu_kernel<<<num_block, num_thread>>>(num_sample,src_data,sample_data,dest_bin_idx_tmp,dest_data);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));


    // VI sorting the bins

    printf("Launching sorting the bins..."); fflush(stdout);
    startTime(&timer);
    sort_bin_gpu(num_sample,num_element,data,sample_data,bin_count,dest_bin_idx,dest_data,num_thread);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));



    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(sort_tmp, data, num_element * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
    cudaDeviceSynchronize();

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results...");fflush(stdout);
    // std::qsort(tmp_data, num_element, sizeof(unsigned int), cmp); 
    verify(tmp_data, sort_tmp, num_element);

    // Free memory ------------------------------------------------------------

    cudaFree(sample_data); cudaFree(src_data);
    cudaFree(bin_count); cudaFree(dest_bin_idx);
    cudaFree(dest_bin_idx_tmp); cudaFree(block_sum);
    cudaFree(block_sum_prefix); cudaFree(dest_data);
    cudaFree(data);
    free(tmp_data); free(sort_tmp); free(tmp_sample); 

    return 0;
}

