/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits>

#include "support.h"

unsigned int hash32_max = std::numeric_limits<unsigned int>::max();

inline unsigned int hash32(unsigned int a) {
	a = (a+0x7ed50d16) + (a<<12);
	a = (a^0xc761c23c) ^ (a>>19);
	a = (a+0x165697b1) + (a<<5);
	a = (a+0xd3a2646c) ^ (a<<9);
	a = (a+0xfd78a6c5) + (a<<3);
	a = (a^0xb55a4f09) ^ (a>>16);
	if (a<0) a = -a;
	return a;
}

void initVector(unsigned int **vec_h, unsigned int size)
{
    *vec_h = (unsigned int*)malloc(size*sizeof(unsigned int));

    if(*vec_h == NULL) {
        FATAL("Unable to allocate host");
    }

    for (unsigned int i=0; i < size; i++) {
        (*vec_h)[i] = (hash32(i)%10000000);
    }

}

void verify(unsigned int* correct, unsigned int* result, unsigned int num_elements) {

  // // Initialize reference
  // unsigned int* bins_ref = (unsigned int*) malloc(num_bins*sizeof(unsigned int));
  // for(unsigned int binIdx = 0; binIdx < num_bins; ++binIdx) {
  //     bins_ref[binIdx] = 0;
  // }

  // // Compute reference bins
  // for(unsigned int i = 0; i < num_elements; ++i) {
  //     unsigned int binIdx = input[i];
  //     ++bins_ref[binIdx];
  // }

  // Compare to reference bins
  for(unsigned int binIdx = 0; binIdx < num_elements; ++binIdx) {
      // printf("%u: %u/%u, ", binIdx, correct[binIdx], result[binIdx]);
      if(correct[binIdx] != result[binIdx]) {
        printf("TEST FAILED at bin %u, cpu = %u, gpu = %u\n\n", binIdx, correct[binIdx], result[binIdx]);
        exit(0);
      }
  }
  printf("\nTEST PASSED\n");
}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

