#pragma once

#include <algorithm>
#include "scan_par.h"

using namespace std;

int32_t* A2;
int32_t* B;
int32_t* F;
int32_t* e1;
int32_t* e2;

void qsort(int32_t* A, int32_t start, int32_t end, int threshold, int k) {
	int32_t n = end - start;
	// coarsening
	if (n < threshold) { 
		sort(A, A+n);
		return; 
	}
	if (start >= end-1) return;
	int32_t pivot;
	if (k == 1)
		pivot = A[start];

	// prepare pivot carefully
	else {
		if (n < k) k = n; 
		int32_t* pivots = new int32_t[k]; 
		int32_t random;
		int i = 0;
		while (i < k) {
			random = start + rand() % n;
			pivots[i] = A[random];
			i++;
		}
		sort(pivots, pivots+k);
		pivot = pivots[k/2];
	}

	cilk_for (int i = start; i < end; i++) A2[i] = A[i];
	
	cilk_for (int i = start; i < end; i++) {
		if (A2[i] < pivot) F[i] = 1; else F[i] = 0;
	}
	scan(F+start, B+start, e1+start, e2+start, end-start, threshold);
	
	cilk_for (int i = start+1; i < end; i++) {
		if (F[i]) A[start+B[i]-1] = A2[i];
	}
	
	int32_t x = B[end-1];
	A[start+x] = pivot;

	cilk_for (int i = start+1; i < end; i++) {
		if (A2[i] >= pivot) F[i] = 1; else F[i] = 0;
	}
	scan(F+start, B+start, e1+start, e2+start, end-start, threshold);
	
	cilk_for (int i = start+1; i < end; i++) {
		if (F[i]) A[start+x+B[i]] = A2[i];
	}	
	
	cilk_spawn
	qsort(A, start, start+x, threshold, k);
	qsort(A, start+x+1, end, threshold, k);
	cilk_sync;
	
	return;
}

void qsort1(int* A, int start, int end, int threshold, int k) {
	int n = end - start;
	// coarsening
	if (n < threshold) { 
		sort(A, A+n);
		return; 
	}
	if (start >= end-1) return;
	int pivot;
	if (k == 1)
		pivot = A[start];
	// prepare pivot carefully
	else {
		int* pivots = new int[k]; 
		int random;
		int i = 0;
		while (i < k) {
			random = start + rand() % n;
			pivots[i] = A[random];
			i++;
		}
		sort(pivots, pivots+k);
		pivot = pivots[k/2];
	}

	cilk_for (int i = start; i < end; i++) A2[i] = A[i];
	// smaller than pivot
	cilk_for (int i = start; i < end; i++) {
		if (A2[i] < pivot) F[i] = 1; else F[i] = 0;
	}
	scan(F+start, B+start, e1+start, e2+start, end-start, threshold);
	
	cilk_for (int i = start+1; i < end; i++) {
		if (F[i]) A[start+B[i]-1] = A2[i];
	}

	// third group equal to pivot
	int y = B[end-1]; 

	cilk_for (int i = start; i < end; i++) {
		if (A2[i] == pivot) F[i] = 1; else F[i] = 0;
	}
	scan(F+start, B+start, e1+start, e2+start, end-start, threshold);
	
	cilk_for (int i = start+1; i < end; i++) {
		if (F[i]) A[start+y+B[i]-1] = A2[i];
	}	
	
	// larger than pivot
	int x = B[end-1];
	// A[start+x] = pivot;

	cilk_for (int i = start; i < end; i++) {
		if (A2[i] > pivot) F[i] = 1; else F[i] = 0;
	}
	scan(F+start, B+start, e1+start, e2+start, end-start, threshold);
	
	cilk_for (int i = start+1; i < end; i++) {
		if (F[i]) A[start+y+x+B[i]-1] = A2[i];
	}	


	
	cilk_spawn
	qsort1(A, start, start+y, threshold, k);
	qsort1(A, start+x+y, end, threshold, k);
	cilk_sync;
	
	return;
}