#pragma once

using namespace std;


// threshold is 1000000
void scan(int* In, int* Out, int* B, int* C, int n, int threshold) {
	if (n==0) return;
	if (n == 1) {
		Out[0] = In[0];
		return; 
	}
	if (n < threshold) {
	for (int i = 1; i < n; i++) {
		Out[i] = Out[i-1] + In[i]; 
	}
	return; 
	}
	cilk_for (int i = 0; i < n/2; i++) 
		B[i] = In[2*i] + In[2*i+1]; 
	
	scan(B, C, B+n/2, C+n/2, n/2, threshold);	
	Out[0] = In[0];
	
	cilk_for (int i = 1; i < n; i++) {
		if (i%2) Out[i] = C[i/2];
		else Out[i] = C[i/2-1] + In[i];
	}
}
