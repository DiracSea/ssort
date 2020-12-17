#pragma once
#include <algorithm> 


// sequential merging
void merge_seq(int* sub, int* pivo, int* buck, int ssize) // pass *
{
    // sub size is subsize
    // pivo size is subsize - 1


    int i, j;
    i = 0; // subarray idx
    j = 1; // pivots idx


    while (i < ssize && j < ssize)
    {
        if (sub[i] <= pivo[j])
        {
            buck[j-1]++; 
            i++;
        }
        else if (sub[i] > pivo[ssize - 1])
        {
            buck[ssize - 1]++;
            i++; 
        }
        else 
        {
            j++;
        }
    }
}

// parallel merging in O(logn) depth

// how to achieve select k in N sorted arrays???

// implement select k firstly O(logk) divide and conquer
int select_k(int* A, int* B, int m, int n, int k) // pass *
{
    if (k > (m+n) || k < 1) return -1; 

    // let m <= n
    if (m > n) return select_k(B, A, n, m, k); 

    // A is empty, return kth element of B
    if (m == 0) return B[k-1]; 

    // k = 1 return minimal
    if (k == 1) return std::min(A[0], B[0]); 

    // divide and conquer
    int i = std::min(m, k/2), j = std::min(n, k/2); 

    if (A[i-1] > B[j-1]) return select_k(A, B+j, m, n-j, k-j);
    else return select_k(A+i, B, m-i, n, k-i);
}



// parallel merging,
// not be used, just as backup
void merge_par(int* sub, int* pivo, int* buck, int ssize)
{
    int i, j, k; 

}
