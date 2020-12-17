#pragma once

#include <math.h> 
#include <algorithm>
#include "scan_par.h"

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#include "qsort.h"
#include "gen_input.h"

#define INT sizeof(int32_t) 
// parallel version

int32_t *nPivo;
int32_t *nBuck;
int32_t *nBuck2;
int32_t *nTran;
int32_t *nScan;
int32_t nsqrt;
int32_t nlog; 

// self-define in the main function
int32_t subsize; // times of sqrtN; also the pivot size
int32_t subnum; // subnum = N/subsize


// step 1, similar to seq, since the sqrtN is small
// self-define subarray size


// quicksort parameter is tested
// k pivot = 20
// threshold = 100000

void merge_seq(int32_t* sub, int32_t* pivo, int32_t* buck, int32_t ssize) // pass *
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

void scan_seq(int32_t* In, int32_t* Out, int32_t n) {
	Out[0] = 0; 
	for (int i = 1; i < n; i++) {
		Out[i] = Out[i-1] + In[i-1]; 
	}
}

void quicksort(int32_t* A, int32_t start, int32_t end, int threshold, int k)
{
    int N = end - start; 
    A2 = new int32_t[N];
	B = new int32_t[N];
	F = new int32_t[N];
	e1 = new int32_t[N];
	e2 = new int32_t[N];

    memset(A2, 0, N * INT);
    memset(B, 0, N * INT);
    memset(F, 0, N * INT);
    memset(e1, 0, N * INT);
    memset(e2, 0, N * INT);

    qsort(A, start, end, threshold, k);

    delete[] A2; 
    delete[] B; 
    delete[] F; 
    delete[] e1; 
    delete[] e2; 
}
// threshold is 100000
void partial_sort_par(int32_t* A, int32_t N, int threshold)
{
    // subsize will be floating arround nsqrt
    // also add the threshold for very large subsize i.e. 100000
    // parallel partial sort
    // if (subsize < threshold)
        cilk_for (int i = 0; i < subnum; i++)
        {
            std::sort(A + i*subsize, A + (i + 1)*subsize);
        }
    // else
    // {
    //     // parallel quicksort
    //     cilk_for (int i = 0; i < subnum; i++)
    //     {
    //         int k = 20; 
    //         quicksort(A, i*subsize, (i + 1)*subsize, threshold, k);
    //     } 
    // }
}

// step 2, pivot, 
void select_sample(int32_t* A, int32_t* B, int32_t num, int32_t sample)
{
    int b = 100;

    cilk_for (int i_b = 0; i_b < num; i_b+=b)
    for (int j_b = 0; j_b < sample; j_b+=b)
    {
    int b1 = i_b + b > num ? num:i_b+b;
    int b2 = j_b + b > sample ? sample:j_b+b;

    for (int i = i_b; i < b1; i += 1)
    for (int j = j_b; j < b2; j += 1)
    {
        int32_t random = hash32(i*sample + j)%num; 
        B[i*sample + j] = A[i*num + random]; 
    }  
    }  
}


// self-define pivot number
void gen_pivot_par(int32_t* A, int32_t N, double cc, int threshold) 
{
    // cc floating around sqrt(logN)
    int32_t nlen = subnum*nlog*cc; // new length

    int32_t* npivot = new int32_t[nlen]; 

    // pick cc*nlog nonrepeated pivot in each segment


    // select every cclogn element range in each segment
    select_sample(A, npivot, subnum, cc*nlog);

    // sort temp pivot
    // if (nlen < threshold)
    // {
        std::sort(npivot, npivot + nlen); 
    // }
    // else
    // {
    //     int k = 20; 
    //     quicksort(npivot, 0, nlen, threshold, k);
    // }
    

    // store in Pivot
    // nlen/subsize
    // pivot lenght is subsize

    cilk_for (int i = 0; i < subsize; i++) 
    {
        int idx = (int)i*cc*nlog;
        nPivo[i] = npivot[idx]; 
    }
}



// remove duplicate. 
int32_t remove_dup(int32_t* A, int32_t n) 
{ 
    if (n==0 || n==1) 
        return n; 
    // To store index of next unique element 
    int j = 0; 
  
    // Just another updated index 
    for (int i=0; i < n-1; i++) 
        if (A[i] != A[i+1]) 
            A[j++] = A[i]; 
  
    A[j++] = A[n-1]; 
  
    return j; 
} 
void gen_pivot_nodup(int32_t* A, int32_t N, int cc, int threshold) 
{
    // cc floating around sqrt(logN)
    int32_t nlen = subnum*nlog*cc; // new length

    int32_t *npivot = new int32_t[nlen]; 
    int32_t *tpivot = new int32_t[nlen]; 

    // pick cc*nlog nonrepeated pivot in each segment


    // select every cclogn element range in each segment
    select_sample(A, npivot, subnum, cc*nlog);

    // sort temp pivot

        std::sort(npivot, npivot + nlen); 

    // else
    // {
    //     int k = 20; 
    //     quicksort(npivot, 0, nlen, threshold, k);
    // }
    std::copy(npivot, npivot+nlen, tpivot); 
    int32_t size = remove_dup(npivot, N); 
    

    cilk_for (int32_t i = 0; i < subsize; i++) 
    {
        int idx = (int)i*cc*nlog;
        nPivo[i] = npivot[idx]; 
    }
    

}



// step 3
// differ from seq version
// use a merge to count the bucket
// use divide and conquer for transposing


// counting for parallel version
void count_par(int32_t *A)
{
    cilk_for (int i = 0; i < subnum; i++) 
    {
        merge_seq(A + i*subsize, nPivo, nBuck + i*subsize, subsize);
    }
}


// threshold is 1000000
// Scan


// two scan policies
// 1. scan all
// 2. seperately scan, for better locality
void scan_trans(int32_t N, int32_t threshold, int m)
{
    if (m == 1)
    {
        int32_t* e1 = new int32_t[N-1]; 
        int32_t* e2 = new int32_t[N-1]; 
        scan(nTran, nScan+1, e1, e2, N-1, threshold); 
    }
        
    else
    {
        int32_t* e1 = new int32_t[subsize]; 
        int32_t* e2 = new int32_t[subsize]; 

        cilk_for (int32_t i = 0; i < subnum; i++) 
        {
            scan(nTran+i*subsize, nScan+i*subsize, e1, e1, subsize, threshold); 
        }
    }
}

// transpose by blocking
// can take any matrix
void transpose_block(int B) // pass *
{
    cilk_for (int i_b = 0; i_b < subsize; i_b += B)
    {
        cilk_for (int j_b = 0; j_b < subnum; j_b += B)
        {
            int subsize_b = i_b + B > subsize ? subsize:i_b+B;
            int subnum_b = j_b + B > subnum ? subnum:j_b+B;
            for (int i = i_b; i < subsize_b; i += 1)
                for (int j = j_b; j < subnum_b; j += 1)
                {
                    nTran[i*subnum+j] = nBuck[j*subsize+i]; 
                }
        }
    }

    
}

// transpose by divide and conquer
// only square
int32_t* unit(int32_t* A, int row, int column, int n, int32_t N) 
{
    return A + (row*n + column)*N/2;
}

void transpose(int32_t* A, int32_t* B, int a, int b, int32_t n)
{
    if (n == 1)
    {
        *A = *B; 
    }
    else
    {
        cilk_spawn transpose(unit(A, 0, 0, a, n), unit(B, 0, 0, b, n), a, b, n/2); 
        cilk_spawn transpose(unit(A, 0, 1, a, n), unit(B, 1, 0, b, n), a, b, n/2); 
        cilk_spawn transpose(unit(A, 1, 0, a, n), unit(B, 0, 1, b, n), a, b, n/2); 
        transpose(unit(A, 1, 1, a, n), unit(B, 1, 1, b, n), a, b, n/2); 
        cilk_sync;
    }
}

void transpose_and_scan_par(int32_t N, int threshold, int block, int m)
{
    if (m == 2)
        transpose(nTran, nBuck, subsize, subnum, subsize); 
               
    else
        transpose_block(block); 
    
    scan_trans(N, threshold, m);
}

// reduce

int reduce(int* A, int n, int threshold) {
	if (n < threshold) {
	int ret = 0; 
	for (int i = 0; i < n; i++) ret += A[i]; 
	return ret; 
	}
	int L, R; 
	L = cilk_spawn reduce(A, n/2, threshold); 
	R = reduce(A+n/2, n-n/2, threshold); 
	cilk_sync; 
	return L+R;
}

void partial_scan()
{

    cilk_for (int i = 0; i < subnum; i++) 
    {
        scan_seq(nBuck+i*subsize, nBuck2+i*subsize, subsize); 
    }
}

void to_buckets_par(int32_t* A, int32_t threshold, int B) 
{
    // cilk_for(int i = 0; i < subsize; i++)
    // {
    //     for (int j = 0; j < subnum; j++)
    //     {
    //         int32_t tmp = i * subnum + j;
    //         int32_t all = 0; /// add all 

    //         // for (int c = 0; c < i; c++)
    //         // {
    //         //     all += nBuck[j * subnum + c];
    //         // }

    //         all = reduce(nBuck+j*subnum, i, threshold);

    //         for (int k = 0; k < nTran[tmp]; k++) // use transpose to get length
    //         {
    //             nBuck2[nScan[tmp] + k] = A[j* subsize + all + k];
    //         }
    //     }
    // }

    cilk_for (int j_b = 0; j_b < subnum; j_b+=B)
    for(int i_b = 0; i_b < subsize; i_b+=B)
       {
            int msqrt_b1 = i_b + B > subsize ? subsize:i_b+B;
            int msqrt_b2 = j_b + B > subnum ? subnum:j_b+B;
            for (int j = j_b; j < msqrt_b2; j += 1)
            for (int i = i_b; i < msqrt_b1; i += 1)
            {
            int tmp = i * subnum + j;

            // pre-processing
            int all = nBuck2[j*subsize+i];

            //for (int c = 0; c < i; c++)
            //{
            //    all += nBuck[j * subsize + c];
            //}
            // all = reduce(nBuck + j * subnum, i, 100000);

            for (int k = 0; k < nTran[tmp]; k++) // use transpose to get length
            {
                nBuck[nScan[tmp] + k] = A[j* subsize + all + k];
            }
            }
       }

}

void partial_reduce()
{

    cilk_for (int i = 0; i < subsize; i++) 
    {
        nBuck2[i] = reduce(nTran + i * subnum, subnum, 100000);
    }
}

void total_sort(int32_t* A) 
{
    partial_reduce(); 
    int32_t L = 0; int32_t R = 0;
    // partial sort for the bucket we got
    cilk_for (int i = 0; i < subsize; i++)
    {

        R += nBuck2[i]; 

        std::sort(nBuck + L, nBuck + R);
        L = R;    
    }

        
}

void ssort_par(int32_t* A, int32_t N, int c, int threshold, int block, int m)
{
    partial_sort_par(A,N,threshold); 
    gen_pivot_par(A,N,c,threshold); 
    count_par(A); 
    transpose_and_scan_par(N,threshold,block,m);
    partial_scan();
    to_buckets_par(A,threshold,block); 
    total_sort(A); 

}