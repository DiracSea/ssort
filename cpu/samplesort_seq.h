#pragma once

#include <math.h> 
#include <algorithm>
#include "scan_seq.h"
#include "gen_input.h"

// global variables, used as storing intermedian ? must clarify??

// pass *** seq verion bugfree
int32_t *Pivots;
int32_t *Buckets; // use pivots as range, also store the result
int32_t *Buckets2; // transpose one
int32_t *Buckets3; // transpose one
int32_t *Scan; // store the scan 
int32_t msqrt; // my sqrt(N)
int32_t mlog; // my log_2(N)

// step one
// seq sample sort in each segment
// why not compressed heavy key in the partition part??
// what is good bucket? uniform distributed???

void partial_sort_seq(int32_t* A) // pass *
{
    // msqrt = (int)sqrt(N); 
    // if (sub < threshold)
 
    // partial sort for segment
    for (int i = 0; i < msqrt; i++)
    {
       std::sort(A + i*msqrt, A + (i + 1)*msqrt);
    }
}

// step two
// simple pivot select
// set c to 4

// what if pivots repeated?
void gen_pivot_seq(int32_t* A, int c) // pass *
{
    // mlog = (int)log2(N);
    int32_t mlen = c*mlog*msqrt; 

    int32_t* mpivot = new int32_t[mlen];


    // mid of each interval
    // pick c*logN nonrepeated temp pivot in each segment
    for (int i = 0; i < msqrt; i++)
        for (int j = 0; j < c*mlog; j++)
        {
            int32_t random = hash32(i*c*mlog + j)%msqrt;
            // get random one from the interval
  
            mpivot[i*c*mlog + j] = A[i*msqrt + random]; 
        }
    // sort with stl sort
    std::sort(mpivot, mpivot + mlen); 

    // store in Pivot, 1st one will be throwed 
    for (int i = 0; i < msqrt; i++) 

        Pivots[i] = mpivot[i*c*mlog]; 
}

// step three
// distribute subarrays into Buckets, according to the Pivots
// exile first element in Pivots, use the sqrtN - 1 pivots to divide bucket
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
// ? what if the element very large
void count_seq(int32_t *A)
{
    for (int i = 0; i < msqrt; i++) // each segment
        // for (int j = 0; j < msqrt; j++) // inner segment
        // {
        //     for (int k = 1; k < msqrt; k++)
        //     {
        //         if(A[i*msqrt+j] <= Pivots[k]) // compare with current pivot
        //         {
        //             Buckets[i*msqrt+k-1]++; 
        //             break;
        //         }
        //     }
        //     // compare with last pivot, to see if it large than it
        //     if(A[i*msqrt+j] > Pivots[msqrt-1])
        //     {
        //         Buckets[i*msqrt+msqrt-1]++; 
        //     }
        // }
        merge_seq(A + i*msqrt, Pivots, Buckets + i*msqrt, msqrt);
}

int* unit(int* A, int row, int column, int n, int N)
{
    return A + (row * n + column) * N / 2;
}

void transpose(int* A, int* B, int a, int b, int n)
{
    if (n == 1)
    {
        *A = *B;

    }
    else {
        transpose(unit(A, 0, 0, a, n), unit(B, 0, 0, b, n), a, b, n / 2);
        transpose(unit(A, 0, 1, a, n), unit(B, 1, 0, b, n), a, b, n / 2);
        transpose(unit(A, 1, 0, a, n), unit(B, 0, 1, b, n), a, b, n / 2);
        transpose(unit(A, 1, 1, a, n), unit(B, 1, 1, b, n), a, b, n / 2);
    }
}

// transpose and scan (scan is modified, right shift all elements)
void transpose_and_scan_seq(int32_t N) 
{
    // sequential transpose
    for (int i = 0; i < msqrt; i++)
        for (int j = 0; j < msqrt; j++)
        
            Buckets2[i*msqrt+j] = Buckets[j*msqrt+i];


    // sequential scan
    scan_seq(Buckets2, Scan, N); 
}
void transpose_and_scan_seq1(int32_t N) 
{

    transpose(Buckets2, Buckets, msqrt, msqrt, msqrt); 

    // sequential scan
    scan_seq(Buckets2, Scan, N); 
}

int reduce(int* A, int n, int threshold) {
    if (n < threshold) {
        int ret = 0;
        for (int i = 0; i < n; i++) ret += A[i];
        return ret;
    }
    int L, R;
    L = reduce(A, n / 2, threshold);
    R = reduce(A + n / 2, n - n / 2, threshold);
    return L + R;
}

// tp reduce the latency of allocate result
// scan the bucket and pass the restult into 3
void partial_scan()
{
    for (int i = 0; i < msqrt; i++) 
    {
        scan_seq(Buckets+i*msqrt, Buckets3+i*msqrt, msqrt); 
    }
}

void to_buckets_seq(int32_t* A) // ****
{
    // ?? Buckets and Buckets2 diff?
    // for (int j = 0; j < msqrt; j++)
    // {
    //     int32_t all = 0;
    //     for (int i = 0; i < msqrt; i++)
    //     {
    //         int32_t tmp = i * msqrt + j; 
    //         //cout << "tmp:" << endl;
    //         //cout << i * msqrt + j << endl;
            

            
        
    //         for (int k = 0; k < Buckets2[tmp]; k++) // use transpose to get length
    //         {

    //             Buckets[Scan[tmp] + k] = A[j*msqrt + all + k]; 

            
    //         }
    //         all += Buckets2[i * msqrt + j];
    //     }
    // }
    partial_scan(); 
    int B = 100; 
    for (int j_b = 0; j_b < msqrt; j_b+=B)
    for (int i_b = 0; i_b < msqrt; i_b+=B)
    {
        int msqrt_b1 = i_b + B > msqrt ? msqrt:i_b+B;
        int msqrt_b2 = j_b + B > msqrt ? msqrt:j_b+B;
        for (int j = j_b; j < msqrt_b2; j += 1)
        for (int i = i_b; i < msqrt_b1; i += 1)
        {
            int tmp = i * msqrt + j;
            int all = Buckets3[j*msqrt+i];

            //for (int c = 0; c < i; c++)
            //{
            //    all += nBuck[j * subsize + c];
            //}

            for (int k = 0; k < Buckets2[tmp]; k++) // use transpose to get length
            {
                Buckets[Scan[tmp] + k] = A[j * msqrt + all + k];
            }
        }
    }

}

// step four, 
// sort every bucket in each bucket
// the new bucket number can be got by transpose
void partial_sort_seq1(int32_t* A) 
{

    int32_t L = 0; int32_t R = 0;
    // partial sort for the bucket we got
    for (int i = 0; i < msqrt; i++)
    {
        for (int j = 0; j < msqrt; j++)
        {
            R += Buckets2[i*msqrt+j];
        }    

        std::sort(Buckets + L, Buckets + R);
        L = R;    
    }

        
}

// total seq algorithm
void ssort_seq(int32_t* A, int32_t N, int c) // N = 10^8; c = 4
{
    // step 1
    partial_sort_seq(A);
    // step 2
    gen_pivot_seq(A,c);
    // step 3
    count_seq(A);
    transpose_and_scan_seq(N);
    to_buckets_seq(A);
    // step 4
    partial_sort_seq1(A);
}