#include <iostream>
#include <cstdio> 
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#include "get_time.h"
#include "samplesort_seq.h"
#include "gen_input.h"


// test correctness
using namespace std;
#define INT sizeof(int32_t) 

int main(int argc, char** argv)
{
    int32_t n = atoi(argv[1]);
    int32_t N = 100000000; 
    msqrt = (int32_t)sqrt(N);
    mlog = (int32_t)log2(N);
    int32_t* A = new int32_t[N];
    int c = 1; 


    Pivots = new int32_t[N]; // 1st one will be throwed out
    Buckets = new int32_t[N]; 
    Buckets2 = new int32_t[N];
    Buckets3 = new int32_t[N];
    Scan = new int32_t[N];

    // ** very important
    memset(A, 0, N*INT);
    memset(Pivots, 0, N * INT);
    memset(Buckets, 0, N * INT);
    memset(Buckets2, 0, N * INT);
    memset(Buckets3, 0, N * INT);
    memset(Scan, 0, N * INT);

    cilk_for (int i = 0; i < N; i++) A[i] = hash32(i);

    cout << "A: " << endl;
    cout << A << endl;


    // cilk_for (int i = 0; i < n; i++) A_exp[i] = (int) exp(lambda, A_rand[i]);

    timer t; t.start();
    partial_sort_seq(A); 
    t.stop(); 

    cout << "partialA: " << endl;
    for(int i = 0; i < 32; i++)
    cout << A[i] << endl;
    cout << "time: " << t.get_total() << endl;

    timer t1; t1.start();
    gen_pivot_seq(A,c);
    t1.stop(); 

    cout << "Pivots: " << endl;
    for(int i = 0; i < 32; i++)
    cout << Pivots[i] << endl;
    cout << "time: " << t1.get_total() << endl;

    timer t2; t2.start();
    count_seq(A);
    t2.stop();

    cout << "Count: " << endl;
    for(int i = 0; i < 32; i++)
    cout << Buckets[i] << endl;
    cout << "time: " << t2.get_total() << endl;

    timer t3; t3.start();
    transpose_and_scan_seq(N);
    t3.stop();

    cout << "Transpose: " << endl;
    for(int i = 0; i < 32; i++)
    cout << Buckets2[i] << endl;

    cout << "Scan: " << endl;
    for(int i = 0; i < 32; i++)
    cout << Scan[i] << endl;
    cout << "time: " << t3.get_total() << endl;


    timer t4; t4.start();
    to_buckets_seq(A);
    t4.stop();

    cout << "Partial Scan: " << endl;
    for(int i = 0; i < 32; i++)
    cout << Buckets3[i] << endl;

    cout << "Allocate to Bucket: " << endl;
    for(int i = 0; i < 32; i++)
    cout << Buckets[i] << endl;
    cout << "time: " << t4.get_total() << endl;
    
    timer t5; t5.start();
    partial_sort_seq1(A);
    t5.stop();

    cout << "result: " << endl;
    for(int i = 0; i < 32; i++)
    cout << Buckets[i] << endl;
    cout << "time: " << t5.get_total() << endl;

    return 0;
}