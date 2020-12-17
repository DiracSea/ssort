#include <iostream>
#include <cstdio> 
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
      

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

#include "get_time.h"
#include "samplesort.h"
#include "gen_input.h"

using namespace std;


// test correctness
int main()
{
    int32_t N = 100000000; 
    nsqrt = (int32_t)sqrt(N);
    nlog = (int32_t)log2(N);

    subsize = nsqrt; subnum = nsqrt;  


    int32_t* A = new int32_t[N];
    double c = 1; 
    int threshold = 100000; 
    int block = 100; 
    int m = 1;

    // subsize = subnum = nsqrt; 


    nPivo = new int32_t[N]; // 1st one will be throwed out
    nBuck = new int32_t[N];
    nBuck2 = new int32_t[N];
    nTran = new int32_t[N];
    nScan = new int32_t[N];

    memset(A, 0, N*INT);
    memset(nPivo, 0, N * INT);
    memset(nBuck, 0, N * INT);
    memset(nBuck2, 0, N * INT);
    memset(nTran, 0, N * INT);
    memset(nScan, 0, N * INT);


    cilk_for (int i = 0; i < N; i++) A[i] = hash32(i);

    cout << "A: " << endl;
    cout << A << endl;


    // cilk_for (int i = 0; i < n; i++) A_exp[i] = (int) exp(lambda, A_rand[i]);

    timer t; t.start();
    partial_sort_par(A,N,threshold);
    t.stop(); 

    cout << "partialA: " << endl;
    for(int i = 0; i < 32; i++)
    cout << A[i] << endl;
    cout << "time: " << t.get_total() << endl;

    timer t1; t1.start();
    gen_pivot_par(A,N,c,threshold); 
    t1.stop(); 

    cout << "Pivots: " << endl;
    for(int i = 0; i < 32; i++)
    cout << nPivo[i] << endl;
    cout << "time: " << t1.get_total() << endl;

    timer t2; t2.start();
    count_par(A); 
    t2.stop();

    cout << "Count: " << endl;
    for(int i = 0; i < 32; i++)
    cout << nBuck[i] << endl;
    cout << "time: " << t2.get_total() << endl;

    timer t3; t3.start();
    transpose_and_scan_par(N,threshold,block,m);
    t3.stop();

    cout << "Transpose: " << endl;
    for(int i = 0; i < 32; i++)
    cout << nTran[i] << endl;

    cout << "Scan: " << endl;
    for(int i = 0; i < 32; i++)
    cout << nScan[i] << endl;
    cout << "time: " << t3.get_total() << endl;


    
 

    timer t4; t4.start();
    partial_scan(); 
    to_buckets_par(A,threshold, block); 
    t4.stop();

    cout << "Pre-calculate: " << endl;
    for(int i = 0; i < 32; i++)
    cout << nBuck2[i] << endl;
    cout << "Allocate to Bucket: " << endl;
    for(int i = 0; i < 32; i++)
    cout << nBuck[i] << endl;
    cout << "time: " << t4.get_total() << endl;
    
    timer t5; t5.start();
    total_sort(A); 
    t5.stop();

    cout << "result: " << endl;
    for(int i = 0; i < 32; i++)
    cout << nBuck[i] << endl;
    cout << "time: " << t5.get_total() << endl;

	return 0;
}