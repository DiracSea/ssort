#pragma once



// generate all the three input types
// random,  exponential, zipfan x
int32_t hash32_max = INT32_MAX;

inline int32_t hash32(int32_t a) {
	a = (a+0x7ed55d16) + (a<<12);
	a = (a^0xc761c23c) ^ (a>>19);
	a = (a+0x165667b1) + (a<<5);
	a = (a+0xd3a2646c) ^ (a<<9);
	a = (a+0xfd7046c5) + (a<<3);
	a = (a^0xb55a4f09) ^ (a>>16);
	if (a<0) a = -a;
	return a;
}


// exponential
// lambda = 1.0/mean; 
double expon(double lambda, int random){

    double u;

    u = (double)random / (hash32_max + 1.0);

    return -log(1- u) / lambda;
}


// zipfian generator
// the result is distrbution, [0,1]
// need to transfer into data
// k is rank, N is cardinality
// skew = 0 is uniform distribution, the larger skew, the  more curve bias 
double zipfian(int32_t k, int32_t N, double skew) {

    double base = 0;

    for (int k = 1; k <= N; k++) base += std::pow(k, -1*skew);

    return std::pow(k, -1*skew)/base; // need to scale N

}
