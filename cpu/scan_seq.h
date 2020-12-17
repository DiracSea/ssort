using namespace std; 

void scan_seq(int32_t* In, int32_t* Out, int32_t n) {
	Out[0] = 0; 
	for (int i = 1; i < n; i++) {
		Out[i] = Out[i-1] + In[i-1]; 
	}
}
