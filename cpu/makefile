CC = g++
CFLAGS = -O2 -I../c++ -mcx16 -march=native -DCILK -fcilkplus -std=c++11

all:	test_seq test_par main

test_seq:	test_seq.cpp samplesort_seq.h scan_seq.h gen_input.h 
	$(CC) $(CFLAGS) -DNDEBUG test_seq.cpp -o test_seq 

test_par:	test_par.cpp samplesort.h qsort.h scan_par.h  gen_input.h 
	$(CC) $(CFLAGS) -DNDEBUG test_par.cpp -o test_par

main:	main.cpp samplesort.h qsort.h scan_par.h  gen_input.h 
	$(CC) $(CFLAGS) -DNDEBUG main.cpp -o main

clean:
	rm -f test_seq test_par main

