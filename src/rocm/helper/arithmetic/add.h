// File: helper/arithmetic/add.h

#ifndef ADD_H
#define ADD_H

// Declaration of a function to perform an addition operation
int callAddKernel(int* d_a, int* d_b, int* d_c, size_t memory_size, int testDuration);

#endif // ADD_H
