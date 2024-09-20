// File: helper/memoryStress.h

#ifndef MEMORYSTRESS_H
#define MEMORYSTRESS_H

// Declaration of a function to perform memory stress operation
int callMemoryStressKernel(int *d_data, size_t memory_size, size_t num_elements, int testDuration);

#endif // MEMORYSTRESS_H
