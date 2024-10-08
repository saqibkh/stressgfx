#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <random>  // Include the random library

__global__ void writeKernel(int* a, int* c, size_t size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        c[idx] = a[idx];
    }
}

void callWriteKernel(int* d_a, int* d_c, int test_duration, size_t memory_size) {
    // Define and allocate memory
    size_t num_elements = memory_size / sizeof(int);
    size_t bytes = memory_size;
    
    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time;

    float bandwidth = 0;
    int total_bytes = 0;
    int count = 0;

    while (std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() < test_duration) {
        hipEvent_t start, stop;
        float elapsedTime;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        
        hipEventRecord(start);
        writeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_c, num_elements);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);
        
        /* Calculate bandwidth for this iteration */
        bandwidth += (bytes * 1.0f) / (elapsedTime / 1000.0f) / (1024.0f * 1024.0f * 1024.0f); // in GB/s
        total_bytes += bytes;
        count++;
        
        hipEventDestroy(start);
        hipEventDestroy(stop);
        
        end_time = std::chrono::high_resolution_clock::now();
    }

    float average_bandwidth = (bandwidth/count); // GB/s
    std::cout << "Average bandwidth for Write: " << average_bandwidth << " GB/s" << std::endl;
}

