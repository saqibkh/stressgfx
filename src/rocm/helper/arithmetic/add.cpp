#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <random>  // Include the random library

#include "../../globals.h"
#include "add.h"

__global__ void addKernel(int* a, int* b, int* c, size_t size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int callAddKernel(int* d_a, int* d_b, int* d_c, size_t memory_size, int testDuration) {
    // Define and allocate memory
    size_t num_elements = memory_size / sizeof(int);
    size_t bytes = memory_size;

    // Define thread hierarchy
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start;
    int iterations = 0;

    // Run the kernel for the specified test duration
    while (std::chrono::duration<double>(end - start).count() < testDuration){
        hipLaunchKernelGGL(addKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_a, d_b, d_c, num_elements);
        hipDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        iterations++;
    }

    // Calculate elapsed time
    double elapsed_time = std::chrono::duration<double>(end - start).count();
    double total_data = static_cast<double>(iterations) * memory_size * 2; // 2 arrays being read, 1 being written

    // Calculate bandwidth in GB/s
    double bandwidth = total_data / elapsed_time / (1 << 30);


    std::cout << "Test Duration: " << testDuration << " seconds" << std::endl;
    std::cout << "Number of iterations: " << iterations << std::endl;
    std::cout << "Elapsed Time: " << elapsed_time << " seconds" << std::endl;
    std::cout << "Total Data Transferred: " << total_data / (1 << 30) << " GB" << std::endl;
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
    return true;
}
