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

__global__ void checkMiscompareKernel(const int* a, const int* b, const int* c, bool* miscompareFlag, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (a[idx] + b[idx] != c[idx]) {
            *miscompareFlag = true;  // Set the flag to true if any mismatch is found
        }
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
    
    // These variables will only be used for checking for miscompares
    bool* d_miscompareFlag;
    bool h_miscompareFlag = false;
    hipMalloc(&d_miscompareFlag, sizeof(bool));
    hipMemcpy(d_miscompareFlag, &h_miscompareFlag, sizeof(bool), hipMemcpyHostToDevice);

    // Run the kernel for the specified test duration
    while (std::chrono::duration<double>(end - start).count() < testDuration){
        hipLaunchKernelGGL(addKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_a, d_b, d_c, num_elements);
        hipDeviceSynchronize();

	// Launch kernel to compare results on the GPU
	hipLaunchKernelGGL(checkMiscompareKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_a, d_b, d_c, d_miscompareFlag, num_elements);
        hipDeviceSynchronize();	

	// Copy result back to host
	if(CHECK_RESULT){
            hipMemcpy(&h_miscompareFlag, d_miscompareFlag, sizeof(bool), hipMemcpyDeviceToHost);
            if (h_miscompareFlag) {
                std::cout << "Miscompare detected!" << std::endl;
            }
	} 

        end = std::chrono::high_resolution_clock::now();
        iterations++;
    }

    // Calculate elapsed time
    double elapsed_time = std::chrono::duration<double>(end - start).count();
    double total_data = static_cast<double>(iterations) * memory_size * 2 * 2; // 2 arrays being read, 1 being written
									       // This needs to multiplied by 2 as we are
									       // doing the same when checking results

    // Calculate bandwidth in GB/s
    double bandwidth = total_data / elapsed_time / (1 << 30);


    std::cout << "Test Duration: " << testDuration << " seconds" << std::endl;
    std::cout << "Number of iterations: " << iterations << std::endl;
    std::cout << "Elapsed Time: " << elapsed_time << " seconds" << std::endl;
    std::cout << "Total Data Transferred: " << total_data / (1 << 30) << " GB" << std::endl;
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
    return true;
}
