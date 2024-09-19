#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstdint>

#include "../../globals.h"
#include "../others/getPhysicalAddress.h"
#include "subtract.h"

__global__ void subtractKernel(int* a, int* b, int* c, size_t size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < (int)size) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void checkMiscompareSubtractKernel(const int* a, const int* b, const int* c, int* miscompareIndex, size_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (int)N) {
        if (a[idx] * b[idx] != c[idx]) {
            *miscompareIndex = idx;  // Set the index of the miscompare
        }
    }
}

int callSubtractKernel(int* d_a, int* d_b, int* d_c, size_t memory_size, int testDuration) {

    // This variable will store the fail count
    int l_fail = 0;

    // Define and allocate memory
    size_t num_elements = memory_size / sizeof(int);
    //size_t bytes = memory_size;

    // Define thread hierarchy
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start;
    int iterations = 0;
    
    // These variables will only be used for checking for miscompares
    int h_miscompareIndex = -1;
    int* d_miscompareIndex;
    // Allocate memory for miscompare index on device
    hipMalloc(&d_miscompareIndex, sizeof(int));
    hipMemcpy(d_miscompareIndex, &h_miscompareIndex, sizeof(int), hipMemcpyHostToDevice);

    // Run the kernel for the specified test duration
    while (std::chrono::duration<double>(end - start).count() < testDuration){
        hipLaunchKernelGGL(subtractKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_a, d_b, d_c, num_elements);
        hipDeviceSynchronize();

	if(CHECK_RESULT == true){
            // Launch kernel to compare results on the GPU
	    hipLaunchKernelGGL(checkMiscompareSubtractKernel, 
			    dim3(blocksPerGrid), 
			    dim3(threadsPerBlock), 
			    0, 0, d_a, d_b, d_c, d_miscompareIndex, num_elements);
            hipDeviceSynchronize();
	    // Copy miscompare index back to host
            hipMemcpy(&h_miscompareIndex, d_miscompareIndex, sizeof(int), hipMemcpyDeviceToHost);

            if (h_miscompareIndex != -1) {
                // Miscompare detected
	        std::cout << "Miscompare detected at index " << h_miscompareIndex << std::endl;
	        // Calculate virtual address
                uintptr_t virtualAddr = reinterpret_cast<uintptr_t>(&d_c[h_miscompareIndex]);
	        std::cout << "Virtual address: " << std::hex << virtualAddr << std::endl;

	        // Get the physical address
	        uintptr_t physicalAddr = getPhysicalAddress(virtualAddr);
                if (physicalAddr != 0) {
		    std::cout << "Physical address: " << std::hex << physicalAddr << std::endl;
	        }

		l_fail += 1;
		
		if(EXIT_ON_MISCOMPARE==true){
		    return l_fail;
		}
	    }
        }
            end = std::chrono::high_resolution_clock::now();
            iterations++;
    }

    // Calculate elapsed time
    double elapsed_time = std::chrono::duration<double>(end - start).count();
    double total_data;
    if(CHECK_RESULT == true){
	// 2 arrays being read, 1 being written
	// This needs to multiplied by 2 as we are doing the same when checking results
        total_data = static_cast<double>(iterations) * memory_size * 2 * 2;
    } else {
	// 2 arrays being read, 1 being written
        total_data = static_cast<double>(iterations) * memory_size * 2;
    }

    // Calculate bandwidth in GB/s
    double bandwidth = total_data / elapsed_time / (1 << 30);

    std::cout << "============= TEST RESULTS FOR SUBTRACT =============" << std::endl;
    std::cout << "Test Duration: " << testDuration << " seconds" << std::endl;
    std::cout << "Number of iterations: " << iterations << std::endl;
    std::cout << "Elapsed Time: " << elapsed_time << " seconds" << std::endl;
    std::cout << "Total Data Transferred: " << total_data / (1 << 30) << " GB" << std::endl;
    std::cout << "Bandwidth (SUBTRACT): " << bandwidth << " GB/s\n" << std::endl;
    return l_fail;
}
