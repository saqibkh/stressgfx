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
#include "memoryStress.h"

__global__ void kernel_test(int* d_data, size_t num_elements) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_elements) {
        // Write operation: fill the array with idx
        d_data[idx] = idx;
    }
}

__global__ void kernel_check(int* d_data, size_t num_elements, int* miscompareIndex, int* actualValue, int* expectedValue) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_elements) {
        // Read and check for miscompares
        if (d_data[idx] != idx) {
            *miscompareIndex = idx;  // Store index of the miscompare
	    *actualValue = d_data[idx];
	    *expectedValue = idx;
        }
    }
}

int callMemoryStressKernel(int *d_data, size_t memory_size, size_t num_elements, int testDuration) {

    std::cout << "============= TEST RESULTS FOR MEMORY_STRESS =============" << std::endl;

    // This variable will store the fail count
    int l_fail = 0;

    // Allocate memory to store miscompare info
    int* d_miscompare;
    hipMalloc(&d_miscompare, sizeof(int));
    int h_miscompare = -1;

    // These variables will store the results to compare
    int h_expectedValue, h_actualValue;
    int* d_actualValue;
    int* d_expectedValue;
    hipMalloc(&d_actualValue, sizeof(int));
    hipMalloc(&d_expectedValue, sizeof(int));

    std::cout << "Running test for " << testDuration << " seconds." << std::endl;

    // Start the timer
    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch kernel to write data to GPU memory once
    size_t blockSize = 256;
    size_t gridSize = (num_elements + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(kernel_test, dim3(gridSize), dim3(blockSize), 0, 0, d_data, num_elements);
    hipDeviceSynchronize();

    // Run loop until test duration is completed
    while (true) {
        // Check 4 times
        for (int i = 0; i < 4; i++) {
            // Reset miscompare variable
            h_miscompare = -1;
            hipMemcpy(d_miscompare, &h_miscompare, sizeof(int), hipMemcpyHostToDevice);

            // Launch kernel to check for miscompares
            hipLaunchKernelGGL(kernel_check, dim3(gridSize), dim3(blockSize), 0, 0, 
			    d_data, num_elements, d_miscompare,
			    d_actualValue, d_expectedValue);
            hipDeviceSynchronize();

	    if(CHECK_RESULT == true){
                // Copy the result back to host
                hipMemcpy(&h_miscompare, d_miscompare, sizeof(int), hipMemcpyDeviceToHost);
		// Copy actual and expected value back to host
                hipMemcpy(&h_actualValue, d_actualValue, sizeof(int), hipMemcpyDeviceToHost);
                hipMemcpy(&h_expectedValue, d_expectedValue, sizeof(int), hipMemcpyDeviceToHost);

                if (h_miscompare != -1) {
	    	    // Calculate virtual address
                    uintptr_t virtualAddr = reinterpret_cast<uintptr_t>(&d_data[h_miscompare]);
                    uintptr_t physicalAddr = getPhysicalAddress(virtualAddr);
		    int XOR_RESULT = h_actualValue ^ h_expectedValue;
		    std::cout << "MISCOMPARE --> " <<
                        "Virtual address: " << std::hex << virtualAddr <<
                        ", Physical address: " << std::hex << physicalAddr <<
                        ", Actual_Value: 0x" << std::hex << std::uppercase << h_actualValue <<
                        ", Expected_Value: 0x" << std::hex << std::uppercase << h_expectedValue <<
                        ", XOR: 0x" << std::hex << std::uppercase << XOR_RESULT << std::endl;

                    if(EXIT_ON_MISCOMPARE==true){
                            hipFree(d_miscompare); // Clean up
			    return l_fail;
		    }
		}
            }
	}

        // Check the elapsed time
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = current_time - start_time;
        if (elapsed.count() >= testDuration) {
            std::cout << "Test time (" << testDuration << " seconds) has been reached." << std::endl;
            break;
        }
    }

    // Clean up
    hipFree(d_miscompare);

    return l_fail;
}
