#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <random>
#include <climits>

#include "bandwidth.h"
#include "../globals.h"
#include "../helper/arithmetic/add.h"  // Include the add.h file
#include "../helper/arithmetic/subtract.h"
#include "../helper/others/boolToString.h"

// Function to parse command-line arguments and store values in references
void parseBandwidthArguments(int argc, char* argv[], std::string& subtest, std::string& time, size_t& memory_size) {

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--time") == 0 || strcmp(argv[i], "-t") == 0) && i + 1 < argc) {
            time = argv[++i]; // Store the value of --time
        } else if ((strcmp(argv[i], "--memory") == 0 || strcmp(argv[i], "-m") == 0) && i + 1 < argc) {
	    memory_size = std::stoull(argv[++i]); // Store the value of --memory_size
	} else if ((strcmp(argv[i], "--subtest") == 0 || strcmp(argv[i], "-s") == 0) && i + 1 < argc) {
            subtest = argv[++i]; // Store the value of --sibtest
	}
    }

    // Print out the values
    std::cout << "======================================================================\n";
    std::cout << "=====================TEST CONFIGURATION (BANDWIDTH)===================\n";
    std::cout << "======================================================================\n";
    std::cout << "Sub Test: " << subtest << std::endl;
    std::cout << "Time: " << time << " seconds" << std::endl;
    std::cout << "Target Memory: " << memory_size/(1024 * 1024) << "MB" << std::endl;
    std::cout << "VERBOSE: " << boolToString(VERBOSE) << std::endl;
    std::cout << "CHECK_RESULT: " << boolToString(CHECK_RESULT) << std::endl;
    std::cout << "EXIT_ON_MISCOMPARE: " << boolToString(EXIT_ON_MISCOMPARE) << std::endl;
    std::cout << "HOST_PINNED_MEMORY: " << boolToString(HOST_PINNED_MEMORY) << std::endl;
    std::cout << "======================================================================\n";
}

int runBandwidthTest(int argc, char* argv[]) {
    std::cout << "\n\nStarting to performing bandwidth test!\n";
    
    // This variable will store the fail count
    int l_fail = 0;

    // Setting Default args
    std::string subtest = "all";
    std::string time = "60";
    size_t memory_size = 256 * 1024 * 1024; // We are targetting 10MB by default (1024 x 1024)

    // Parse user provided user arguments; and set default variables if not provided
    parseBandwidthArguments(argc, argv, subtest, time, memory_size);

    // Convert string inputs to appropriate types
    int test_duration = std::stoi(time);


    /* Initialize memory with random values */
    // First define the pointer to host and device memory that will be filled
    size_t num_elements = memory_size / sizeof(int);
    size_t bytes = memory_size;
    int *h_a = new int[num_elements];
    int *h_b = new int[num_elements];
    int *h_c = new int[num_elements];
    int *d_a, *d_b, *d_c;

    // Use a fixed seed for deterministic results
    std::mt19937 gen(42); // Fixed seed value (42)
    std::uniform_int_distribution<int> dis(INT_MIN, INT_MAX); // Range from INT_MIN to INT_MAX
    // Fill the array with signed random integers
    for (size_t i = 0; i < num_elements; ++i) {
        h_a[i] = dis(gen);
	h_b[i] = dis(gen);
	h_c[i] = dis(gen);

        //std::cout << "h_a[i]: " << h_a[i] << std::endl;
	//std::cout << "h_b[i]: " << h_b[i] << std::endl;
	//std::cout << "h_c[i]: " << h_c[i] << std::endl;
    }

    if(HOST_PINNED_MEMORY == true) {
	/* hipHostMalloc: Allocates pinned host memory, which is mapped to the GPU 
	 * and can be used for more efficient data transfer between host and device.

	Benefit: Uncached Memory Allocation (Using HIP API): In HIP/ROCm, you can 
	allocate memory that bypasses certain cache hierarchies by using pinned 
	memory (hipHostMalloc with hipHostMallocNonCoherent flag). This flag might
	bypass caching and can be useful in cases where you want direct access to 
	host memory without it being cached.

	*/
        hipHostMalloc((void**)&d_a, bytes, hipHostMallocNonCoherent);
        hipHostMalloc((void**)&d_b, bytes, hipHostMallocNonCoherent);
        hipHostMalloc((void**)&d_c, bytes, hipHostMallocNonCoherent);
    } else {
        hipMalloc(&d_a, bytes);
        hipMalloc(&d_b, bytes);
        hipMalloc(&d_c, bytes);
    }

    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_c, h_c, bytes, hipMemcpyHostToDevice);

    int total_tests = 2;
    if (subtest == "add") {
        l_fail += callAddKernel(d_a, d_b, d_c, memory_size, test_duration);
    } else if (subtest == "subtract"){
        l_fail += callSubtractKernel(d_a, d_b, d_c, memory_size, test_duration);
    } else if (subtest == "multiply"){
	std::cout << "This option is still under construction\n";
    } else if (subtest == "divide"){
	std::cout << "This option is still under construction\n";
    } else if (subtest == "read"){
	std::cout << "This option is still under construction\n";
    } else if (subtest == "write"){
	std::cout << "This option is still under construction\n";
    } else if (subtest == "matrix_multiply"){
	std::cout << "This option is still under construction\n";
    } else if (subtest == "all"){
        l_fail += callAddKernel(d_a, d_b, d_c, memory_size, test_duration/total_tests);
	l_fail += callSubtractKernel(d_a, d_b, d_c, memory_size, test_duration/total_tests);
    }

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    return l_fail;
}
