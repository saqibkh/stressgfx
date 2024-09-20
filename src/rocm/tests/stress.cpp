#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <random>
#include <climits>
#include <hip/hip_runtime.h>

#include "stress.h"
#include "../globals.h"
#include "../helper/others/boolToString.h"
#include "../helper/stress/memoryStress.h"

// Function to parse command-line arguments and store values in references
void parseStressArguments(int argc, char* argv[], std::string& subtest, std::string& time, float& memory_percentage) {

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--time") == 0 || strcmp(argv[i], "-t") == 0) && i + 1 < argc) {
            time = argv[++i]; // Store the value of --time
        } else if ((strcmp(argv[i], "--memory_percentage") == 0) && i + 1 < argc) {
            memory_percentage = std::stoull(argv[++i]); // Store the value of --memory_size
	    memory_percentage = memory_percentage/100;
        } else if ((strcmp(argv[i], "--subtest") == 0 || strcmp(argv[i], "-s") == 0) && i + 1 < argc) {
            subtest = argv[++i]; // Store the value of --sibtest
        }
    }

    // Print out the values
    std::cout << "======================================================================\n";
    std::cout << "=====================TEST CONFIGURATION (STRESS)======================\n";
    std::cout << "======================================================================\n";
    std::cout << "Sub Test: " << subtest << std::endl;
    std::cout << "Time: " << time << " seconds" << std::endl;
    std::cout << "Memory Percentage: " << (memory_percentage*100) << "%" << std::endl;
    std::cout << "VERBOSE: " << boolToString(VERBOSE) << std::endl;
    std::cout << "CHECK_RESULT: " << boolToString(CHECK_RESULT) << std::endl;
    std::cout << "EXIT_ON_MISCOMPARE: " << boolToString(EXIT_ON_MISCOMPARE) << std::endl;
    std::cout << "HOST_PINNED_MEMORY: " << boolToString(HOST_PINNED_MEMORY) << std::endl;
    std::cout << "======================================================================\n";
}


int runStressTest(int argc, char* argv[]) {
    std::cout << "Starting to performing stress test!\n";

    int l_fail = 0;

    // Parse additional arguments as needed
    std::cout << "Checking user supplied input arguments!\n";

    // Setting Default args
    std::string subtest = "all";
    std::string time = "60";
    float memory_percentage = 0.9;

    // Parse user provided user arguments; and set default variables if not provided
    parseStressArguments(argc, argv, subtest, time, memory_percentage);
    //std::cout << "memory_percentage: " << memory_percentage << std::endl;

    // Convert string inputs to appropriate types
    int test_duration = std::stoi(time);

    // Get GPU memory information
    size_t totalMem = 0;
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    totalMem = props.totalGlobalMem;
    //std::cout << "totalMem: " << totalMem << std::endl;

    size_t memory_size_bytes = totalMem * memory_percentage;
    //std::cout << "memory_size_bytes: " << memory_size_bytes << std::endl;

    size_t mem_size = static_cast<size_t>(memory_size_bytes);
    //std::cout << "mem_size: " << mem_size << std::endl;
    
    // Allocate memory_percentage of GPU memory
    int* d_data;
    hipMalloc(&d_data, mem_size);

    size_t num_elements = mem_size / sizeof(int);
    std::cout << "Allocated " << static_cast<float>(mem_size)/(1024*1024*1024) << " Gigabytes on the GPU, with " << num_elements << " elements." << std::endl;

    int total_tests = 1;
    if (subtest == "memoryStress") {
        l_fail += callMemoryStressKernel(d_data, mem_size, num_elements, test_duration);
    } else if (subtest == "TEST1") {
        std::cout << "This option is still under construction\n";
    } else if (subtest == "TEST2") {
        std::cout << "This option is still under construction\n";
    } else if (subtest == "TEST3") {
        std::cout << "This option is still under construction\n";
    } else if (subtest == "all") {
	l_fail += callMemoryStressKernel(d_data, mem_size, num_elements, test_duration/total_tests);
    }

    // Cleanup
    hipFree(d_data);

    return l_fail;
}
