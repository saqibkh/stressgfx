#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

#include "bandwidth.h"
#include "../globals.h"
#include "../helper/arithmetic/add.h"  // Include the add.h file

// Function to convert bool to string "True" or "False"
std::string boolToString(bool value) {
    return value ? "True" : "False";
}

// Function to parse command-line arguments and store values in references
void parseArguments(int argc, char* argv[], std::string& subtest, std::string& time, size_t& memory_size, bool* enableVerify) {

    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--time") == 0 || strcmp(argv[i], "-t") == 0) && i + 1 < argc) {
            time = argv[++i]; // Store the value of --time
        } else if ((strcmp(argv[i], "--memory") == 0 || strcmp(argv[i], "-m") == 0) && i + 1 < argc) {
	    memory_size = std::stoull(argv[++i]); // Store the value of --memory_size
	} else if ((strcmp(argv[i], "--subtest") == 0 || strcmp(argv[i], "-s") == 0) && i + 1 < argc) {
            subtest = argv[++i]; // Store the value of --sibtest
	} else if ((strcmp(argv[i], "--check") == 0 || strcmp(argv[i], "-c") == 0) && i + 1 < argc) {
            *enableVerify = true;
	}
    }

    // Print out the values
    std::cout << "======================================================================\n";
    std::cout << "=====================TEST CONFIGURATION===============================\n";
    std::cout << "======================================================================\n";
    std::cout << "Sub Test: " << subtest << std::endl;
    std::cout << "Time: " << time << " seconds" << std::endl;
    std::cout << "Target Memory: " << memory_size/(1024 * 1024) << "MB" << std::endl;
    std::cout << "VERBOSE: " << boolToString(VERBOSE) << std::endl;
    std::cout << "CHECK_RESULT: " << boolToString(CHECK_RESULT) << std::endl;
    std::cout << "EXIT_ON_MISCOMPARE: " << boolToString(EXIT_ON_MISCOMPARE) << std::endl;
    std::cout << "======================================================================\n";
}

void runBandwidthTest(int argc, char* argv[]) {
    std::cout << "\n\nStarting to performing bandwidth test!\n";
    
    // Setting Default args
    std::string subtest = "all";
    std::string time = "60";
    size_t memory_size = 256 * 1024 * 1024; // We are targetting 10MB by default (1024 x 1024)
    bool enableVerify = false;

    // Parse user provided user arguments; and set default variables if not provided
    parseArguments(argc, argv, subtest, time, memory_size, &enableVerify);

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

    // Seed the random number generator
    std::srand(42); // Use a fixed seed for reproducibility
    // Extreme min and max values for int
    int min_val = std::numeric_limits<int>::min();
    int max_val = std::numeric_limits<int>::max();
    // Fill the array with signed random integers
    for (size_t i = 0; i < num_elements; ++i) {
        h_a[i] = min_val + std::rand() % (max_val - min_val + 1);
	h_b[i] = min_val + std::rand() % (max_val - min_val + 1);
	h_c[i] = min_val + std::rand() % (max_val - min_val + 1);
    }

    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);
    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_c, h_c, bytes, hipMemcpyHostToDevice);

    if (subtest == "add") {
        callAddKernel(d_a, d_b, d_c, memory_size, test_duration);
    } else if (subtest == "subtract"){
	std::cout << "This option is still under construction\n";
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
	std::cout << "This option is still under construction\n";
    }

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}
