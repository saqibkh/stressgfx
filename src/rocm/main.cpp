#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

// Function prototypes
void callAddKernel(int* d_a, int* d_b, int* d_c, int test_duration, size_t memory_size);
void callSubtractKernel(int* d_a, int* d_b, int* d_c, int test_duration, size_t memory_size);
void callDivideKernel(int* d_a, int* d_b, int* d_c, int test_duration, size_t memory_size);
void callMultiplyKernel(int* d_a, int* d_b, int* d_c, int test_duration, size_t memory_size);
void callReadKernel(int* d_a, int test_duration, size_t memory_size);
void callWriteKernel(int* d_a, int* d_c, int test_duration, size_t memory_size);

// Function to parse command-line arguments and store values in references
void parseArguments(int argc, char* argv[], std::string& workload, std::string& time, size_t& memory_size) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--workload") == 0 && i + 1 < argc) {
            workload = argv[++i]; // Store the value of --workload
        } else if (strcmp(argv[i], "--time") == 0 && i + 1 < argc) {
            time = argv[++i]; // Store the value of --time
        } else if (strcmp(argv[i], "--memory") == 0 && i + 1 < argc) {
	    memory_size = std::stoull(argv[++i]); // Store the value of --memory_size
	}
    }

    // Print out the values
    std::cout << "Workload: " << workload << std::endl;
    std::cout << "Time: " << time << std::endl;
    std::cout << "Target Memory: " << memory_size << std::endl;
}

int main(int argc, char* argv[]) { 
    std::string workload = "all";
    std::string time = "60";
    size_t memory_size = 256 * 1024 * 1024; // We are targetting 10MB by default (1024 x 1024)

    // Parse user provided user arguments; and set default variables if not provided
    parseArguments(argc, argv, workload, time, memory_size);

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


    if (workload == "all") {
	int num_workloads = 6;
        callAddKernel(d_a, d_b, d_c, ceil(test_duration/num_workloads), memory_size);
	callSubtractKernel(d_a, d_b, d_c, ceil(test_duration/num_workloads), memory_size);
	callMultiplyKernel(d_a, d_b, d_c, ceil(test_duration/num_workloads), memory_size);
	callDivideKernel(d_a, d_b, d_c, ceil(test_duration/num_workloads), memory_size);
	callReadKernel(d_a, ceil(test_duration/num_workloads), memory_size);
	callWriteKernel(d_a, d_c, ceil(test_duration/num_workloads), memory_size);
    } else if (workload == "add") {
    	callAddKernel(d_a, d_b, d_c, test_duration, memory_size);
    } else if (workload == "subtract") {
        callSubtractKernel(d_a, d_b, d_c, test_duration, memory_size);
    } else if (workload == "multiply") {
        callMultiplyKernel(d_a, d_b, d_c, test_duration, memory_size);
    } else if (workload == "divide") {
        callDivideKernel(d_a, d_b, d_c, test_duration, memory_size);
    } else if (workload == "read") {
        callReadKernel(d_a, test_duration, memory_size);
    } else if (workload == "write") {
        callWriteKernel(d_a, d_c, test_duration, memory_size);
    } else {
        std::cout << "No valid workload specified or workload is not supported." << std::endl;
    }

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    return 0;
}
