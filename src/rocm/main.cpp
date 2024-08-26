#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

// Function prototypes
void callAddKernel(int test_duration, size_t memory_size);
void callSubtractKernel(int test_duration, size_t memory_size);

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
    size_t memory_size = 1048576; // We are targetting 1MB by default (1024 x 1024)

    parseArguments(argc, argv, workload, time, memory_size);

    // Convert string inputs to appropriate types
    int test_duration = std::stoi(time);

    if (workload == "all") {
        callAddKernel(test_duration, ceil(memory_size/2));
	callSubtractKernel(test_duration, ceil(memory_size/2));
    } else if (workload == "add") {
	callAddKernel(test_duration, memory_size);
    } else {
        std::cout << "No valid workload specified or workload is not supported." << std::endl;
    }

    return 0;
}
