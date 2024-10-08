#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

#include "globals.h"
#include "tests/bandwidth.h"
#include "tests/stress.h"
#include "tests/coherency.h"

// Define the global variables
bool VERBOSE = false;
bool EXIT_ON_MISCOMPARE = false;
bool CHECK_RESULT = false;
bool HOST_PINNED_MEMORY = false;

// Function to convert bool to string "True" or "False"
std::string bool2String(bool value) {
    return value ? "True" : "False";
}

void printUsage() {
    std::cout << "Usage: ./gfxstress [options]\n"
              << "Options:\n"
              << "  -h,  --help            Show this help message and exit\n"
              << "  -w,  --workload <>     Specify workload type (bandwidth, stress, coherency). Default is 'all'.\n"
	      << "  -m,  --memory   <>     Specifies the amount of memory used for each test in bytes (Only used by bandwidth test).\n"
	      << "  -c,  --check           Specifies a bool variable if we need to check for miscompares or not. Default is 'False'.\n"
              << "  -t,  --time     <>     Specify the duration of the test in seconds. Default is '60'.\n"
              << "  -s,  --subtests <>     Subtests are defined seperately for each workload. Default we run 'all'.\n"
	      << "  -v,  --verbose         Enables verbose. Default is set to `False`.\n"
	      << "  --exit_on_miscompare   Exits test in case of a miscompare. Only available with --check.\n"
	      << "  --hostpinnedmemory     Allocates pinned host memory, which is mapped to GPU and can be used for more efficient data transfer between host and device.\n"
	      << "  --memory_percentage    Specifies the percentage of gpu memory to use 0-100 (Only used by stress test).\n";
}


int main(int argc, char* argv[]) {
    
    // This variable will store the fail count
    int l_fail = 0;

    std::string workload;
    for (int i = 1; i < argc; ++i) {
	if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage();
	    return 0;
	} else if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--workload") == 0) {
            workload = argv[i + 1];
	} else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--check") == 0) {
            CHECK_RESULT = true;
	} else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            VERBOSE = true;
	} else if (strcmp(argv[i], "--exit_on_miscompare") == 0) {
            EXIT_ON_MISCOMPARE = true;
	    CHECK_RESULT = true; // Can't have miscompare without checking first
	} else if (strcmp(argv[i], "--hostpinnedmemory") == 0) {
            HOST_PINNED_MEMORY = true;
	}
    }

    if (workload.empty()) {
	std::cout << "Please select atleast one of the workloads to run!\n";
        printUsage();
        return 1;
    }

    if (workload == "bandwidth") {
        l_fail += runBandwidthTest(argc, argv);  // Function defined in bandwidth.cpp
    } else if (workload == "stress") {
        runStressTest(argc, argv);  // Function defined in stress.cpp
    } else if (workload == "coherency") {
        runCoherencyTest(argc, argv);  // Function defined in coherency.cpp
    } else {
        std::cout << "Invalid workload specified.\n";
        printUsage();
        return -1;
    }

    return l_fail;
}

