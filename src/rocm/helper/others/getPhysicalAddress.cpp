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

// Function to get the physical address from the virtual address using /proc/self/pagemap
uintptr_t getPhysicalAddress(uintptr_t virtualAddr) {
    uintptr_t physicalAddr = 0;
    uint64_t value;

    // Open the pagemap file
    std::ifstream pagemap("/proc/self/pagemap", std::ios::binary);
    if (!pagemap) {
        std::cerr << "Failed to open /proc/self/pagemap" << std::endl;
        return physicalAddr;
    }

    // Read the entry corresponding to the virtual address
    uint64_t offset = (virtualAddr / sysconf(_SC_PAGESIZE)) * sizeof(value);
    pagemap.seekg(offset, pagemap.beg);
    pagemap.read(reinterpret_cast<char*>(&value), sizeof(value));
    
    if (value & (1ULL << 63)) { // Page present flag
        physicalAddr = (value & ((1ULL << 55) - 1)) * sysconf(_SC_PAGESIZE);
        physicalAddr |= (virtualAddr & (sysconf(_SC_PAGESIZE) - 1)); // Add the page offset
    } else {
        std::cerr << "Page not present in memory" << std::endl;
    }
    
    return physicalAddr;
}
