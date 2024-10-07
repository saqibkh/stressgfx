TESTS

1) Bandwidth:
	add
	subtract
	multiply
	divide
	matrix_multiply

2) Stress:
	memoryStress: 
		Writes to a memory location, and then reads it 4 times and compares the
		result for mismatches. Implements a sequential access to memory
	randomMemoryStress: 
		Random memory access stresses memory latency and cache performance, 
		as GPUs are optimized for sequential accesses. The kernel generates
                random memory indices for each thread to access and modify memory
__global__ void random_access_test(float* data, size_t size, int* indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        int random_idx = indices[idx % size]; // Random access pattern
        data[random_idx] += 1.0f; // Read and write memory
    }
}
	

	stridedMemoryAccess(Non-coalesced Memory Access): 
		Strided access patterns force the GPU to access non-sequential memory locations, 
		which reduces memory access efficiency and stresses the memory controller.
		Set different stride values (e.g., 2, 4, 8, etc.) to simulate non-coalesced access
		patterns and stress memory bandwidth efficiency.
__global__ void strided_access_test(float* data, size_t size, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_idx = idx * stride; // Stride to access non-consecutive elements
    if (stride_idx < size) {
        data[stride_idx] += 1.0f;
    }
}

	sustainedMemoryWorkload (Looping Access to Simulate Heavy Workloads): 
		Implement a kernel that continuously loops over memory reads and writes for a
		fixed period, simulating a sustained heavy workload on the memory subsystem.
__global__ void sustained_workload_kernel(float* data, size_t size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int i = 0; i < iterations; i++) {
            data[idx] += 1.0f; // Stress memory through repeated access
        }
    }
}

	
