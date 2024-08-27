#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <random>  // Include the random library

__global__ void divideKernel(float* a, float* b, float* c, size_t size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < size) {
        c[idx] = a[idx] / b[idx];
    }
}

void callDivideKernel(int test_duration, size_t memory_size) {
    // Define and allocate memory
    size_t num_elements = memory_size / sizeof(float);
    size_t bytes = memory_size;
    
    float *h_a = new float[num_elements];
    float *h_b = new float[num_elements];
    float *h_c = new float[num_elements];
    float *d_a, *d_b, *d_c;

    // Initialize random number generator
    /*std::random_device rd;  // Seed for random number generator
    std::mt19937 gen(rd()); // Mersenne Twister RNG
    std::uniform_real_distribution<float> dis(0.0f, 1.0f); // Uniform distribution [0, 1)
    */

    // Initialize input data
    for (size_t i = 0; i < num_elements; ++i) {
        //h_a[i] = 1.0f;
        //h_b[i] = 2.0f;
	//h_a[i] = dis(gen);  // Assign random float to h_a[i]
        //h_b[i] = dis(gen);  // Assign random float to h_b[i]
	h_a[i] = 0xDEADBEEF;
        h_b[i] = 0xDEADBEEF;
    }

    hipMalloc(&d_a, bytes);
    hipMalloc(&d_b, bytes);
    hipMalloc(&d_c, bytes);
    hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time;

    float bandwidth = 0;
    float total_bytes = 0;
    int count = 0;

    while (std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() < test_duration) {
        hipEvent_t start, stop;
        float elapsedTime;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        
        hipEventRecord(start);
        divideKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, num_elements);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);
        
        // Calculate bandwidth for this iteration
        bandwidth += (bytes * 2.0f) / (elapsedTime / 1000.0f) / (1024.0f * 1024.0f * 1024.0f); // in GB/s
        total_bytes += bytes;
        count++;
        
        hipEventDestroy(start);
        hipEventDestroy(stop);
        
        end_time = std::chrono::high_resolution_clock::now();
    }

    float average_bandwidth = (bandwidth/count); // GB/s
    std::cout << "Average bandwidth for Divide: " << average_bandwidth << " GB/s" << std::endl;

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}

