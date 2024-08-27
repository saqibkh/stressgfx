#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>
#include <random>  // Include the random library

__global__ void matrixMultiply(int* a, int* b, int* c, size_t size) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < size && col < size) {
        float value = 0.0f;
        for (int k = 0; k < size; k++) {
            value += a[row * size + k] * b[k * size + col];
        }
        c[row * size + col] = value;
    }
}

void callMatrixMultiplyKernel(int* d_a, int* d_b, int* d_c, int test_duration, size_t memory_size) {
    // Define and allocate memory
    size_t num_elements = memory_size / sizeof(int);
    size_t bytes = memory_size;
    
    // Kernel configuration
    MATRIX_SIZE = sqrt(memory_size) 
    std::cout << "Running matrix multiplication kernel..." << std::endl;
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((MATRIX_SIZE + 15) / 16, (MATRIX_SIZE + 15) / 16);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time;

    float bandwidth = 0;
    int total_bytes = 0;
    int count = 0;

    while (std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() < test_duration) {
        hipEvent_t start, stop;
        float elapsedTime;
        hipEventCreate(&start);
        hipEventCreate(&stop);
        
        hipEventRecord(start);
        addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, num_elements);
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&elapsedTime, start, stop);
        
        /* Calculate bandwidth for this iteration */
        bandwidth += (bytes * 2.0f) / (elapsedTime / 1000.0f) / (1024.0f * 1024.0f * 1024.0f); // in GB/s
        total_bytes += bytes;
        count++;
        
        hipEventDestroy(start);
        hipEventDestroy(stop);
        
        end_time = std::chrono::high_resolution_clock::now();
    }

    float average_bandwidth = (bandwidth/count); // GB/s
    std::cout << "Average bandwidth for Matrix Multiply: " << average_bandwidth << " GB/s" << std::endl;
}

