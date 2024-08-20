#include <stdio.h>
#include "cuda_runtime.h"

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    // Launch the kernel with a single thread
    cuda_hello<<<1,1>>>();

    // Check for any errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for the GPU to finish before accessing the result
    cudaDeviceSynchronize();

    return 0;
}
