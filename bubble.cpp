#include <iostream>
#include <vector>

#include <cuda_runtime.h>
global void vectorAddition(const int* a, const int* b, int* result, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size)
    {
        result[tid] = a[tid] + b[tid];
    }
}
void performVectorAddition(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& result)
{
    // Size of the vectors
    const int size = a.size();

    // Allocate device memory
    int* dev_a;
    int* dev_b;
    int* dev_result;
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_result, size * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_result, size);

    // Copy the result back to the host
    cudaMemcpy(result.data(), dev_result, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);
}
int main()
{
    // Define the input vectors
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {6, 7, 8, 9, 10};
    const int size = a.size();

    // Define the result vector
    std::vector<int> result(size);

    // Perform the vector addition
    performVectorAddition(a, b, result);

    // Print the result
    std::cout << "Result: ";
    for (const auto& value : result)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}