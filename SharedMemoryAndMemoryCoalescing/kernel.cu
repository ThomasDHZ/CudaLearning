#include <iostream>
#include <cmath>

//## Sources
//- Shared Memory + Memory Coalescing | Blockwise Prefix Sum Algorithm based on NVIDIA CUDA samples and common tutorials.
//https://www.youtube.com/watch?v=A1RawS1v1j4

__global__ void PartialSumKernel(int* input, int* output, int n)
{
	extern __shared__ int sharedMemory[];

	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x * 2 + tid;

	if (index < n)
	{
		sharedMemory[tid] = input[index] + input[index + blockDim.x];
		__syncthreads();

		for (int stride = 1; stride < blockDim.x; stride *= 2)
		{
			int temp = 0;
			if (tid >= stride) temp = sharedMemory[tid - stride];
			__syncthreads();

			sharedMemory[tid] += temp;
			__syncthreads();
		}
		output[index] = sharedMemory[tid];
	}
}

int main()
{
	const int n = 16;
	const int blockSize = 8;

	int input[n] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
	int output[n];

	int* d_input = nullptr;
	int* d_output = nullptr;
	cudaMalloc(&d_input, n * sizeof(int));
	cudaMalloc(&d_output, n * sizeof(int));
	cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);

	PartialSumKernel <<<(int)n / blockSize, blockSize, n * sizeof(int)>>> (d_input, d_output, n);
	cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; i++)
	{
		std::cout << input[i] << ", ";
	}

	std::cout << std::endl << std::endl;
	for (int i = 0; i < n; i++)
	{
		std::cout << output[i] << ", ";
	}

	cudaFree(d_input);
	cudaFree(d_output);
}