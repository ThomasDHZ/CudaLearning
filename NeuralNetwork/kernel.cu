#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

//## Sources
//- Matrix Add based on NVIDIA CUDA samples and common tutorials.
//https://www.youtube.com/watch?v=4SHIaNaUc4s&list=PLVVBQldz3m5tEiH9MLE4c-9wxxyjaqaLk&index=5

__global__ void layerNorm(const float* a, float* b, int rows, int cols)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows)
	{
		extern __shared__ float shared[];
		
		float* row_data = shared;
		for (int col = threadIdx.y; col < cols; col += blockDim.y)
		{
			row_data[col] = a[row * cols + col];
		}

		float mean = 0.0f;
		for (int col = 0; col < cols; col++)
		{
			mean += row_data[col];
		}
		mean /= cols;

		float variance = 0.0f;
		for (int col = 0; col < cols; col++)
		{
			variance += (row_data[col] - mean) * (row_data[col] - mean);
		}
		variance /= cols;

		float standardDeviation = sqrtf(variance + 1e-7);
		for (int col = threadIdx.y; col < cols; col += blockDim.y)
		{
			b[row * cols + col] = row_data[col] - mean /standardDeviation;
		}
	}
}

int main()
{
	int rows = 10;
	int cols = 10;
	float* a;
	float* b;

	a = new float[rows * cols];
	b = new float[rows * cols];

	for (int x = 0; x < rows; x++)
	{
		for (int y = 0; y < cols; y++)
		{
			a[x * cols + y] = (float)rand() / RAND_MAX;
		}
	}

	float* d_a;
	float* d_b;
	cudaMalloc(&d_a, rows * cols * sizeof(float));
	cudaMalloc(&d_b, rows * cols * sizeof(float));
	cudaMemcpy(d_a, a, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int gridSize = (rows + blockSize - 1) / 1;
	layerNorm << <gridSize, blockSize, rows* cols * sizeof(float) >> > (d_a, d_b, rows, cols);
	cudaMemcpy(b, d_b, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

	for (int x = 0; x < rows; x++)
	{
		for (int y = 0; y < cols; y++)
		{
			std::cout << a[x * cols + y] << ", ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	for (int x = 0; x < rows; x++)
	{
		for (int y = 0; y < cols; y++)
		{
			std::cout << b[x * cols + y] << ", ";
		}
		std::cout << std::endl;
	}

	cudaFree(d_a);
	cudaFree(d_b);
	delete a;
	delete b;

	return 0;
}