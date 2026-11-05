#include <iostream>
#include <cmath>

//## Sources
//- Matrix Vector Multiplication based on NVIDIA CUDA samples and common tutorials.
//https://www.youtube.com/watch?v=-7fKipIW4S0&list=PLVVBQldz3m5tEiH9MLE4c-9wxxyjaqaLk&index=7

__global__ void MatrixVectorMultiply(const float* a, const float* x, float* b, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < n)
	{
		float sum = 0.0f;
		for (int j = 0; j < n; ++j)
		{
			sum += a[index * n + j] * x[j];
		}
		b[index] = sum;
	}
}

int main()
{
	const int n = 10;

	float* a;
	float* x;
	float* b;
	
	a = new float[n * n];
	x = new float[n];
	b = new float[n];

	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			a[i * n + j] = 1.0f;
		}
		x[i] = 2.0f;
		b[i] = 0.0f;
	}

	float* d_a;
	float* d_x;
	float* d_b;

	cudaMalloc(&d_a, n * n * sizeof(float));
	cudaMalloc(&d_x, n * sizeof(float));
	cudaMalloc(&d_b, n * sizeof(float));

	cudaMemcpy(d_a, a, n * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int gridSize = (int)ceil((float)n / blockSize);
	MatrixVectorMultiply<<<gridSize, blockSize>>> (d_a, d_x, d_b, n);
	cudaDeviceSynchronize();

	cudaMemcpy(b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < n * n; ++i)
	{
		std::cout << a[i] << ", ";
	}

	std::cout  << std::endl << std::endl;
	for (int i = 0; i < n; ++i)
	{
		std::cout << x[i] << ", ";
	}

	std::cout << std::endl << std::endl;
	for (int i = 0; i < n; ++i)
	{
		std::cout << b[i] << ", ";
	}
	
	cudaFree(d_a);
	cudaFree(d_x);
	cudaFree(d_b);

	delete a;
	delete x;
	delete b;

	return 0;
}