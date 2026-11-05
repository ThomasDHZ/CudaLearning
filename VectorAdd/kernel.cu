#include <iostream>
#include <cmath>

//## Sources
//- Vector add based on NVIDIA CUDA samples and common tutorials.
//https://www.youtube.com/watch?v=k24z0067lv4&list=PLVVBQldz3m5v1VDhlCyB1DhPfjREsJWmf&index=10

__global__ void VectorAdd(const float* a, const float* b, float* c, int n)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n)
	{
		c[i] = a[i] + b[i];
	}
}

int main()
{
	const int n = 10;
	float a[n];
	float b[n];
	float c[n];
	float* d_a;
	float* d_b;
	float* d_c;

	for (int i = 0; i < n; ++i)
	{
		a[i] = (float)i + 1.0f;
		b[i] = 2.0f;
	}

	cudaMalloc(&d_a, n * sizeof(float));
	cudaMalloc(&d_b, n * sizeof(float));
	cudaMalloc(&d_c, n * sizeof(float));
	cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, c, n * sizeof(float), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int gridSize = (int)ceil((float)n / blockSize);
	VectorAdd << <gridSize, blockSize >> > (d_a, d_b, d_c, n);

	cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "Vector A: " << std::endl;
	for (int i = 0; i < n; ++i)
	{
		std::cout << a[i] << (", ");
	}

	std::cout << "\n Vector B: " << std::endl;
	for (int i = 0; i < n; ++i)
	{
		std::cout << b[i] << (", ");
	}

	std::cout << "\n Vector C: " << std::endl;
	for (int i = 0; i < n; ++i)
	{
		std::cout << c[i] << (", ");
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}