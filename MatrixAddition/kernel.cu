#include <iostream>
#include <cmath>

//## Sources
//- Matrix Add based on NVIDIA CUDA samples and common tutorials.
//https://www.youtube.com/watch?v=viTFnXRPnsY&list=PLVVBQldz3m5v1VDhlCyB1DhPfjREsJWmf&index=8

const int N = 10;
__global__ void MatrixAddition(int* a, int* b, int* c, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < size &&
		j < size)
	{
		int index = i * size + j;
		c[index] = a[index] + b[index];
	}
}

int main()
{
	const int matrixSize = N * N * sizeof(int);
	int* h_a = new int[N * N];
	int* h_b = new int[N * N];
	int* h_c = new int[N * N];
	
	for (int i = 0; i < N * N; ++i)
	{
		h_a[i] = 1;
		h_b[i] = 2;
		h_c[i] = 0;
	}

	int* d_a;
	int* d_b;
	int* d_c;
	
	cudaMalloc((void**)&d_a, matrixSize);
	cudaMalloc((void**)&d_b, matrixSize);
	cudaMalloc((void**)&d_c, matrixSize);

	cudaMemcpy(d_a, h_a, matrixSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, matrixSize, cudaMemcpyHostToDevice);

	dim3 blockDim(32, 16);
	dim3 gridDim((int)ceil((float)N / blockDim.x), (int)ceil((float)N / blockDim.y));
	MatrixAddition <<<gridDim, blockDim>>>(d_a, d_b, d_c, N);
	
	cudaDeviceSynchronize();
	cudaMemcpy(h_c, d_c, matrixSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N * N; ++i)
	{
		std::cout << h_c[i] << ", ";
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}