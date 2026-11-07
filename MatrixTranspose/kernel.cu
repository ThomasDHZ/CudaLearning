#include <iostream>
#include <cmath>

//## Sources
//- Matrix Add based on NVIDIA CUDA samples and common tutorials.
//https://www.youtube.com/watch?v=IPnUJsc4upE&list=PLVVBQldz3m5tEiH9MLE4c-9wxxyjaqaLk&index=4

__global__ void MatrixTranspose(const float* input, float* output, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width &&
		y < height)
	{
		int inputIndex = y * width + x;
		int outputIndex = x * height + y;
		output[outputIndex] = input[inputIndex];
	}
}

int main()
{
	int width = 10;
	int height = 10;

	float* h_input = new float[width * height];
	float* h_output = new float[width * height];
	for (int x = 0; x < width * height; x++)
	{
		h_input[x] = (float)x;
	}

	float* d_input = nullptr;
	float* d_output = nullptr;
	cudaMalloc(&d_input, width * height * sizeof(float));
	cudaMalloc(&d_output, width * height * sizeof(float));
	cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blocksize(32, 32);
	dim3 gridsize((width + blocksize.x - 1) / blocksize.x, (height + blocksize.y - 1) / blocksize.y);
	MatrixTranspose << <gridsize, blocksize >> > (d_input, d_output, width, height);
	
	cudaDeviceSynchronize();
	cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "Input Matrix" << std::endl;
	for (int x = 0; x < width * height; x++)
	{
		std::cout << h_input[x] << ", ";
	}
	std::cout << std::endl;

	std::cout << "Output Matrix" << std::endl;
	for (int x = 0; x < width * height; x++)
	{
		std::cout << h_output[x] << ", ";
	}

	cudaFree(d_input);
	cudaFree(d_output);
	return 0;
}