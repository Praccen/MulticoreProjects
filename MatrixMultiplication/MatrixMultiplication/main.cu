#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

const int Size = 2;

__global__
void matrixMult(float *N, float *M, float *P, int n) { //Client
	int threadIndex = threadIdx.x;
	int blockIndex = blockIdx.x;
	int blockSize = blockDim.x;
	int stride = gridDim.x * blockSize;

	for (int i = threadIndex + blockIndex * blockSize; i < n * n; i += stride) {
		P[i] = 0;
		//Calculate x and y based on i (index to 2d coords conversion/translation)
		int x = i % Size;
		int y = (int) floor((double)i / Size);

		//Calculate output value
		for (int j = 0; j < n; j++) {
			P[i] += M[x + j * Size] * N[j + y * Size];
		}
	}
}

void GetMatrix(float matrix[Size][Size], std::string file) {
	for (int i = 0; i < Size; i++) {
		for (int j = 0; j < Size; j++) {
			matrix[j][i] = j + Size * i;
		}
	}
}

void PutMatrix(float matrix[Size][Size], std::string file) {
	std::cout << "Printing matrix: \n";

	for (int i = 0; i < Size; i++) {
		for (int j = 0; j < Size; j++) {
			std::cout << matrix[j][i] << ", ";
		}
		std::cout << "\n";
	}
}

void main() { //Host
	float N[Size][Size], M[Size][Size], P[Size][Size]; // Matrices
	float N_h[Size * Size], M_h[Size * Size], P_h[Size * Size]; // Matrices written as 1d array
	float *N_d, *M_d, *P_d; // Device variables
	int allocSize = Size * Size * sizeof(float);

	GetMatrix(N, ""); GetMatrix(M, ""); /* Read N and M */

	//Convert N and M to N_h and M_h
	for (int i = 0; i < Size; i++) {
		for (int j = 0; j < Size; j++) {
			N_h[j + i * Size] = N[j][i];
			M_h[j + i * Size] = M[j][i];
		}
	}

	//Allocate memory on device
	cudaMalloc(&N_d, allocSize);
	cudaMalloc(&M_d, allocSize);
	cudaMalloc(&P_d, allocSize);

	//Copy N_h and M_h to device (N_d, M_d)
	cudaMemcpy(N_d, N_h, allocSize, cudaMemcpyHostToDevice);
	cudaMemcpy(M_d, M_h, allocSize, cudaMemcpyHostToDevice);

	//Set number of blocks and number of threads per block
	int numberOfBlocks = 4;
	int numberOfThreads = 32;

	//Call device function
	matrixMult << <numberOfBlocks, numberOfThreads >> > (N_d, M_d, P_d, Size);

	//Copy device variable P_d to host variable P_h
	cudaMemcpy(P_h, P_d, allocSize, cudaMemcpyDeviceToHost);

	//Convert P_h to P
	for (int i = 0; i < Size; i++) {
		for (int j = 0; j < Size; j++) {
			P[j][i] = P_h[j + i * Size];
		}
	}

	PutMatrix(P, "");  /* Skriv ut P */

	//Deallocate device memory
	cudaFree(N_d);
	cudaFree(M_d);
	cudaFree(P_d);
}
