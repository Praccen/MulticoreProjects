#include <iostream>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


__global__
void matrixMult(float *N, float *M, float *P, int Size) { //Client
	int threadIndex = threadIdx.x;
	int blockIndex = blockIdx.x;
	int blockSize = blockDim.x;
	int stride = gridDim.x * blockSize;

	for (int i = threadIndex + blockIndex * blockSize; i < Size * Size; i += stride) {

	}
}

void main() { //Host
//float N[Size][Size], M[Size][Size], P[Size][Size]; //cudaMalloc?
	float *N_d, *M_d, *P_d;
	int allocSize = Size * Size * sizeof(float);

	GetMatrix(N, file1); GetMatrix(M, file2); /* Read N and M */

	cudaMalloc((void **), &N_d, allocSize);
	cudaMemcpy(N_d, N, allocSize, cudaMemcpyHostToDevice);

	cudaMalloc((void **), &M_d, allocSize);
	cudaMemcpy(M_d, M, allocSize, cudaMemcpyHostToDevice);

	cudaMalloc((void **), &P_d, allocSize);

	int numberOfBlocks = 4;
	int numberOfThreads = 32;
	matrixMult << <numberOfBlocks, numberOfThreads >> > (N_d, M_d, P_d, Size);

	cudaMemcpy(P, P_d, allocSize, cudaMemcpyDeviceToHost);

	PutMatrix(P, file3);  /* Skriv ut P */

	cudaFree(N_d);
	cudaFree(M_d);
	cudaFree(P_d);
}
