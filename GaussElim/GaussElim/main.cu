#include <iostream>
#include <time.h>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

//Options
int g_matrixSize;
const char* g_init;
int g_maxNum;

double **initMatrixSeq() {
	double **matrix = new double*[g_matrixSize];

	for (int i = 0; i < g_matrixSize; i++) {
		matrix[i] = new double[g_matrixSize];
	}

	if (strcmp(g_init, "rand") == 0) {
		for (int i = 0; i < g_matrixSize; i++) {
			for (int j = 0; j < g_matrixSize; j++) {
				if (i == j) { //diagonal dominance
					matrix[i][j] = (double)(rand() % g_maxNum) + 5.0;
				}
				else {
					matrix[i][j] = (double)(rand() % g_maxNum) + 1.0;
				}
			}
		}
	}

	if (strcmp(g_init, "fast") == 0) {
		for (int i = 0; i < g_matrixSize; i++) {
			for (int j = 0; j < g_matrixSize; j++) {
				if (i == j) { //diagonal dominance
					matrix[i][j] = 5.0;
				}
				else {
					matrix[i][j] = 2.0;
				}
			}
		}
	}

	return matrix;
}

double **initMatrixPar() {
	double **matrix;// = new double*[g_matrixSize];
	cudaMallocManaged(&matrix, g_matrixSize * sizeof(double *));

	for (int i = 0; i < g_matrixSize; i++) {
		cudaMallocManaged(&matrix[i], g_matrixSize * sizeof(double));
	}

	if (strcmp(g_init, "rand") == 0) {
		for (int i = 0; i < g_matrixSize; i++) {
			for (int j = 0; j < g_matrixSize; j++) {
				if (i == j) { //diagonal dominance
					matrix[i][j] = (double)(rand() % g_maxNum) + 5.0;
				}
				else {
					matrix[i][j] = (double)(rand() % g_maxNum) + 1.0;
				}
			}
		}
	}

	if (strcmp(g_init, "fast") == 0) {
		for (int i = 0; i < g_matrixSize; i++) {
			for (int j = 0; j < g_matrixSize; j++) {
				if (i == j) { //diagonal dominance
					matrix[i][j] = 5.0;
				}
				else {
					matrix[i][j] = 2.0;
				}
			}
		}
	}

	return matrix;
}

double *initVectorBSeq() {
	double *vectorB = new double[g_matrixSize];

	for (int i = 0; i < g_matrixSize; i++) {
		vectorB[i] = 2.0;
	}

	return vectorB;
}

double *initVectorYSeq() {
	double* vectorY = new double[g_matrixSize];

	for (int i = 0; i < g_matrixSize; i++) {
		vectorY[i] = 1.0;
	}

	return vectorY;
}

double *initVectorBPar() {
	double *vectorB;
	cudaMallocManaged(&vectorB, g_matrixSize * sizeof(double));

	for (int i = 0; i < g_matrixSize; i++) {
		vectorB[i] = 2.0;
	}

	return vectorB;
}

double *initVectorYPar() {
	double* vectorY;
	cudaMallocManaged(&vectorY, g_matrixSize * sizeof(double));

	for (int i = 0; i < g_matrixSize; i++) {
		vectorY[i] = 1.0;
	}

	return vectorY;
}

void gaussSeq(double **matrix, int matrixSize, double *vectorB, double *vectorY) {
	/* Gaussian elimination algorithm, Algo 8.4 from Grama */
	for (int k = 0; k < matrixSize; k++) { /* Outer loop */
		for (int j = k + 1; j < matrixSize; j++) {
			matrix[k][j] = matrix[k][j] / matrix[k][k]; /* Division step */
		}
		vectorY[k] = vectorB[k] / matrix[k][k];
		matrix[k][k] = 1.0;

		for (int i = k + 1; i < matrixSize; i++) {
			for (int j = k + 1; j < matrixSize; j++) {
				matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j]; /* Elimination step */
			}
			vectorB[i] = vectorB[i] - matrix[i][k] * vectorY[k];
			matrix[i][k] = 0.0;
		}
	}
}

__global__
void gaussPar1(double **matrix, int matrixSize, double *vectorB, double *vectorY, int k) {
	int threadIndex = threadIdx.x;
	int blockIndex = blockIdx.x;
	int blockSize = blockDim.x;
	int stride = gridDim.x * blockSize;

	for (int j = k + 1 + threadIndex + blockIndex * blockSize; j < matrixSize; j += stride) {
		matrix[k][j] = matrix[k][j] / matrix[k][k]; /* Division step */
		for (int i = k + 1; i < matrixSize; i++) {
			matrix[i][j] = matrix[i][j] - matrix[i][k] * matrix[k][j]; /* Elimination step */
		}
	}
}

void print(double **matrix, int matrixSize, double *vectorB, double *vectorY) {
	printf("Matrix A: \n");
	for (int i = 0; i < matrixSize; i++) {
		printf("[");
		for (int j = 0; j < matrixSize; j++) {
			printf(" %5.2f,", matrix[i][j]);
		}
		printf("]\n");
	}

	printf("Vector b: \n[");
	for (int i = 0; i < matrixSize; i++) {
		printf(" %5.2f,", vectorB[i]);
	}
	printf("]\n");

	printf("Vector y: \n[");
	for (int i = 0; i < matrixSize; i++) {
		printf(" %5.2f,", vectorY[i]);
	}
	printf("]\n\n");
}

bool compareMatrices(double **matrix1, double **matrix2, int matrixSize) {
	//Check if matrix1 has the same values as matrix2
	bool isSame = true;

	for (int i = 0; i < matrixSize; i++) {
		for (int j = 0; j < matrixSize; j++) {
			if (abs(matrix1[i][j] - matrix2[i][j]) > 0.000001) {
				isSame = false;
				i = matrixSize;
				j = matrixSize;
			}
		}
	}

	return isSame;
}

bool compareVectors(double *vector1, double *vector2, int vectorSize) {
	//Check if vector1 has the same values as vector2
	bool isSame = true;

	for (int i = 0; i < vectorSize; i++) {
		if (abs(vector1[i] - vector2[i]) > 0.000001) {
			isSame = false;
			i = vectorSize;
		}
	}

	return isSame;
}

int main() {
	time_point<Clock> m_start, m_end;

	g_matrixSize = 2048;
	g_maxNum = 15;
	//g_init = "rand";
	g_init = "fast";

	//Allocate
	double **seqMatrix = initMatrixSeq();
	double *seqVectorB = initVectorBSeq();
	double *seqVectorY = initVectorYSeq();

	if (g_matrixSize < 15) {
		print(seqMatrix, g_matrixSize, seqVectorB, seqVectorY);
	}

	m_start = Clock::now();
	gaussSeq(seqMatrix, g_matrixSize, seqVectorB, seqVectorY);
	m_end = Clock::now();
	int seqGaussTime = (int)duration_cast<milliseconds>(m_end - m_start).count();
	std::cout << "Sequential gauss elimination took " << seqGaussTime << " milliseconds.\n";

	if (g_matrixSize < 15) {
		print(seqMatrix, g_matrixSize, seqVectorB, seqVectorY);
	}

	//Allocate
	double **parMatrix = initMatrixPar();
	double *parVectorB = initVectorBPar();
	double *parVectorY = initVectorYPar();

	int numberOfBlocks = 32;
	int numberOfThreadsPerBlock = 512;

	m_start = Clock::now();
	for (int k = 0; k < g_matrixSize; k++) {
		gaussPar1 << <numberOfBlocks, numberOfThreadsPerBlock >> > (parMatrix, g_matrixSize, parVectorB, parVectorY, k);
	}
	cudaDeviceSynchronize();

	for (int k = 0; k < g_matrixSize; k++) {
		parVectorY[k] = parVectorB[k] / parMatrix[k][k];
		parMatrix[k][k] = 1.0;

		for (int i = k + 1; i < g_matrixSize; i++) {
			parVectorB[i] = parVectorB[i] - parMatrix[i][k] * parVectorY[k];
			parMatrix[i][k] = 0.0;
		}
	}
	m_end = Clock::now();

	int parGaussTime = (int)duration_cast<milliseconds>(m_end - m_start).count();
	std::cout << "Parallell gauss elimination took " << parGaussTime << " milliseconds.\n";

	if (g_matrixSize < 15) {
		print(parMatrix, g_matrixSize, parVectorB, parVectorY);
	}

	std::cout << "Check if the matrices are the same: ";
	if (compareMatrices(seqMatrix, parMatrix, g_matrixSize)) {
		std::cout << "Yes\n";
	}
	else {
		std::cout << "No \n";
	}

	std::cout << "Check if the B-vectors are the same: ";
	if (compareVectors(seqVectorB, parVectorB, g_matrixSize)) {
		std::cout << "Yes\n";
	}
	else {
		std::cout << "No \n";
	}

	std::cout << "Check if the Y-vectors are the same: ";
	if (compareVectors(seqVectorY, parVectorY, g_matrixSize)) {
		std::cout << "Yes\n";
	}
	else {
		std::cout << "No \n";
	}

	//Deallocate seq memory
	for (int i = 0; i < g_matrixSize; i++) {
		delete[] seqMatrix[i];
	}
	delete[] seqMatrix;
	delete[] seqVectorB;
	delete[] seqVectorY;

	//Deallocate unified (par) memory
	for (int i = 0; i < g_matrixSize; i++) {
		cudaFree(parMatrix[i]);
	}
	cudaFree(parMatrix);
	cudaFree(parVectorB);
	cudaFree(parVectorY);

	getchar();
	return 0;
}