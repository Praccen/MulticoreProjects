#include <iostream>
#include <time.h>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

//Data variables
double **g_matrix;
double *g_vectorB;
double *g_vectorY;

//Options
int g_matrixSize;
const char* g_init;
int g_maxNum;

void initMatrix() {
	g_matrix = new double*[g_matrixSize];
	g_vectorB = new double[g_matrixSize];
	g_vectorY = new double[g_matrixSize];

	for (int i = 0; i < g_matrixSize; i++) {
		g_matrix[i] = new double[g_matrixSize];

		//Init vectors
		g_vectorB[i] = 2.0;
		g_vectorY[i] = 1.0;
	}

	if (strcmp(g_init, "rand") == 0) {
		for (int i = 0; i < g_matrixSize; i++) {
			for (int j = 0; j < g_matrixSize; j++) {
				if (i == j) { //diagonal dominance
					g_matrix[i][j] = (double)(rand() % g_maxNum) + 5.0;
				}
				else {
					g_matrix[i][j] = (double)(rand() % g_maxNum) + 1.0;
				}
			}
		}
	}

	if (strcmp(g_init, "fast") == 0) {
		for (int i = 0; i < g_matrixSize; i++) {
			for (int j = 0; j < g_matrixSize; j++) {
				if (i == j) { //diagonal dominance
					g_matrix[i][j] = 5.0;
				}
				else {
					g_matrix[i][j] = 2.0;
				}
			}
		}
	}
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
void gaussPar(double **matrix, int matrixSize, double *vectorB, double *vectorY) {

}

void printMatrix() {
	printf("Matrix A: \n");
	for (int i = 0; i < g_matrixSize; i++) {
		printf("[");
		for (int j = 0; j < g_matrixSize; j++) {
			printf(" %5.2f,", g_matrix[i][j]);
		}
		printf("]\n");
	}

	printf("Vector b: \n[");
	for (int i = 0; i < g_matrixSize; i++) {
		printf(" %5.2f,", g_vectorB[i]);
	}
	printf("]\n");

	printf("Vector y: \n[");
	for (int i = 0; i < g_matrixSize; i++) {
		printf(" %5.2f,", g_vectorY[i]);
	}
	printf("]\n\n");
}

int main() {
	g_matrixSize = 5;
	g_maxNum = 15;
	g_init = "rand";
	//init = "fast";

	initMatrix();

	printMatrix();

	gaussSeq(g_matrix, g_matrixSize, g_vectorB, g_vectorY);

	printMatrix();

	getchar();
	return 0;
}