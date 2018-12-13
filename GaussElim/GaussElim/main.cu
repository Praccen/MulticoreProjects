#include <iostream>

//Data variables
double **matrix;
double *vectorB;
double *vectorY;

//Options
int matrixSize;
const char* init;
int maxNum;

void initMatrix() {
	matrix = new double*[matrixSize];
	vectorB = new double[matrixSize];
	vectorY = new double[matrixSize];

	for (int i = 0; i < matrixSize; i++) {
		matrix[i] = new double[matrixSize];

		//Init vectors
		vectorB[i] = 2.0;
		vectorY[i] = 1.0;
	}

	if (strcmp(init, "rand") == 0) {
		for (int i = 0; i < matrixSize; i++) {
			for (int j = 0; j < matrixSize; j++) {
				if (i == j) { //diagonal dominance
					matrix[i][j] = (double)(rand() % maxNum) + 5.0;
				}
				else {
					matrix[i][j] = (double)(rand() % maxNum) + 1.0;
				}
			}
		}
	}

	if (strcmp(init, "fast") == 0) {
		for (int i = 0; i < matrixSize; i++) {
			for (int j = 0; j < matrixSize; j++) {
				if (i == j) { //diagonal dominance
					matrix[i][j] = 5.0;
				}
				else {
					matrix[i][j] = 2.0;
				}
			}
		}
	}
}

void gaussSeq() {
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

void printMatrix() {
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

int main() {
	matrixSize = 5;
	maxNum = 15;
	init = "rand";
	//init = "fast";

	initMatrix();

	printMatrix();

	gaussSeq();

	printMatrix();

	getchar();
	return 0;
}