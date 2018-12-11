#include <iostream>
#include <vector>
#include <time.h>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

void seqOddEvenSort(int *numSeq, int randomArrayLength) {
	bool unsorted = true;
	int temp = 0;
	while (unsorted) {
		unsorted = false;
		for (int k = 0; k < 2; k++) {
			for (int i = k, j = k + 1; j < randomArrayLength; i += 2, j += 2) {
				if (numSeq[j] < numSeq[i]) {
					temp = numSeq[i];
					numSeq[i] = numSeq[j];
					numSeq[j] = temp;
					unsorted = true;
				}
			}
		}
	}
}


__global__
void parSort(int *numSeq, int randomArrayLength, bool *sortedArr) {
	int threadIndex = threadIdx.x;
	int blockIndex = blockIdx.x;
	int blockSize = blockDim.x;
	int stride = gridDim.x * blockSize;

	sortedArr[blockIndex] = true;
	for (int k = 0; k < 2; k++) {
		for (int i = (blockIndex * blockSize + threadIndex) * 2 + k, j = (blockIndex * blockSize + threadIndex) * 2 + k + 1; j < randomArrayLength; i += stride * 2, j += stride * 2) {
			if (numSeq[j] < numSeq[i]) {
				int temp = numSeq[i];
				numSeq[i] = numSeq[j];
				numSeq[j] = temp;
				sortedArr[blockIndex] = false;
			}
		}
		__syncthreads();
	}
}


void printArray(int *numSeq, int randomArrayLength) {
	// Print out sequence after one iteration of odd-even sort
	std::cout << "Printing array:\n";
	for (int i = 0; i < randomArrayLength; i++) {
		std::cout << numSeq[i] << " ";
	}
	std::cout << "\n";
}

int main() {
	//srand(time(NULL)); //No srand to assure the same array each time. No random variables in the testing.

	time_point<Clock> m_start, m_end;

	int randomArrayLength = 100000;
	int *numbers;
	numbers = new int[randomArrayLength];

	// Initiate number sequence with random numbers
	for (int i = 0; i < randomArrayLength; i++) {
		numbers[i] = rand() % 100; // between 0 and 9
	}

	//printArray(numbers, randomArrayLength);

	//----Sequential sort----
	int *numSeq;
	numSeq = new int[randomArrayLength];

	//Copy the values from the random array to the sequential array.
	for (int i = 0; i < randomArrayLength; i++) {
		numSeq[i] = numbers[i];
	}

	//Sort sequentially
	//Time it
	m_start = Clock::now();
	seqOddEvenSort(numSeq, randomArrayLength);
	m_end = Clock::now();
	int seqSortTime = (int)duration_cast<milliseconds>(m_end - m_start).count();
	std::cout << "Sequential sort took " << seqSortTime << " milliseconds.\n";

	//printArray(numSeq, randomArrayLength);

	delete[] numSeq;
	//-----------------------


	//----Sort in parallel----
	int numberOfBlocks = 32;
	int numberOfThreadsPerBlock = 256;

	int *parNumSeq;
	bool *sortedArr;

	//Allocate memory on the GPU.
	cudaMallocManaged(&parNumSeq, randomArrayLength * sizeof(int));
	cudaMallocManaged(&sortedArr, numberOfBlocks * sizeof(bool));

	//Copy the random numbers to the parrallell array.
	for (int i = 0; i < randomArrayLength; i++) {
		parNumSeq[i] = numbers[i];
	}

	//Initialize the sorted array. Used for stop condition
	for (int i = 0; i < numberOfBlocks; i++) {
		sortedArr[i] = false;
	}

	bool sorted = false;
	int syncCounter = 0;

	//Sort in parallell
	//Time it
	m_start = Clock::now();
	while (!sorted) {
		syncCounter++;
		parSort << <numberOfBlocks, numberOfThreadsPerBlock >> > (parNumSeq, randomArrayLength, sortedArr);
		if (syncCounter % 200 == 0) { //Check the stop condition every 200 executions.
			cudaDeviceSynchronize(); //Very costly, therefore we only do it once every 200 executions.

			//See if the array is completely sorted.
			sorted = true;
			for (int i = 0; i < numberOfBlocks; i++) {
				if (sortedArr[i] == false) {
					sorted = false; //Break the while loop.
					i = numberOfBlocks;
				}
			}
		}
	}
	m_end = Clock::now();

	int parSortTime = (int)duration_cast<milliseconds>(m_end - m_start).count();
	std::cout << "Parallell sort took " << parSortTime << " milliseconds.\n";

	//printArray(parNumSeq, randomArrayLength);

	//Free the GPU memory.
	cudaFree(parNumSeq);
	cudaFree(sortedArr);
	//------------------------

	getchar();
	return 0;
}