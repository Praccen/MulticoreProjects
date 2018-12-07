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
	time_point<Clock> m_start, m_end;
	int sortTime;

	m_start = Clock::now();
	// One iteration of odd-even sort
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
	m_end = Clock::now();

	sortTime = (int)duration_cast<milliseconds>(m_end - m_start).count();
	std::cout << "Sequential sort took " << sortTime << " milliseconds.\n";
}

__global__
void parSort(int *numSeq, int randomArrayLength) {
	int index = threadIdx.x;
	int stride = blockDim.x;

	__shared__ bool unsorted;
	unsorted = true;
	while (unsorted) {
		__syncthreads();
		unsorted = false;
		for (int k = 0; k < 2; k++) {
			for (int i = index * 2 + k, j = index * 2 + k + 1; j < randomArrayLength; i += stride * 2, j += stride * 2) {
				if (numSeq[j] < numSeq[i]) {
					int temp = numSeq[i];
					numSeq[i] = numSeq[j];
					numSeq[j] = temp;
					unsorted = true;
				}
			}
			__syncthreads();
		}
	}
}


void printArray(int *numSeq, int randomArrayLength) {
	// Print out sequence after one iteration of odd-even sort
	std::cout << "Printing array:\n";
	for (int i = 0; i < randomArrayLength; i++) {
		std::cout << numSeq[i] << " ";
	}
}

int main() {
	srand(time(NULL));

	int randomArrayLength = 1000;

	//----Sequential sort----
	int *numSeq;
	numSeq = new int[randomArrayLength];

	// Initiate number sequence with random numbers
	for (int i = 0; i < randomArrayLength; i++) {
		numSeq[i] = rand() % 100; // between 0 and 9
	}

	//Sort sequentially
	//seqOddEvenSort(numSeq, randomArrayLength);
	//printArray(numSeq, randomArrayLength);

	delete[] numSeq;
	//-----------------------

	//----Sort in parallel----
	int *parNumSeq;
	cudaMallocManaged(&parNumSeq, randomArrayLength * sizeof(int));

	// Initiate number sequence with random numbers
	for (int i = 0; i < randomArrayLength; i++) {
		parNumSeq[i] = rand() % 100; // between 0 and 9
	}

	time_point<Clock> m_start, m_end;
	int sortTime;

	m_start = Clock::now();
	parSort <<<1, 1024>>> (parNumSeq, randomArrayLength);

	cudaDeviceSynchronize();
	m_end = Clock::now();

	sortTime = (int)duration_cast<milliseconds>(m_end - m_start).count();
	std::cout << "Parallell sort took " << sortTime << " milliseconds.\n";

	//printArray(parNumSeq, randomArrayLength);

	cudaFree(numSeq);
	//------------------------


	getchar();
	return 0;
}