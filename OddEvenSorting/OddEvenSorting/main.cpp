#include <iostream>
#include <vector>
#include <time.h>
#include <chrono>

using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

void seqOddEvenSort(std::vector<int> *numSeq, int randomArrayLength) {
	time_point<Clock> m_start, m_end;
	int sortTime;

	m_start = Clock::now();
	// One iteration of odd-even sort
	bool unsorted = true;
	int temp = 0;
	int counter = 0;
	while (unsorted) {
		unsorted = false;
		for (int k = 0; k < 2; k++) {
			for (int i = k, j = k + 1; j < randomArrayLength; i += 2, j += 2) {
				if (numSeq->at(j) < numSeq->at(i)) {
					temp = numSeq->at(i);
					numSeq->at(i) = numSeq->at(j);
					numSeq->at(j) = temp;
					unsorted = true;
				}
			}
		}
		counter++;
	}
	m_end = Clock::now();

	sortTime = (int)duration_cast<milliseconds>(m_end - m_start).count();
	std::cout << "Sorted in " << counter << " iterations! It took " << sortTime << " milliseconds.\n";
}

void printArray(std::vector<int> *numSeq, int randomArrayLength) {
	// Print out sequence after one iteration of odd-even sort
	std::cout << "Printing array:\n";
	for (int i = 0; i < randomArrayLength; i++) {
		std::cout << numSeq->at(i) << " ";
	}
}

int main() {
	int randomArrayLength = 1000;
	std::vector<int> numSeq;

	srand(time(NULL));


	// Initiate number sequence with random numbers
	for (int i = 0; i < randomArrayLength; i++) {
		numSeq.push_back(rand() % 100); // between 0 and 9
	}

	//Sort sequentially
	seqOddEvenSort(&numSeq, randomArrayLength);
	//printArray(&numSeq, randomArrayLength);

	//Sort in parallel


	getchar();
	return 0;
}