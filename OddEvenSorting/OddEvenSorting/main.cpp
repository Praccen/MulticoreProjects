#include <iostream>
#include <vector>
#include <time.h>

int main() {
	int randomArrayLength = 1000;
	std::vector<int> numSeq;

	srand(time(NULL));


	// Initiate number sequence with random numbers
	for (int i = 0; i < randomArrayLength; i++) {
		numSeq.push_back(rand() % 100); // between 0 and 9
	}


	// One iteration of odd-even sort
	bool unsorted = true;
	int temp = 0;
	int counter = 0;
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
		counter++;
	}


	// Print out sequence after one iteration of odd-even sort
	for (int i = 0; i < randomArrayLength; i++) {
		std::cout << numSeq[i] << " ";
	}

	std::cout << "\nNumber of iterations: " << counter << "\n";

	getchar();
	return 0;
}