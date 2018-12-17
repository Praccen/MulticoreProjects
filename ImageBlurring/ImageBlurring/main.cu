#include <iostream>
#include <time.h>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "CImg.h"

using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

using namespace cimg_library;

void blurSeq() {

	// Don't use the small cake 
	// CImg<unsigned char> image("cake-small.ppm"), blurimage("cake-small.ppm");

	// Use the big cake
	CImg<unsigned char> image("cake.ppm"), blurimage("cake.ppm");

	// Don't use the small mask
	/* Create a mask of weights (3 x 3 Gaussian blur)
	double a3 = 1.0 / 16.0;
	double b3 = 2.0 / 16.0;
	double c3 = 4.0 / 16.0;
	CImg<> mask3 = CImg<>(3, 3).fill(
		a3, b3, a3,
		b3, c3, b3,
		a3, b3, a3);
	 */

	 // Use the big mask
	 // Create the mask of weights (5 x 5 Gaussian blur)
	CImg<double> mask5(5, 5);
	mask5(0, 0) = mask5(0, 4) = mask5(4, 0) = mask5(4, 4) = 1.0 / 256.0;
	mask5(0, 1) = mask5(0, 3) = mask5(1, 0) = mask5(1, 4) = mask5(3, 0) = mask5(3, 4) = mask5(4, 1) = mask5(4, 3) = 4.0 / 256.0;
	mask5(0, 2) = mask5(2, 0) = mask5(2, 4) = mask5(4, 2) = 6.0 / 256.0;
	mask5(1, 1) = mask5(1, 3) = mask5(3, 1) = mask5(3, 3) = 16.0 / 256.0;
	mask5(1, 2) = mask5(2, 1) = mask5(2, 3) = mask5(3, 2) = 24.0 / 256.0;
	mask5(2, 2) = 36.0 / 256.0;

	// Print the mask that is being used. Note: Doesn't look so pretty.
	for (int i = 0; i <= 4; i++)
	{
		for (int j = 0; j <= 4; j++)
			std::cout << mask5(i, j) << " ";
		std::cout << "\n";
	}

	// Convolve and record the time taken to do the operation
	auto begin = std::chrono::high_resolution_clock::now();
	// Blur the image!
	blurimage.convolve(mask5);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - begin;
	std::cout << "Time taken to convolve = " << elapsed.count() << " seconds";

	// Show the original and the blurred images and compare.

	// To display the images as 400 x 300
	/*
	CImgDisplay main_disp(400, 300, "Original image");
	CImgDisplay main_disp2(400, 300, "Blurred image");
	main_disp.render(image);
	main_disp2.render(blurimage);
	*/

	// Display the images in their original size
	CImgDisplay main_disp(image, "Original image");
	CImgDisplay main_disp2(blurimage, "Blurred image");

	while (1)
	{
		main_disp.wait(); main_disp2.wait();
	}
}

int main() {

	std::cout << "Hello world!\n\n";

	blurSeq();

	getchar();
	return 0; 
}