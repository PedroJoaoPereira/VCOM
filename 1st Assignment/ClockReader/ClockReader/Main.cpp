#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main() {
	Mat image1, image2;

	//read an image
	image1 = imread("./Resources/img.jpg");
	if (!image1.data) {
		cout << "Image reading error !\n";

		waitKey(0);
		return 1;
	}

	//show the image on window
	imshow("Original image", image1);

	//show original image size
	cout << "Original image:\n";
	cout << "height = " << image1.size().height << endl;
	cout << "width = " << image1.size().width << endl;

	//horizontal flip image1 and save result in image2
	flip(image1, image2, 1);

	//show the image on window
	imshow("Flipped image", image2);

	//save flipped image in 'bmp' format
	imwrite("img_flip.bmp", image2);

	waitKey(0);
	return 0;
}