#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

//used to determine width od the picture after resizing, height will be resized proportionaly
const int NEW_WIDTH = 400;

//resizes the image to width defined by second argument, height set proportionaly
Mat resizeImage(Mat original, int newWidth) {
	Mat resizedImage;
	Size newSize;

	newSize.width = newWidth;
	newSize.height = newWidth*1.0 / original.size().width*original.size().height;	
	cv::resize(original, resizedImage, newSize);
	return resizedImage;
}

//returns an image that was taken wth camera or loaded from path specified in console
Mat getImage() {
	char mode;
	Mat src;
	while (true) {
		cout << "Please select mode : (C for camera input, P for path):";
		cin >> mode;
		cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');


		if (tolower(mode) == 'c') {
			cout << "Press ESC key to take picture";
			VideoCapture cap;
			// open the default camera, use something different from 0 otherwise;
			// Check VideoCapture documentation.
			if (!cap.open(0))
				break;
			for (;;)
			{
				
				cap >> src;
				if (src.empty()) break; // end of video stream
				imshow("Take a picture with ESC key", src);
				if (waitKey(10) == 27) break; // stop capturing by pressing ESC 
			}
			// the camera will be closed automatically upon exit
			// cap.close();
			return resizeImage(src, NEW_WIDTH);
			//break;
		}


		else if (tolower(mode) == 'p') {
			string imPath;

			while (true) {

				cout << "enter a path:";
				getline(cin, imPath);

				src = imread(imPath);
				// Verify reading success
				if (src.data) {
					return resizeImage(src, NEW_WIDTH);
				}
			}
		}


		else {
			cout << "You entered wrong character";
			cout << "\n";
		}
	}
}

int main() {

	// Reads the image
	Mat src = getImage();

	// Verify reading success
	if (!src.data) {
		cout << "Image reading error!" << endl;
		return 1;
	}
	imshow("Original (resized)", src);
	waitKey(0);
	return 0;
}
