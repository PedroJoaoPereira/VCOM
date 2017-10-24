#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

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

	// duje

	// IMAGE READING -------------------------------
	// Reads the image
	Mat src = getImage();

	// Verify reading success
	if (!src.data) {
		cout << "Image reading error!" << endl;
		return 1;
	}

	// IMAGE PREPARATION ---------------------------
	// Noise removal with Gaussian blur
	Mat srcBlur;
	GaussianBlur(src, srcBlur, Size(3, 3), 0, 0, BORDER_DEFAULT);

	// Convert image to grayscale
	Mat srcGray;
	cvtColor(srcBlur, srcGray, CV_BGR2GRAY);

	// Apply Laplacian operator
	Mat srcLaplacian;
	Laplacian(srcGray, srcLaplacian, CV_8U, 3, 1, 0, BORDER_DEFAULT);

	// CLOCK SEGMENTATION --------------------------
	// Find all clock contours
	vector<vector<Point>> contours;
	findContours(srcLaplacian, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Find the area of larger contour
	double maxArea = 0;
	vector<Point> relevantContour;
	for (int i = 0; i< contours.size(); i++) {
		double temp = contourArea(contours[i]);

		if (temp > maxArea) {
			maxArea = temp;
			relevantContour = contours[i];
		}
	}

	// Select inner larger contour
	double minArea = maxArea;
	for (int i = 0; i< contours.size(); i++) {
		double temp = contourArea(contours[i]);
		if (temp > maxArea * 0.4 && temp < minArea) {
			minArea = temp;
			relevantContour = contours[i];
		}
	}

	// Clock inner contour
	vector<vector<Point>> relevantContourVec = vector<vector<Point>>();
	relevantContourVec.push_back(relevantContour);

	// Create clock mask to segment the background
	Mat srcRawMask = Mat::zeros(src.size(), CV_8U);
	drawContours(srcRawMask, relevantContourVec, 0, Scalar(255, 255, 255), CV_FILLED);

	// Close noise in the raw mask
	Mat srcMask;
	morphologyEx(srcRawMask, srcMask, MORPH_CLOSE, Mat());

	// Segmenting background
	Mat srcClock;
	src.copyTo(srcClock, srcMask);

	// DEBUGGING -----------------------------------
	// Show the image
	imshow("Original", src);
	imshow("Blur", srcBlur);
	imshow("Gray", srcGray);
	imshow("Laplacian", srcLaplacian);
	imshow("Raw Mask", srcRawMask);
	imshow("Mask", srcMask);
	imshow("Clock", srcClock);

	// Wait for key press
	waitKey(0);
	return 0;
}
