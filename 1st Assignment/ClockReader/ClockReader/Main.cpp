#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/*
// Using this to debug

for (int i = 300; i < 450; i = i + 10) {
	Mat temp;
	Canny(srcBlur, temp, 50, i, 3);
	string str = "";
	str += to_string(i);
	imshow(str, temp);
}
*/

int main() {

	// IMAGE READING -------------------------------
	// Reads the image
	Mat src = imread("./Resources/img1.jpg");

	// Verify reading success
	if (!src.data) {
		cout << "Image reading error!" << endl;

		// Wait for key press
		waitKey(0);
		return 1;
	}

	// IMAGE PREPARATION ---------------------------
	// Increases image contrast and brightness for clearer edge detection
	Mat srcContrast = Mat::zeros(src.size(), src.type());
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			for (int c = 0; c < 3; c++) {
				srcContrast.at<Vec3b>(y, x)[c] =
					saturate_cast<uchar>(2.2*(src.at<Vec3b>(y, x)[c]) - 100);
			}
		}
	}

	// Noise removal with Gaussian blur
	Mat srcBlur;
	GaussianBlur(srcContrast, srcBlur, Size(3, 3), 2, 2);

	// Mean filters the color
	Mat srcMeanFilter;
	pyrMeanShiftFiltering(srcBlur, srcMeanFilter, 10, 40, 5);

	// Converts image to gray scale
	Mat srcGray;
	cvtColor(srcMeanFilter, srcGray, CV_BGR2GRAY);

	// Transforms image to binary
	Mat srcThreshold;
	threshold(srcGray, srcThreshold, 150, 255, THRESH_BINARY);

	//morphologyEx(srcThreshold, srcThreshold, MORPH_CLOSE, Mat(), Point(-1, -1), 4);

	// Apply Canny edge detection
	Mat srcCanny;
	//Canny(srcThreshold, srcCanny, 100, 100, 3);
	Laplacian(srcThreshold, srcCanny, CV_8UC1, 3, 1, 0, BORDER_CONSTANT);

	// CLOCK SEGMENTATION --------------------------
	// Find all clock contours
	vector<vector<Point>> contours;
	findContours(srcCanny, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));

	RNG rng(12345);
	vector<Vec4i> hierarchy;
	Mat drawing = Mat::zeros(srcCanny.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

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




	vector<vector<Point>> relevantContourVec = vector<vector<Point>>();
	relevantContourVec.push_back(relevantContour);

	

	//vector<vector<Point> > contours;
	
	findContours(srcCanny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	








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
	//vector<vector<Point>> relevantContourVec = vector<vector<Point>>();
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
	//imshow("Contrast", srcContrast);
	//imshow("Blur", srcBlur);
	//imshow("Mean Filter", srcMeanFilter);
	//imshow("Gray", srcGray);
	imshow("Threshold", srcThreshold);
	imshow("Canny", srcCanny);

	//imshow("Raw Mask", srcRawMask);
	//imshow("Mask", srcMask);
	//imshow("Clock", srcClock);

	imshow("Contours", drawing);

	// Wait for key press
	waitKey(0);
	return 0;
}