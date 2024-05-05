#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

double distanceCalculate(Point p1, Point p2) {
	double x1 = p1.x;
	double y1 = p1.y;
	double x2 = p2.x;
	double y2 = p2.y;

	double x = x1 - x2; // calculating number to square in next step
	double y = y1 - y2;
	double dist;

	dist = pow(x, 2) + pow(y, 2); // calculating Euclidean distance
	dist = sqrt(dist);

	return dist;
}

int main() {
	// FOR DEBUGGING
	vector<Mat> images = vector<Mat>();
	images.push_back(imread("./Resources/img9.jpg"));
	images.push_back(imread("./Resources/img4.jpg"));
	images.push_back(imread("./Resources/img8.jpg"));
	images.push_back(imread("./Resources/img1.jpg"));
	images.push_back(imread("./Resources/img7.jpg"));
	images.push_back(imread("./Resources/img2.jpg"));
	images.push_back(imread("./Resources/img3.jpg"));
	images.push_back(imread("./Resources/img5.jpg"));
	images.push_back(imread("./Resources/img6.jpg"));
	images.push_back(imread("./Resources/img10.jpg"));

	for (int index = 0; index < images.size(); index++) {
		// IMAGE READING -------------------------------
		// Reads the image
		Mat src = images.at(index);

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
		pyrMeanShiftFiltering(srcBlur, srcMeanFilter, 10, 150, 1);

		// Converts image to gray scale
		Mat srcGray;
		cvtColor(srcMeanFilter, srcGray, CV_BGR2GRAY);

		// Threshold the image to binary
		Mat srcThreshold;
		threshold(srcGray, srcThreshold, 170, 255, THRESH_BINARY);

		// CLOCK SEGMENTATION --------------------------
		// Find all contours from edges
		vector<vector<Point>> allContours;
		findContours(srcThreshold, allContours, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));

		// Comparing operator of the priority queue
		class CompareArea {
		public:
			bool operator()(pair<double, pair<vector<Point>, double>> pair1, pair<double, pair<vector<Point>, double>> pair2) {
				return pair1.first < pair2.first;
			}
		};

		// Priority queue mapping contours, its area and its distance to the center of the image
		priority_queue<pair<double, pair<vector<Point>, double>>,
			vector<pair<double, pair<vector<Point>, double>>>,
			CompareArea> pq = priority_queue<pair<double, pair<vector<Point>, double>>,
				vector<pair<double, pair<vector<Point>, double>>>,
				CompareArea>();

		// Creates variables for image information
		Point centerOfImage = Point(src.size().width / 2, src.size().height / 2);
		double areaOfImage = src.size().width * src.size().height;

		// Maps the distance of the center of mass of the contour to the center of the image
		for (int i = 0; i < allContours.size(); i++) {
			Rect bounding = boundingRect(allContours[i]);
			Point centerOfMass = Point(bounding.x + bounding.width / 2, bounding.y + bounding.height / 2);

			double contourAreaValue = contourArea(allContours[i]);
			double distanceToCenter = distanceCalculate(centerOfImage, centerOfMass);
			pq.push(pair<double, pair<vector<Point>, double>>(contourAreaValue, pair<vector<Point>, double>(allContours[i], distanceToCenter)));
		}

		// Finds the best suiting contour
		double contourThreshold = .25;
		vector<Point> bestContour = pq.top().second.first;
		double pivotDistance = distanceCalculate(centerOfImage,
			Point(
				(src.size().width / 2) * contourThreshold + src.size().width / 2,
				(src.size().height / 2) * contourThreshold + src.size().height / 2));
		do {
			pair<double, pair<vector<Point>, double>> pair = pq.top();

			if (contourArea(pair.second.first) < areaOfImage * .95 && pair.second.second < pivotDistance) {
				bestContour = pair.second.first;
				break;
			}

			pq.pop();
		} while (!pq.empty());

		// Selects the best contour that probably is the clock face
		vector<vector<Point>> bestContours = vector<vector<Point>>();
		bestContours.push_back(bestContour);

		// Create clock mask to segment the background
		Mat srcRawMask = Mat::zeros(src.size(), CV_8U);
		drawContours(srcRawMask, bestContours, 0, Scalar(255, 255, 255), CV_FILLED);

		// Clears noise from the raw mask
		Mat srcMask;
		morphologyEx(srcRawMask, srcMask, MORPH_OPEN, Mat(), Point(-1, -1), 5);

		// Segmenting background
		Mat srcClock;
		src.copyTo(srcClock, srcMask);

		// DEBUGGING -----------------------------------
		// IMAGE PREPARATION
		imshow("Original", src); // original image
		//imshow("Contrast", srcContrast); // contrasted image
		//imshow("Blur", srcBlur); // noise blur from the image
		//imshow("Mean Filter", srcMeanFilter); // cluster image colors
		//imshow("Gray", srcGray); // grayscales image
		//imshow("Threshold", srcThreshold); // binarize image

		// CLOCK SEGMENTATION
		//imshow("Raw Mask", srcRawMask); // noisy mask of the face of the clock
		//imshow("Mask", srcMask); // noise clear from the mask
		imshow("Clock", srcClock); // segments clock face from the background

		waitKey(0);
	}

	// Wait for key press
	waitKey(0);
	return 0;
}