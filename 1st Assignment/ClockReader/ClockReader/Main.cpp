#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#define SIZE 400

/**
 * Calculates the distance between two points
 * @param p1 the first point
 * @param p2 the second point
 */
double distanceCalculate(Point p1, Point p2) {
	double x = p1.x - p2.x;
	double y = p1.y - p2.y;
	return sqrt(pow(x, 2) + pow(y, 2));
}

/**
 * Resizes the image
 * @param original the image to be resized
 */
Mat resizeImage(Mat original) {
	Mat resizedImage;
	resize(original, resizedImage, Size(SIZE, SIZE));
	return resizedImage;
}

/**
 * Returns an image that was shot with the webcam or loaded from a path specified in the console
 */
Mat getImage() {
	char mode;
	Mat src;

	while (true) {
		cout << "Please select mode: (C for camera input, P for path, Q for quit): ";
		cin >> mode;
		cin.ignore(numeric_limits<streamsize>::max(), '\n');

		// Camera mode
		if (tolower(mode) == 'c') {
			cout << "Press ESC key to take picture..." << endl;

			VideoCapture cap;
			// Open the default camera
			if (!cap.open(0))
				break;

			// Waits for image capture
			for (;;) {
				cap >> src;

				// End of video stream
				if (src.empty())
					break;

				imshow("Take a picture with ESC key", src);
				// Stop capturing by pressing ESC key
				if (waitKey(10) == 27)
					break;
			}

			// The camera will be closed automatically upon exit
			return resizeImage(src);

			// Path mode
		} else if (tolower(mode) == 'p') {
			string imPath;

			while (true) {
				cout << "Enter a path: ";
				getline(cin, imPath);

				src = imread(imPath);

				// Verify reading success
				if (src.data) {
					if (src.size().width > SIZE)
						return resizeImage(src);
					else
						return src;
				}
			}
		} else if (tolower(mode) == 'q') {
			exit(0);
		} else {
			cout << "You entered an invalid character! Try again." << endl;
		}
	}

	return src;
}

/**
 * Function to process the image
 * @param src the image to be processed
 */
void imageProcessing(Mat src) {
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

	// DEBUGGING -----------------------------------
	// IMAGE PREPARATION
	imshow("Original", src); // original image
	// imshow("Contrast", srcContrast); // contrasted image
	// imshow("Blur", srcBlur); // noise blur from the image
	// imshow("Mean Filter", srcMeanFilter); // cluster image colors
	// imshow("Gray", srcGray); // grayscales image
	// imshow("Threshold", srcThreshold); // binarize image

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
	// CLOCK SEGMENTATION
	// imshow("Raw Mask", srcRawMask); // noisy mask of the face of the clock
	// imshow("Mask", srcMask); // noise clear from the mask
	imshow("Clock", srcClock); // segments clock face from the background

	// CLOCK HANDS SEGMENTATION --------------------
	// Converts image to gray scale
	Mat clockGrayscale;
	cvtColor(srcClock, clockGrayscale, CV_BGR2GRAY);

	// Canny edge detection
	Mat clockEdges;
	Canny(clockGrayscale, clockEdges, 200, 250, 3);

	// Detects staight lines
	vector<Vec4i> linesUnprocessed;
	HoughLinesP(clockEdges, linesUnprocessed, 1, CV_PI / 180, 10, 30, 15);

	// Clock display center of mass
	Rect displayBounding = boundingRect(bestContour);
	Point displayCenterOfMass = Point(displayBounding.x + displayBounding.width / 2, displayBounding.y + displayBounding.height / 2);

	// Select lines closer to the center of the clock display
	double linesThreshold = .08;
	double pivotLineDistance = distanceCalculate(displayCenterOfMass,
	                                             Point(
			                                             (src.size().width / 2) * linesThreshold + displayCenterOfMass.x,
			                                             (src.size().height / 2) * linesThreshold + displayCenterOfMass.y));
	vector<Vec4i> lines = vector<Vec4i>();
	for (size_t i = 0; i < linesUnprocessed.size(); i++) {
		Vec4i l = linesUnprocessed[i];
		Point pt1 = Point(l[0], l[1]);
		Point pt2 = Point(l[2], l[3]);

		if (distanceCalculate(pt1, displayCenterOfMass) < pivotLineDistance || distanceCalculate(pt2, displayCenterOfMass) < pivotLineDistance) {
			lines.push_back(l);
		}
	}

	// Vector mapping angles and lines
	vector<pair<double, pair<Vec4i, double>>> anglesVec = vector<pair<double, pair<Vec4i, double>>>();

	// Finds the angles of the lines
	for (size_t i = 0; i < lines.size(); i++) {
		Vec4i l = lines[i];
		Point pt1 = Point(l[0], l[1]);
		Point pt2 = Point(l[2], l[3]);

		Point distantPoint;
		if (distanceCalculate(displayCenterOfMass, pt1) > distanceCalculate(displayCenterOfMass, pt2))
			distantPoint = Point(pt1.x - displayCenterOfMass.x, pt1.y - displayCenterOfMass.y);
		else
			distantPoint = Point(pt2.x - displayCenterOfMass.x, pt2.y - displayCenterOfMass.y);

		double radAngle = atan2(distantPoint.y, distantPoint.x) + atan2(1, 0);
		double degreeAngle = (radAngle * 360) / (2 * CV_PI);
		if (degreeAngle < 0)
			degreeAngle += 360;

		anglesVec.push_back(pair<double, pair<Vec4i, double>>(degreeAngle,
		                                                      pair<Vec4i, double>(l, distanceCalculate(displayCenterOfMass, distantPoint))));
	}

	// Comparator used in sorting algorithm
	struct ComparatorAngle {
		bool operator() (pair<double, pair<Vec4i, double>> i, pair<double, pair<Vec4i, double>> j) {
			return (i.first < j.first);
		}
	} comparator;

	// Sorting process
	sort(anglesVec.begin(), anglesVec.end(), comparator);

	// Remove irrelevant or dup lines
	double degreeDiff = 6;
	REPEAT:
	for (int i = 0; i < anglesVec.size() - 1; i++) {
		if (abs(anglesVec[i].first - anglesVec[i + 1].first) < degreeDiff) {
			Vec4i lineTemp;
			double distanceTemp;
			if (anglesVec[i].second.second > anglesVec[i + 1].second.second)
				distanceTemp = anglesVec[i].second.second;
			else
				distanceTemp = anglesVec[i + 1].second.second;

			Vec4i l1 = anglesVec[i].second.first;
			Vec4i l2 = anglesVec[i + 1].second.first;
			if (distanceCalculate(Point(l1[0], l1[1]), Point(l1[2], l1[3]))
			    > distanceCalculate(Point(l2[0], l2[1]), Point(l2[2], l2[3])))
				lineTemp = l1;
			else
				lineTemp = l2;

			pair<double, pair<Vec4i, double>> temp = pair<double, pair<Vec4i, double>>((anglesVec[i].first + anglesVec[i + 1].first) / 2, pair<Vec4i, double>(lineTemp, distanceTemp));
			anglesVec[i].swap(temp);
			anglesVec.erase(anglesVec.begin() + (i + 1));

			goto REPEAT;
		}
	}

	// Create new defined lines
	vector<Vec4i> handsLines = vector<Vec4i>();
	for (int i = 0; i < anglesVec.size(); i++)
		handsLines.push_back(anglesVec[i].second.first);

	// Show hands detection
	Mat clockFinalized;
	srcClock.copyTo(clockFinalized);

	// Calculates times
	if (anglesVec.size() == 2) {
		Vec4i l1 = anglesVec[0].second.first;
		Vec4i l2 = anglesVec[1].second.first;

		if (distanceCalculate(Point(l1[0], l1[1]), Point(l1[2], l1[3]))
		    < distanceCalculate(Point(l2[0], l2[1]), Point(l2[2], l2[3]))) {
			int hour, minutes;
			hour = anglesVec[0].first * 12 / 360;
			minutes = anglesVec[1].first * 60 / 360;
			cout << "Draws hours in blue, minutes in green and seconds in red." << endl;
			cout << "The clock time is: " << hour << "h " << minutes << "m" << endl << endl;

			// Draws hours blue, minutes green and seconds red
			Vec4i l1 = anglesVec[0].second.first;
			line(clockFinalized, Point(l1[0], l1[1]), Point(l1[2], l1[3]), Scalar(255, 0, 0), 2, CV_AA);
			Vec4i l2 = anglesVec[1].second.first;
			line(clockFinalized, Point(l2[0], l2[1]), Point(l2[2], l2[3]), Scalar(0, 255, 0), 2, CV_AA);
		} else {
			int hour, minutes;
			hour = anglesVec[1].first * 12 / 360;
			minutes = anglesVec[0].first * 60 / 360;
			cout << "Draws hours in blue, minutes in green and seconds in red." << endl;
			cout << "The clock time is: " << hour << "h " << minutes << "m" << endl << endl;

			// Draws hours blue, minutes green and seconds red
			Vec4i l1 = anglesVec[1].second.first;
			line(clockFinalized, Point(l1[0], l1[1]), Point(l1[2], l1[3]), Scalar(255, 0, 0), 2, CV_AA);
			Vec4i l2 = anglesVec[0].second.first;
			line(clockFinalized, Point(l2[0], l2[1]), Point(l2[2], l2[3]), Scalar(0, 255, 0), 2, CV_AA);
		}

		// Draws center of clock display
		circle(clockFinalized, displayCenterOfMass, 2, Scalar(255, 255, 255), 2);

	} else if (anglesVec.size() == 3) {
		pair<double, pair<Vec4i, double>> secondsPair;
		double distanceHandPoint = 0;
		for (int i = 0; i < anglesVec.size(); i++) {
			Vec4i l = anglesVec[i].second.first;
			Point pt1 = Point(l[0], l[1]);
			Point pt2 = Point(l[2], l[3]);

			Point closerPoint;
			if (distanceCalculate(displayCenterOfMass, pt1) < distanceCalculate(displayCenterOfMass, pt2))
				closerPoint = pt1;
			else
				closerPoint = pt2;

			if (distanceHandPoint < distanceCalculate(displayCenterOfMass, closerPoint)) {
				secondsPair = anglesVec[i];
				distanceHandPoint = distanceCalculate(displayCenterOfMass, closerPoint);
			}
		}

		vector<pair<double, pair<Vec4i, double>>> handsTemp = vector<pair<double, pair<Vec4i, double>>>();
		for (int i = 0; i < anglesVec.size(); i++) {
			if (anglesVec[i].first != secondsPair.first)
				handsTemp.push_back(anglesVec[i]);
		}

		if (handsTemp[0].second.second < handsTemp[1].second.second) {
			int hour, minutes, seconds;
			hour = handsTemp[0].first * 12 / 360;
			minutes = handsTemp[1].first * 60 / 360;
			seconds = secondsPair.first * 60 / 360;
			cout << "Draws hours in blue, minutes in green and seconds in red." << endl;
			cout << "The clock time is: " << hour << "h " << minutes << "m " << seconds << "s" << endl << endl;

			// Draws hours blue, minutes green and seconds red
			Vec4i l1 = handsTemp[0].second.first;
			line(clockFinalized, Point(l1[0], l1[1]), Point(l1[2], l1[3]), Scalar(255, 0, 0), 2, CV_AA);
			Vec4i l2 = handsTemp[1].second.first;
			line(clockFinalized, Point(l2[0], l2[1]), Point(l2[2], l2[3]), Scalar(0, 255, 0), 2, CV_AA);
			Vec4i l3 = secondsPair.second.first;
			line(clockFinalized, Point(l3[0], l3[1]), Point(l3[2], l3[3]), Scalar(0, 0, 255), 2, CV_AA);
		} else {
			int hour, minutes, seconds;
			hour = handsTemp[1].first * 12 / 360;
			minutes = handsTemp[0].first * 60 / 360;
			seconds = secondsPair.first * 60 / 360;
			cout << "Draws hours in blue, minutes in green and seconds in red." << endl;
			cout << "The clock time is: " << hour << "h " << minutes << "m " << seconds << "s" << endl << endl;

			// Draws hours blue, minutes green and seconds red
			Vec4i l1 = handsTemp[1].second.first;
			line(clockFinalized, Point(l1[0], l1[1]), Point(l1[2], l1[3]), Scalar(255, 0, 0), 2, CV_AA);
			Vec4i l2 = handsTemp[0].second.first;
			line(clockFinalized, Point(l2[0], l2[1]), Point(l2[2], l2[3]), Scalar(0, 255, 0), 2, CV_AA);
			Vec4i l3 = secondsPair.second.first;
			line(clockFinalized, Point(l3[0], l3[1]), Point(l3[2], l3[3]), Scalar(0, 0, 255), 2, CV_AA);
		}

		// Draws center of clock display
		circle(clockFinalized, displayCenterOfMass, 2, Scalar(255, 255, 255), 2);

	} else
		cout << "Something went wrong!" << endl;

	// DEBUGGING -----------------------------------
	// CLOCK HANDS SEGMENTATION
	// imshow("Clock Grayscale", clockGrayscale); // converts image to grayscale
	// imshow("Clock Edges", clockEdges); // finds clock display's edges with canny detector
	imshow("Clock Hands", clockFinalized); // finalized detection
}

/**
 * Application's starting point
 */
int main() {
	// Reads the image
	Mat src = getImage();

	// Verify reading success
	if (!src.data) {
		cout << "ERROR: Image file not found!" << endl;
		return 1;
	}

	// Image processing
	imageProcessing(src);

	// Wait for a key press
	cout << "Press any key to exit..." << endl;
	waitKey(0);

	return 0;
}
