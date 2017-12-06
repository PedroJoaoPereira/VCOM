#include <string>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
namespace fs = std::experimental::filesystem;

vector<string> labels = vector<string>();
vector<string> imageDirs = vector<string>();

void loadTrainningImagesPath(string datasetDirectory);
void trainWithImages();

int main(int argc, char** argv) {
	// Read calling arguments
	// TODO

	// Load trainning images path
	loadTrainningImagesPath(".\\AID");

	system("pause");
	return 0;
}

void loadTrainningImagesPath(string datasetDirectory) {

	cout << "Fetching dataset images paths ..." << endl;
	// Iterate through dataset directory
	for (auto & fileItem : fs::directory_iterator(datasetDirectory)) {
		stringstream ss = stringstream();
		ss << fileItem;

		// Get label of images
		string label = ss.str();
		label = label.substr(label.find_last_of("\\") + 1);
		labels.push_back(label);

		// Iterate through a labeled image directory
		for (auto & imageItem : fs::directory_iterator(fileItem)) {
			stringstream ss = stringstream();
			ss << imageItem;

			// Get images directory
			imageDirs.push_back(ss.str());
		}
	}
}

void trainWithImages() {

	Ptr<SIFT> detector = SIFT::create();

	// Iterate through images to train
	for (int i = 0; i < imageDirs.size(); i++) {
		cout << "Trainning with image " << i + 1 << " / " << imageDirs.size() << endl;
		Mat imageToTrain = imread(imageDirs.at(i), CV_LOAD_IMAGE_GRAYSCALE);
		if (!imageToTrain.data)
			continue;

		vector<KeyPoint> keypoints = vector<KeyPoint>();
		detector->detect(imageToTrain, keypoints);
	}

	/*
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
	//Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
	//Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
	Mat image;
	std::vector<KeyPoint> keypoints;
	Mat descriptors;
	// for cycle to extract from all of the 50k images from the train folder
	for (int i = 0; i<listOfImages.size(); i++) {
		if (!openImage("train/" + listOfImages[i], image))
			continue;
		cout << "Extracting from Image: " + listOfImages[i] + " of " + to_string(listOfImages.size()) << endl;
		detector->detect(image, keypoints);
		extractor->compute(image, keypoints, descriptors);
		allTrainDescriptors.push_back(descriptors);
	}
	*/
}