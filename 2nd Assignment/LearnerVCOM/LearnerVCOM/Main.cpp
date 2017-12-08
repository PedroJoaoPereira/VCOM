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

static int MODE = -1;
static const int COMPUTE_VOCABULARY = 0;
static const int TRAIN_CLASSIFIER = 1;
static const int ONLY_IMAGE_DETECTION = 2;
static const int MULTIPLE_IMAGE_DETECTION = 3;

void loadTrainningImagesPath(string datasetDirectory, vector<string> &labels, vector<string> &imageDirs);
void detectFeaturesOfDataset(vector<string> imageDirs, Mat &featuresUnclustered);
void createVocabulary(string dictionaryDirectory, Mat &featuresUnclustered);

int main(int argc, char** argv) {
	// Read calling arguments
	// If it is trainning (n bag of words, dataset...), if it is only recnozing single image or multiple features in image
	// TODO

	// Debug
	MODE = COMPUTE_VOCABULARY;

	switch (MODE) {
		case COMPUTE_VOCABULARY:
		{
			// Create vocabulary from dataset
			cout << "Mode: Create Vocabulary From Dataset" << endl;

			// Load trainning images path
			vector<string> labels = vector<string>();
			vector<string> imageDirs = vector<string>();
			loadTrainningImagesPath(".\\AID", labels, imageDirs);

			// Detects features of dataset images
			Mat featuresUnclustered = Mat();
			detectFeaturesOfDataset(imageDirs, featuresUnclustered);

			// Train with image features
			createVocabulary(".\\", featuresUnclustered);
			break;
		}
		case TRAIN_CLASSIFIER:
		{
			// Train classifier from dataset vocabulary
			cout << "Mode: Train Classifier From Dataset Vocabulary" << endl;

			// Load trainning images path
			vector<string> labels = vector<string>();
			vector<string> imageDirs = vector<string>();
			loadTrainningImagesPath(".\\AID", labels, imageDirs);

			// Load vocabulary
			Mat vocabulary;
			FileStorage fs(".\\dictionary.yml", FileStorage::READ);
			fs["dictionary"] >> vocabulary;

			break;
		}
		case ONLY_IMAGE_DETECTION:
		{
			// Create vocabulary from dataset
			cout << "Mode: Create Vocabulary From Dataset" << endl;
			break;
		}
		case MULTIPLE_IMAGE_DETECTION:
		{
			// Create vocabulary from dataset
			cout << "Mode: Create Vocabulary From Dataset" << endl;
			break;
		}
	}	

	system("pause");
	return 0;
}

void loadTrainningImagesPath(string datasetDirectory, vector<string> &labels, vector<string> &imageDirs) {

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

void detectFeaturesOfDataset(vector<string> imageDirs, Mat &featuresUnclustered) {

	// Create SIFT features detector
	Ptr<SIFT> siftObj = SIFT::create();

	// Iterate through images to train
	#pragma omp parallel for schedule(dynamic, 3)
	for (int i = 0; i < imageDirs.size(); i = i + 4) {
		// Loads image in grayscale
		Mat imageToTrain = imread(imageDirs.at(i), CV_LOAD_IMAGE_GRAYSCALE);
		if (!imageToTrain.data)
			continue;

		// Detects image keypoints
		vector<KeyPoint> keypoints;
		// Creates descriptors from image keypoints
		Mat descriptors;

		siftObj->detectAndCompute(imageToTrain, Mat(), keypoints, descriptors);

		#pragma omp critical
		{
			// Saves the image descriptors
			featuresUnclustered.push_back(descriptors);
			cout << "Detecting features of image " << (i + 1) / 4 << " / " << imageDirs.size() / 4 << endl;
		}
	}
}

void createVocabulary(string dictionaryDirectory, Mat &featuresUnclustered) {

	cout << "Creating vocabulary of images ... " << endl;

	// Cluster a bag of words with kmeans
	Mat vocabulary;
	kmeans(featuresUnclustered, 300, Mat(), TermCriteria(), 1, KMEANS_PP_CENTERS, vocabulary);

	// Store dictionary
	FileStorage fs(dictionaryDirectory + "dictionary.yml", FileStorage::WRITE);
	fs << "dictionary" << vocabulary;
	fs.release();
}