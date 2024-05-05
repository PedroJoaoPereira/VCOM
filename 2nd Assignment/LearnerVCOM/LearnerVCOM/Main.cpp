#include <string>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;
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
void trainMachine(vector<string> &labels, vector<string> &imageDirs, Mat &vocabulary);

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

			// Machine learning SVM
			trainMachine(labels, imageDirs, vocabulary);

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

		// Iterate through a labeled image directory
		for (auto & imageItem : fs::directory_iterator(fileItem)) {
			stringstream ss = stringstream();
			ss << imageItem;

			// Get images directory and label
			imageDirs.push_back(ss.str());
			labels.push_back(label);
		}
	}
}

void detectFeaturesOfDataset(vector<string> imageDirs, Mat &featuresUnclustered) {

	// Create SIFT features detector
	Ptr<SIFT> siftObj = SIFT::create();

	// Iterate through images to train
	for (int i = 0; i < imageDirs.size(); i = i + 2) {
		// Loads image in grayscale
		Mat imageToTrain = imread(imageDirs.at(i), CV_LOAD_IMAGE_GRAYSCALE);
		if (!imageToTrain.data)
			continue;

		// Detects image keypoints
		vector<KeyPoint> keypoints;
		// Creates descriptors from image keypoints
		Mat descriptors;

		siftObj->detectAndCompute(imageToTrain, Mat(), keypoints, descriptors);

		// Saves the image descriptors
		featuresUnclustered.push_back(descriptors);
		cout << "Detecting features of image " << (i + 1) / 2 << " / " << imageDirs.size() / 2 << endl;
	}
}

void createVocabulary(string dictionaryDirectory, Mat &featuresUnclustered) {

	cout << "Creating vocabulary of images ... " << endl;

	// Cluster a bag of words with kmeans
	Mat vocabulary;
	kmeans(featuresUnclustered, 100, Mat(), TermCriteria(), 1, KMEANS_PP_CENTERS, vocabulary);

	// Store dictionary
	FileStorage fs(dictionaryDirectory + "dictionary.yml", FileStorage::WRITE);
	fs << "dictionary" << vocabulary;
	fs.release();
}

void trainMachine(vector<string> &labels, vector<string> &imageDirs, Mat &vocabulary) {

	cout << "Trainning machine ... " << endl;

	// Create SIFT features detector
	Ptr<SIFT> siftObj = SIFT::create();

	// Create SIFT features detector
	Ptr<DescriptorExtractor> extractor = SIFT::create();
	// Create Flann based matcher
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
	// Bag of words descriptor and extractor
	BOWImgDescriptorExtractor bowDE(extractor, matcher);

	// Sets previously obtained vocabulary
	bowDE.setVocabulary(vocabulary);

	// Store image labels
	int labelIndex = 0;
	Mat labelsMat(labels.size(), 1, CV_32FC1);

	// Create SVM object
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	// Bag of words
	Mat bagOfWords = Mat();

	// Prepare data to train machine
	for (int i = 0; i < imageDirs.size(); i++) {
		// Loads image in grayscale
		Mat imageToTrain = imread(imageDirs.at(i), CV_LOAD_IMAGE_GRAYSCALE);
		if (!imageToTrain.data)
			continue;

		// Detects image keypoints
		vector<KeyPoint> keypoints;
		// Creates descriptors from image keypoints
		Mat descriptors;

		siftObj->detect(imageToTrain, keypoints);
		bowDE.compute(imageToTrain, keypoints, descriptors);

		siftObj->detectAndCompute(imageToTrain, Mat(), keypoints, descriptors);

		// Saves to bag of words
		bagOfWords.push_back(descriptors);
		// Saves label
		if (i - 1 >= 0 && !labels.at(i - 1).compare(labels.at(i)))
			labelIndex++;
		labelsMat.push_back(labelIndex);

		cout << "Detecting features of image " << i << " / " << imageDirs.size() << endl;
	}

	// SVM trainning
	svm->train(bagOfWords, ROW_SAMPLE, labelsMat);

	// Save SVM model
	svm->save(".\\svm.xml");

	// Parse labels to get only unique labels
	vector<string> labelsToExport = vector<string>();
	for (int i = 0; i < labels.size(); i++) {
		if (find(labelsToExport.begin(), labelsToExport.end(), labels.at(i)) == labelsToExport.end())
			labelsToExport.push_back(labels.at(i));
	}

	// Save labels
	ofstream outFile;
	outFile.open("labels.txt");
	for (int i = 0; i < labelsToExport.size(); i++) {
		outFile << labelsToExport.at(i) << endl;
	}
	outFile.close();
}