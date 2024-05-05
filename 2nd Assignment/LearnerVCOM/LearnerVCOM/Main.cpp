#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;
namespace fs = std::experimental::filesystem;

static int MODE = -1;
static const int CALCULATE_DESCRIPTORS = 0;
static const int CALCULATE_VOCABULARY = 1;
static const int CALCULATE_HISTOGRAMS = 2;
static const int CALCULATE_MODEL = 3;

static const int VOCABULARY_WORDS = 1000;
static const string VOCABULARY_DESCRIPTORS_PATH = "Vocabulary_Descriptors";
static const string DESCRIPTORS_PATH = "Calculated_Descriptors";
static const string MODEL_HISTOGRAMS_PATH = "Model_Descriptors";
static const string HISTOGRAMS_PATH = "Calculated_Histograms";
static const string LABELS_FILENAME = "ImageLabels.txt";
static const string UNIQUE_LABELS_FILENAME = "UniqueImageLabels.txt";
static const string RESPONSES_FILENAME = "Responses.txt";
static const string MODEL_PATH = "Model_Result";

void loadImagesDirFromPath(string datasetDirectory, bool isCustom, int step, vector<string> &imageLabels, vector<string> &imageDirs);
void calculateImageDescriptors(string savingDirectory, vector<string> &imageDirs);
void calculateVocabulary(Mat &descriptors, Mat &vocabulary);
void calculateImageHistograms(string savingDirectory, vector<string> &imageDirs, BOWImgDescriptorExtractor &bowDE);

int main(int argc, char** argv) {
	// Read calling arguments
	// If it is trainning (n bag of words, dataset...), if it is only recnozing single image or multiple features in image
	// TODO

	// Debug
	MODE = CALCULATE_HISTOGRAMS;

	switch (MODE) {
		case CALCULATE_DESCRIPTORS:
		{
			// Initializes descriptors calculator
			cout << "[STARTING] Mode: Descriptors Calculator" << endl;

			// Load images to calculate the descriptors
			vector<string> imageLabels = vector<string>();
			vector<string> imageDirs = vector<string>();
			loadImagesDirFromPath(".\\AID", false, 2, imageLabels, imageDirs);

			// Calculates images descriptors
			fs::create_directory(VOCABULARY_DESCRIPTORS_PATH);
			fs::create_directory(VOCABULARY_DESCRIPTORS_PATH + "\\" + DESCRIPTORS_PATH);
			calculateImageDescriptors(VOCABULARY_DESCRIPTORS_PATH + "\\" + DESCRIPTORS_PATH + "\\", imageDirs);

			// Save image labels
			cout << "Saving Image Labels To A File ..." << endl;
			ofstream outFile;
			outFile.open(VOCABULARY_DESCRIPTORS_PATH + "\\" + LABELS_FILENAME);
			for (int i = 0; i < imageLabels.size() - 1; i++) {
				outFile << imageLabels.at(i) << endl;
			}
			outFile << imageLabels.at(imageLabels.size() - 1);
			outFile.close();

			// Finalizes descriptors calculator
			cout << "[ENDING] Mode: Descriptors Calculator" << endl;
			break;
		}
		case CALCULATE_VOCABULARY:
		{
			// Initializes vocabulary calculator
			cout << "[STARTING] Mode: Vocabulary Calculator" << endl;

			// Load image labels
			ifstream inFile;
			inFile.open(VOCABULARY_DESCRIPTORS_PATH + "\\" + LABELS_FILENAME);
			if (!inFile) {
				cout << "[ERROR] Mode: Vocabulary Calculator - Lacking Labels File" << endl;
				break;
			}
			vector<string> imageLabels = vector<string>();
			string line;
			while (inFile >> line) {
				imageLabels.push_back(line);
			}
			inFile.close();

			cout << "Loading descriptors ..." << endl;

			// Load calculated descriptors
			Mat descriptors;
			for (int i = 0; i < imageLabels.size(); i++) {
				string descriptorFileName = VOCABULARY_DESCRIPTORS_PATH + "\\" + DESCRIPTORS_PATH + "\\" + "descriptor" + to_string(i) + ".yml";
				Mat imageDescriptor;
				FileStorage fsImageDescriptor(descriptorFileName, FileStorage::READ);
				fsImageDescriptor["descriptor"] >> imageDescriptor;
				fsImageDescriptor.release();
				descriptors.push_back(imageDescriptor);
			}

			// Calculate vocabulary
			Mat vocabulary;
			calculateVocabulary(descriptors, vocabulary);

			// Save vocabulary Mat
			cout << "Saving Vocabulary To A File ..." << endl;
			FileStorage fsVocabulary(VOCABULARY_DESCRIPTORS_PATH + "\\" + "vocabulary.yml", FileStorage::WRITE);
			fsVocabulary << "vocabulary" << vocabulary;
			fsVocabulary.release();

			// Finalizes vocabulary calculator
			cout << "[ENDING] Mode: Vocabulary Calculator" << endl;
			break;
		}
		case CALCULATE_HISTOGRAMS:
		{
			// Initializes histogram calculator
			cout << "[STARTING] Mode: Histograms Calculator" << endl;

			// Load images to calculate the histograms
			vector<string> imageLabels = vector<string>();
			vector<string> imageDirs = vector<string>();
			loadImagesDirFromPath(".\\AID", false, 1, imageLabels, imageDirs);

			// Load calculated vocabulary
			Mat vocabulary;
			FileStorage fsVocabulary(VOCABULARY_DESCRIPTORS_PATH + "\\" + "vocabulary.yml", FileStorage::READ);
			if (!fsVocabulary.isOpened()) {
				cout << "[ERROR] Mode: Histograms Calculator - Lacking Vocabulary File" << endl;
				break;
			}
			fsVocabulary["vocabulary"] >> vocabulary;
			fsVocabulary.release();

			// Create Bag of Words object
			// Create SIFT features detector
			Ptr<DescriptorExtractor> extractor = SIFT::create();
			// Create Flann based matcher
			Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
			// Bag of words descriptor and extractor
			BOWImgDescriptorExtractor bowDE(extractor, matcher);
			// Sets previously obtained vocabulary	
			bowDE.setVocabulary(vocabulary);

			// Calculates images histograms
			fs::create_directory(MODEL_HISTOGRAMS_PATH);
			fs::create_directory(MODEL_HISTOGRAMS_PATH + "\\" + HISTOGRAMS_PATH);
			calculateImageHistograms(MODEL_HISTOGRAMS_PATH + "\\" + HISTOGRAMS_PATH + "\\", imageDirs, bowDE);

			// Save image labels
			cout << "Saving Image Labels To A File ..." << endl;
			ofstream outFile;
			outFile.open(MODEL_HISTOGRAMS_PATH + "\\" + LABELS_FILENAME);
			for (int i = 0; i < imageLabels.size() - 1; i++) {
				outFile << imageLabels.at(i) << endl;
			}
			outFile << imageLabels.at(imageLabels.size() - 1);
			outFile.close();

			// Finalizes histogra calculator
			cout << "[ENDING] Mode: Histograms Calculator" << endl;
			break;
		}
		case CALCULATE_MODEL:
		{
			// Initializes model calculator
			cout << "[STARTING] Mode: Model Calculator" << endl;

			// Load image labels
			ifstream inFile;
			inFile.open(MODEL_HISTOGRAMS_PATH + "\\" + LABELS_FILENAME);
			if (!inFile) {
				cout << "[ERROR] Mode: Model Calculator - Lacking Labels File" << endl;
				break;
			}
			vector<string> imageLabels = vector<string>();
			string line;
			while (inFile >> line) {
				imageLabels.push_back(line);
			}
			inFile.close();

			cout << "Loading histograms ..." << endl;

			// Unique labels variable
			vector<string> uniqueLabels;
			// Load calculated histograms
			Mat trainData;
			for (int i = 0; i < imageLabels.size(); i++) {
				string histogramFileName = MODEL_HISTOGRAMS_PATH + "\\" + HISTOGRAMS_PATH + "\\" + "histogram" + to_string(i) + ".yml";
				Mat imageHistogram;
				FileStorage fsImageHistogram(histogramFileName, FileStorage::READ);
				fsImageHistogram["histogram"] >> imageHistogram;
				fsImageHistogram.release();
				trainData.push_back(imageHistogram);

				if (find(uniqueLabels.begin(), uniqueLabels.end(), imageLabels.at(i)) == uniqueLabels.end())
					uniqueLabels.push_back(imageLabels.at(i));
			}

			cout << "Creating Responses ... " << endl;

			// Create responses
			Mat responses;
			for (int i = 0; i < imageLabels.size(); i++) {
				Mat imageResponse = Mat::zeros(Size((int)uniqueLabels.size(), 1), CV_32F);
				int index = find(uniqueLabels.begin(), uniqueLabels.end(), imageLabels.at(i)) - uniqueLabels.begin();
				imageResponse.at<float>(index) = 1;
				responses.push_back(imageResponse);
			}

			// Create layers for neural network
			Mat_<int> layerSizes(1, 3);
			layerSizes(0, 0) = trainData.cols;
			layerSizes(0, 1) = trainData.cols / 2;
			layerSizes(0, 2) = responses.cols;

			// Create neural network
			Ptr<ANN_MLP> mlp = ANN_MLP::create();
			mlp->setLayerSizes(layerSizes);
			mlp->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0.1, 0.1);
			mlp->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.1);
			Ptr<TrainData> trainDataObj = TrainData::create(trainData, ROW_SAMPLE, responses);

			cout << "Training Model ... " << endl;

			// Train with data
			mlp->train(trainDataObj);

			// Save data for future use
			string modelResultPath = MODEL_PATH + "\\";
			fs::create_directory(modelResultPath);

			cout << "Saving Responses ... " << endl;

			// Save labels
			ofstream outFile;
			outFile.open(modelResultPath + RESPONSES_FILENAME);
			for (int i = 0; i < uniqueLabels.size() - 1; i++) {
				outFile << uniqueLabels.at(i) << endl;
			}
			outFile << uniqueLabels.at(uniqueLabels.size() - 1);
			outFile.close();

			cout << "Saving Model ... " << endl;

			// Save SVM model
			mlp->save(modelResultPath + "model.yaml");

			// Finalizes model calculator
			cout << "[ENDING] Mode: Model Calculator" << endl;
			break;
		}
	}

	system("pause");
	return 0;
}

void loadImagesDirFromPath(string datasetDirectory, bool isCustom, int step, vector<string> &imageLabels, vector<string> &imageDirs) {

	if (step <= 0)
		return;

	cout << "Loading Images Directories ..." << endl;

	// Used variables
	int index = 0;

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
			if (!isCustom && index % step == 0) {
				imageDirs.push_back(ss.str());
				imageLabels.push_back(label);
			} else if (isCustom) {
				imageDirs.push_back(ss.str());
				imageLabels.push_back(label);
			}

			// Step through the dataset
			if(!isCustom)
				index++;
			else {
				if (index < step - 1)
					index++;
				else {
					index = 0;
					break;
				}
			}
		}
	}
}

void calculateImageDescriptors(string savingDirectory, vector<string> &imageDirs) {

	cout << "Calculating Images Descriptors ..." << endl;

	// Create SIFT features detector
	Ptr<SIFT> siftObj = SIFT::create();

	// Iterate through images to train
	for (int i = 0; i < imageDirs.size(); i++) {
		// Loads image in grayscale
		Mat imageObj = imread(imageDirs.at(i), CV_LOAD_IMAGE_GRAYSCALE);
		if (!imageObj.data)
			continue;

		// Detects image keypoints
		vector<KeyPoint> keypoints;
		// Creates descriptors from image keypoints
		Mat descriptors;

		cout << "Detecting And Saving Features Of Image " << i + 1 << " / " << imageDirs.size() << endl;

		// Calculates descriptors
		siftObj->detectAndCompute(imageObj, Mat(), keypoints, descriptors);
		// Saves the image descriptors
		string descriptorFileName = savingDirectory + "descriptor" + to_string(i) + ".yml";
		FileStorage fs(descriptorFileName, FileStorage::WRITE);
		fs << "descriptor" << descriptors;
		fs.release();
	}
}

void calculateVocabulary(Mat &descriptors, Mat &vocabulary) {

	cout << "Creating Descriptors Vocabulary ... " << endl;

	// Cluster vocabulary words with kmeans
	BOWKMeansTrainer bowTrainer(VOCABULARY_WORDS, TermCriteria(), 1, KMEANS_PP_CENTERS);
	vocabulary = bowTrainer.cluster(descriptors);
}

void calculateImageHistograms(string savingDirectory, vector<string> &imageDirs, BOWImgDescriptorExtractor &bowDE) {

	cout << "Calculating Images Histograms ..." << endl;

	// Create SIFT features detector
	Ptr<SIFT> siftObj = SIFT::create();

	// Iterate through images to train
	for (int i = 0; i < imageDirs.size(); i++) {
		// Loads image in grayscale
		Mat imageObj = imread(imageDirs.at(i), CV_LOAD_IMAGE_GRAYSCALE);
		if (!imageObj.data)
			continue;

		// Detects image keypoints
		vector<KeyPoint> keypoints;
		// Creates descriptors from image keypoints
		Mat descriptors;

		cout << "Detecting And Saving Histograms Of Image " << i + 1 << " / " << imageDirs.size() << endl;

		// Calculates histogram of words
		siftObj->detect(imageObj, keypoints);
		bowDE.compute(imageObj, keypoints, descriptors);

		// Normalize histogram
		Mat normalizedHistogram;
		normalize(descriptors, normalizedHistogram, 0, descriptors.rows, NORM_MINMAX, -1, Mat());

		// Saves the image histograms
		string histogramFileName = savingDirectory + "histogram" + to_string(i) + ".yml";
		FileStorage fs(histogramFileName, FileStorage::WRITE);
		fs << "histogram" << descriptors;
		fs.release();
	}
}
