#include <string>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <iterator>
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

void loadTrainningImagesPath(string datasetDirectory, vector<string> &labels, vector<string> &imageLabels, vector<string> &imageDirs);
Mat* detectFeaturesOfDataset(vector<string> imageDirs);
void createVocabulary(string dictionaryDirectory, Mat* featuresUnclustered);
void trainMachine(vector<string> &labels, vector<string> &imageLabels, vector<string> &imageDirs, Mat &vocabulary);
void predictFeature(vector<string> &labels, Ptr<SIFT> &siftObj, BOWImgDescriptorExtractor &bowDE, Ptr<ANN_MLP> &mlp, Mat &image);

int main(int argc, char** argv) {
	// Read calling arguments
	// If it is trainning (n bag of words, dataset...), if it is only recnozing single image or multiple features in image
	// TODO

	// Debug
	MODE = TRAIN_CLASSIFIER;

	switch (MODE) {
		case COMPUTE_VOCABULARY:
		{
			// Create vocabulary from dataset
			cout << "[STARTING] Mode: Create Vocabulary From Dataset" << endl;

			// Load trainning images path
			vector<string> labels = vector<string>();
			vector<string> imageLabels = vector<string>();
			vector<string> imageDirs = vector<string>();
			loadTrainningImagesPath(".\\AID", labels, imageLabels, imageDirs);

			// Detects features of dataset images
			Mat* featuresUnclustered = detectFeaturesOfDataset(imageDirs);

			// Train with image features
			createVocabulary(".\\", featuresUnclustered);

			cout << "[ENDING] Mode: Create Vocabulary From Dataset" << endl;
			break;
		}
		case TRAIN_CLASSIFIER:
		{
			// Train classifier from dataset vocabulary
			cout << "[STARTING] Mode: Train Classifier From Dataset Vocabulary" << endl;

			// Load trainning images path
			vector<string> labels = vector<string>();
			vector<string> imageLabels = vector<string>();
			vector<string> imageDirs = vector<string>();
			loadTrainningImagesPath(".\\AID", labels, imageLabels, imageDirs);

			// Load vocabulary
			Mat vocabulary;
			FileStorage fs(".\\dictionary.yml", FileStorage::READ);
			fs["dictionary"] >> vocabulary;

			// Machine learning SVM
			trainMachine(labels, imageLabels, imageDirs, vocabulary);

			cout << "[ENDING] Mode: Train Classifier From Dataset Vocabulary" << endl;
			break;
		}
		case ONLY_IMAGE_DETECTION:
		{
			// Detect feature of image
			cout << "[STARTING] Mode: Detect Feature Of Image" << endl;

			// Opens labels file
			ifstream inFile;
			inFile.open("labels.txt");

			// Verifies file
			if (!inFile) {
				cout << "[ERROR] Mode: Detect Feature Of Image" << endl;
				cout << "[ERROR] Reason: File Not Valid" << endl;
				return -1;
			}

			// Load labels
			vector<string> labels = vector<string>();
			string line;
			while (inFile >> line) {
				labels.push_back(line);
			}
			inFile.close();

			// Load vocabulary
			Mat vocabulary;
			FileStorage fs(".\\dictionary.yml", FileStorage::READ);
			fs["dictionary"] >> vocabulary;

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

			// Create neural network
			Ptr<ANN_MLP> mlp = ANN_MLP::create();
			mlp->load(".\\model.yaml");

			// Loads image in grayscale
			Mat imageToPredict = imread(".\\Beach_Test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
			if (!imageToPredict.data) {
				cout << "[ERROR] Mode: Detect Feature Of Image" << endl;
				cout << "[ERROR] Reason: Could Not Load Image" << endl;
				return -1;
			}

			// Predict image
			predictFeature(labels, siftObj, bowDE, mlp, imageToPredict);

			cout << "[ENDING] Mode: Detect Feature Of Image" << endl;
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

void loadTrainningImagesPath(string datasetDirectory, vector<string> &labels, vector<string> &imageLabels, vector<string> &imageDirs) {

	cout << "Fetching dataset images paths ..." << endl;
	
	// DEBUG
	int i = 0;

	// Iterate through dataset directory
	for (auto & fileItem : fs::directory_iterator(datasetDirectory)) {
		stringstream ss = stringstream();
		ss << fileItem;

		// Get label of images
		string label = ss.str();
		label = label.substr(label.find_last_of("\\") + 1);

		// DEBUG
		/*if (imageDirs.size() >= 6)
			return;*/

		labels.push_back(label);

		// Iterate through a labeled image directory
		for (auto & imageItem : fs::directory_iterator(fileItem)) {
			stringstream ss = stringstream();
			ss << imageItem;

			// Get images directory and label
			imageDirs.push_back(ss.str());
			imageLabels.push_back(label);

			// DEBUG
			i++;
			if (i >= 700) {
				return;
				/*i = 0;
				break;*/
			}
		}
	}
}

Mat* detectFeaturesOfDataset(vector<string> imageDirs) {

	// Create SIFT features detector
	Ptr<SIFT> siftObj = SIFT::create();

	Mat* featuresUnclustered = new Mat[imageDirs.size() / 2];

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
		featuresUnclustered[i / 2] = descriptors;
		cout << "Detecting features of image " << (i + 1) / 2 << " / " << imageDirs.size() / 2 << endl;
	}

	return featuresUnclustered;
}

void createVocabulary(string dictionaryDirectory, Mat* featuresUnclustered) {

	cout << "Creating vocabulary of images ... " << endl;

	// Cluster a bag of words with kmeans
	BOWKMeansTrainer bowTrainer(1000, TermCriteria(), 1, KMEANS_PP_CENTERS);
	Mat vocabulary = bowTrainer.cluster(*featuresUnclustered);

	// Store dictionary
	FileStorage fs(dictionaryDirectory + "dictionary.yml", FileStorage::WRITE);
	fs << "dictionary" << vocabulary;
	fs.release();
}

void trainMachine(vector<string> &labels, vector<string> &imageLabels, vector<string> &imageDirs, Mat &vocabulary) {

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
	Mat* labelsArr = new Mat[imageDirs.size()];

	// Bag of words
	Mat* bagOfWords = new Mat[imageDirs.size()];

	// Prepare data to train machine
	for (int i = 0; i < imageDirs.size(); i++) {
		// Loads image in grayscale
		Mat imageToTrain = imread(imageDirs.at(i), CV_LOAD_IMAGE_GRAYSCALE);
		if (!imageToTrain.data) {
			cout << "Error could not read image " + i << endl;
			return;
		}

		// Detects image keypoints
		vector<KeyPoint> keypoints;
		// Creates descriptors from image keypoints
		Mat descriptors;

		// Calculates histogram of words
		siftObj->detect(imageToTrain, keypoints);
		bowDE.compute(imageToTrain, keypoints, descriptors);

		// Normalize histogram
		Mat normalizedHistogram;
		normalize(descriptors, normalizedHistogram, 0, descriptors.rows, NORM_MINMAX, -1, Mat());

		// Saves to bag of words
		bagOfWords[i] = normalizedHistogram;
		// Saves label
		int labelIndex = find(labels.begin(), labels.end(), imageLabels.at(i)) - labels.begin();
		// Create binary label
		Mat imageLabel = Mat::zeros(Size((int)labels.size(), 1), CV_32F);
		imageLabel.at<float>(labelIndex) = 1;
		labelsArr[i] = imageLabel;

		cout << "Detecting features of image " << i + 1 << " / " << imageDirs.size() << endl;

		cout << normalizedHistogram.row << " | " << normalizedHistogram.cols << " | " << imageDirs.size() << endl;
	}

	// Train Data for train
	Mat trainData(imageDirs.size(), 1000, CV_32FC1);
	for (int i = 0; i < imageDirs.size(); i++) {
		bagOfWords[i].row(0).copyTo(trainData.row(i));
		bagOfWords[i].release();
	}
	// Labels Mat for train
	Mat labelsMat(imageDirs.size(), labels.size(), CV_32F);
	for (int i = 0; i < imageDirs.size(); i++) {
		labelsArr[i].row(0).copyTo(labelsMat.row(i));
		labelsArr[i].release();
	}

	// Create layers for neural network
	Mat_<int> layerSizes(1, 3);
	layerSizes(0, 0) = trainData.cols;
	layerSizes(0, 1) = trainData.cols / 2;
	layerSizes(0, 2) = labelsMat.cols;

	// Create neural network
	Ptr<ANN_MLP> mlp = ANN_MLP::create();
	mlp->setLayerSizes(layerSizes);
	mlp->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0.1, 0.1);
	mlp->setTrainMethod(ANN_MLP::BACKPROP, 0.1, 0.1);
	Ptr<TrainData> trainDataObj = TrainData::create(trainData, ROW_SAMPLE, labelsMat);

	// Train with data
	mlp->train(trainDataObj);

	// Save SVM model
	mlp->save("model.yaml");	

	// Parse labels to get only unique labels
	vector<string> labelsToExport = vector<string>();
	for (int i = 0; i < labels.size(); i++) {
		if (find(labelsToExport.begin(), labelsToExport.end(), labels.at(i)) == labelsToExport.end())
			labelsToExport.push_back(labels.at(i));
	}

	// Save labels
	ofstream outFile;
	outFile.open("labels.txt");
	for (int i = 0; i < labelsToExport.size() - 1; i++) {
		outFile << labelsToExport.at(i) << endl;
	}
	outFile << labelsToExport.at(labelsToExport.size() - 1);
	outFile.close();
}

void predictFeature(vector<string> &labels, Ptr<SIFT> &siftObj, BOWImgDescriptorExtractor &bowDE, Ptr<ANN_MLP> &mlp, Mat &image) {

	// Detects image keypoints
	vector<KeyPoint> keypoints;
	// Creates descriptors from image keypoints
	Mat descriptors;

	siftObj->detect(image, keypoints);
	bowDE.compute(image, keypoints, descriptors);

	// Detect feature
	float response = mlp->predict(descriptors);

	cout << response << endl;
}