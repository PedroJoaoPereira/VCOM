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
static const int CALCULATE_DESCRIPTORS_MAT = 1;
static const int CALCULATE_VOCABULARY = 2;
static const int CALCULATE_HISTOGRAMS = 3;
static const int CALCULATE_TRAINDATA = 4;
static const int CALCULATE_MODEL = 5;

static const int VOCABULARY_WORDS = 1500;
static const string VOCABULARY_DESCRIPTORS_PATH = "Vocabulary_Descriptors";
static const string DESCRIPTORS_PATH = "Calculated_Descriptors";
static const string MODEL_HISTOGRAMS_PATH = "Model_Descriptors";
static const string HISTOGRAMS_PATH = "Calculated_Histograms";
static const string LABELS_FILENAME = "ImageLabels.txt";
static const string UNIQUE_LABELS_FILENAME = "UniqueImageLabels.txt";
static const string RESPONSES_FILENAME = "Responses.txt";
static const string MODEL_PATH = "Model_Result";

void MatRead(string filename, Mat &result);
void MatWrite(string filename, Mat &mat);

void loadImagesDirFromPath(string datasetDirectory, bool isCustom, int step, vector<string> &imageLabels, vector<string> &imageDirs);
void calculateImageDescriptors(string savingDirectory, vector<string> &imageDirs);
void calculateVocabulary(Mat &descriptors, Mat &vocabulary);
void calculateImageHistograms(string savingDirectory, vector<string> &imageDirs, BOWImgDescriptorExtractor &bowDE);

int main(int argc, char** argv) {
	// Read calling arguments
	// If it is trainning (n bag of words, dataset...), if it is only recnozing single image or multiple features in image
	// TODO

	// Debug
	MODE = CALCULATE_DESCRIPTORS;

	switch (MODE) {
		case CALCULATE_DESCRIPTORS:
		{
			// Initializes descriptors calculator
			cout << "[STARTING] Mode: Descriptors Calculator" << endl;

			// Load images to calculate the descriptors
			vector<string> imageLabels = vector<string>();
			vector<string> imageDirs = vector<string>();
			loadImagesDirFromPath(".\\AID", false, 4, imageLabels, imageDirs);

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
		case CALCULATE_DESCRIPTORS_MAT:
		{
			// Initializes mat for vocabulary calculator
			cout << "[STARTING] Mode: Prepare Mat For Vocabulary Calculator" << endl;

			// Load image labels
			ifstream inFile;
			inFile.open(VOCABULARY_DESCRIPTORS_PATH + "\\" + LABELS_FILENAME);
			if (!inFile) {
				cout << "[ERROR] Mode: Prepare Mat For Vocabulary Calculator - Lacking Labels File" << endl;
				break;
			}
			
			cout << "Reading Image Labels ..." << endl;

			// Read image labels from file
			vector<string> imageLabels = vector<string>();
			string line;
			while (inFile >> line) {
				imageLabels.push_back(line);
			}
			inFile.close();

			// Load calculated descriptors
			Mat descriptors;
			for (int i = 0; i < imageLabels.size(); i++) {
				string descriptorFileName = VOCABULARY_DESCRIPTORS_PATH + "\\" + DESCRIPTORS_PATH
					+ "\\" + "descriptor" + to_string(i) + ".bin";

				cout << "Loading Features Of Image " << i + 1 << " / " << imageLabels.size() << endl;

				// Loads the descriptor
				Mat imageDescriptor;
				MatRead(descriptorFileName, imageDescriptor);

				// Saves descriptor in the result Mat
				descriptors.push_back(imageDescriptor);
			}

			// Save descriptors Mat
			cout << "Saving Descriptors Mat To A File ..." << endl;
			MatWrite(VOCABULARY_DESCRIPTORS_PATH + "\\" + "merged_descriptors.bin", descriptors);

			// Finalizes mat for vocabulary calculator
			cout << "[ENDING] Mode: Prepare Mat For Vocabulary Calculator" << endl;
			break;
		}
		case CALCULATE_VOCABULARY:
		{
			// Initializes vocabulary calculator
			cout << "[STARTING] Mode: Vocabulary Calculator" << endl;
			cout << "Vocabulary k = " << VOCABULARY_WORDS << endl;

			cout << "Loading Merged Descriptors Mat ..." << endl;

			// Load the merged descriptors Mat
			string mergedDescriptorsFileName = VOCABULARY_DESCRIPTORS_PATH + "\\" + "merged_descriptors.bin";
			Mat mergedDescriptors;
			MatRead(mergedDescriptorsFileName, mergedDescriptors);

			// Calculate vocabulary
			Mat vocabulary;
			calculateVocabulary(mergedDescriptors, vocabulary);

			// Save vocabulary Mat
			cout << "Saving Vocabulary To A File ..." << endl;
			MatWrite(VOCABULARY_DESCRIPTORS_PATH + "\\" + "vocabulary.bin", vocabulary);

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
			MatRead(VOCABULARY_DESCRIPTORS_PATH + "\\" + "vocabulary.bin", vocabulary);

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

			// Finalizes histogram calculator
			cout << "[ENDING] Mode: Histograms Calculator" << endl;
			break;
		}
		case CALCULATE_TRAINDATA:
		{
			// Initializes traindata for model calculator
			cout << "[STARTING] Mode: Prepare Traindata For Model Calculator" << endl;

			// Load image labels
			ifstream inFile;
			inFile.open(MODEL_HISTOGRAMS_PATH + "\\" + LABELS_FILENAME);
			if (!inFile) {
				cout << "[ERROR] Mode: Prepare Traindata For Model Calculator - Lacking Labels File" << endl;
				break;
			}

			cout << "Reading Image Labels ..." << endl;

			// Read image labels from file
			vector<string> imageLabels = vector<string>();
			string line;
			while (inFile >> line) {
				imageLabels.push_back(line);
			}
			inFile.close();

			// Unique labels variable
			vector<string> uniqueLabels;
			// Load calculated histograms
			Mat trainData;
			for (int i = 0; i < imageLabels.size(); i++) {
				string histogramFileName = MODEL_HISTOGRAMS_PATH + "\\" + HISTOGRAMS_PATH + "\\" + "histogram" + to_string(i) + ".bin";

				cout << "Loading Histogram Of Image " << i + 1 << " / " << imageLabels.size() << endl;

				// Loads the histogram
				Mat imageHistogram;
				MatRead(histogramFileName, imageHistogram);

				// Saves histogram in the training Mat
				trainData.push_back(imageHistogram);

				// Selects unique labels to a vector
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

			cout << "Saving Classes ... " << endl;

			// Save labels
			ofstream outFile;
			outFile.open(MODEL_HISTOGRAMS_PATH + "\\" + "UniqueLabels.txt");
			for (int i = 0; i < uniqueLabels.size() - 1; i++) {
				outFile << uniqueLabels.at(i) << endl;
			}
			outFile << uniqueLabels.at(uniqueLabels.size() - 1);
			outFile.close();

			cout << "Saving Traindata ... " << endl;

			// Save traindata
			MatWrite(MODEL_HISTOGRAMS_PATH + "\\" + "traindata.bin", trainData);
			MatWrite(MODEL_HISTOGRAMS_PATH + "\\" + "responses.bin", responses);

			// Finalizes traindata for model calculator
			cout << "[ENDING] Mode: Prepare Traindata For Model Calculator" << endl;
			break;
		}
		case CALCULATE_MODEL:
		{
			// Initializes model calculator
			cout << "[STARTING] Mode: Model Calculator" << endl;

			cout << "Loading Trainingdata And Responses ..." << endl;

			// Load traindata
			Mat trainData;
			MatRead(MODEL_HISTOGRAMS_PATH + "\\" + "traindata.bin", trainData);

			// Load responses
			Mat responses;
			MatRead(MODEL_HISTOGRAMS_PATH + "\\" + "responses.bin", responses);


			// Load image unique labels
			ifstream inFile;
			inFile.open(MODEL_HISTOGRAMS_PATH + "\\" + "UniqueLabels.txt");
			if (!inFile) {
				cout << "[ERROR] Mode: Model Calculator - Lacking Unique Labels File" << endl;
				break;
			}

			cout << "Reading Unique Labels ..." << endl;

			// Read image labels from file
			vector<string> uniqueLabels = vector<string>();
			string line;
			while (inFile >> line) {
				uniqueLabels.push_back(line);
			}
			inFile.close();

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

			cout << "Saving Classes ... " << endl;

			// Save labels
			ofstream outFile;
			outFile.open(modelResultPath + "UniqueLabels.txt");
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

void MatRead(string filename, Mat &result) {

	// Reads binary stram of data
	ifstream fs(filename, fstream::binary);

	// Loads Mat headers
	int rows, cols, type, channels;
	fs.read((char*)&rows, sizeof(int));
	fs.read((char*)&cols, sizeof(int));
	fs.read((char*)&type, sizeof(int));
	fs.read((char*)&channels, sizeof(int));

	// Loads data
	result = Mat(rows, cols, type);
	fs.read((char*)result.data, CV_ELEM_SIZE(type) * rows * cols);

	fs.close();
}

void MatWrite(string filename, Mat &mat) {

	// Writes binary stram of data
	ofstream fs(filename, fstream::binary);

	// Writes Mat headers
	int type = mat.type();
	int channels = mat.channels();
	fs.write((char*)&mat.rows, sizeof(int));
	fs.write((char*)&mat.cols, sizeof(int));
	fs.write((char*)&type, sizeof(int));
	fs.write((char*)&channels, sizeof(int));

	// Writes data
	if (mat.isContinuous()) {
		fs.write(mat.ptr<char>(0), (mat.dataend - mat.datastart));
	} else {
		int rowsz = CV_ELEM_SIZE(type) * mat.cols;
		for (int r = 0; r < mat.rows; ++r) {
			fs.write(mat.ptr<char>(r), rowsz);
		}
	}

	fs.close();
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
		string descriptorFileName = savingDirectory + "descriptor" + to_string(i) + ".bin";
		MatWrite(descriptorFileName, descriptors);
	}
}

void calculateVocabulary(Mat &descriptors, Mat &vocabulary) {

	cout << "Creating Vocabulary ... " << endl;

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
		string histogramFileName = savingDirectory + "histogram" + to_string(i) + ".bin";
		MatWrite(histogramFileName, normalizedHistogram);
	}
}
