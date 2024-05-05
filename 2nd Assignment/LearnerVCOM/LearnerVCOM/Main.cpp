#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

const String keys =
"{ ? h help || Usage For this Application }"
"{ @model || Path Of Model File (.pb) }"
"{ @labels || Path Of Labels File (.txt) }"
"{ @image || Path Of Image To Process }"
"{ b |false| Should Detect Multiple Classes (boolean) }"
;

int main(int argc, char **argv) {

	// Parses calling arguments
	CommandLineParser parser(argc, argv, keys);

	// If user requests for help
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	// Checks if all needed variables are filled
	bool hasModel = parser.has("@model");
	bool hasLabels = parser.has("@labels");
	bool hasImage = parser.has("@image");
	if (!hasModel || !hasLabels || !hasImage) {
		if (!hasModel)
			cout << "Missing Parameter: Path Of Model File!" << endl;
		if (!hasLabels)
			cout << "Missing Parameter: Path Of Labels File!" << endl;
		if (!hasImage)
			cout << "Missing Parameter: Path Of Image To Process!" << endl;
		return -1;
	}

	// Retrieves arguments passed by the user
	String modelFile = parser.get<String>("@model");
	String labelsFile = parser.get<String>("@labels");
	String imageFile = parser.get<String>("@image");
	bool isMultipleClassification = parser.get<bool>("b");

	// Checks if all parameters are complete
	if (!parser.check()) {
		parser.printErrors();
		return -1;
	}

	// Initializes neural network model
	Net model = readNetFromTensorflow(modelFile);
	if (model.empty()) {
		cout << "Can Not Load Model!" << endl;
		return -1;
	}

	// Opens labels file
	ifstream inFile(labelsFile);
	if (!inFile.is_open()) {
		cout << "Can Not Load Labels!" << endl;
		return -1;
	}

	// Reads class names from labels file
	vector<String> classes;
	string label;
	while (!inFile.eof()) {
		getline(inFile, label);

		// If it is not an empty line
		if (label.length())
			classes.push_back(label);
	}
	inFile.close();

	for (int i = 0; i < classes.size(); i++) {
		cout << classes.at(i) << endl;
	}

	return 0;
}