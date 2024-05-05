# Import required packages
import os
import sys
import pandas as pd
import tensorflow as tf

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load model and labels
labelLines = [line.rstrip() for line in tf.gfile.GFile('./labels.txt')]
with tf.gfile.FastGFile('./model.pb', 'rb') as f:
    graphDef = tf.GraphDef()
    graphDef.ParseFromString(f.read())
    _ = tf.import_graph_def(graphDef, name='')

# Global variable
accConfidence = 0
accCorrectlyPredicted = 0;
predictedLabels = []
allResults = []

# Classify a given image
def classifyImage(imagePath):
    # Get image data
    imageData = tf.gfile.FastGFile(imagePath, 'rb').read()

    # Final value to be returned
    largestLabel = ''
    largestScore = 0.0

    # Calculate predictions
    with tf.Session() as sess:
        softmaxTensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmaxTensor, { 'DecodeJpeg/contents:0': imageData })
        topK = predictions[0].argsort()[-len(predictions[0]):][::-1]

        # Iterate through each prediction
        for nodeID in topK:
            # Get label and score
            label = labelLines[nodeID]
            score = predictions[0][nodeID]

            # Check if this new score is larger than the last
            if score > largestScore:
                # Set newest values to return
                largestScore = score
                largestLabel = label
    
    # Return most probable candidate
    return { 'label': largestLabel, 'score': largestScore, 'path': imagePath }

# Check CMD arguments
if len(sys.argv) != 4:
    print('ERROR: Invalid number of arguments!')
    print('Usage: python %s <path_to_dataset> <images_format> <path_to_labels>' % (sys.argv[0]))
    print('Where:')
    print('\t<path_to_dataset> is the path to the dataset')
    print('\t<images_format> is the images format extension')
    print('\t<path_to_labels> is the path to the dataset images labels')
    exit()	

# Reads parameters passed by the user
dirPath = sys.argv[1]
imagesFormat = sys.argv[2]
actualLabelsPath = sys.argv[3]

# Load actual labels
actualLabels = [line.rstrip() for line in tf.gfile.GFile(actualLabelsPath)]

# Iterate through the dataset
print("\nPredictions:")
for i in range(0, len(actualLabels)):
	result = classifyImage(dirPath + "/" + str(i + 1) + "." + imagesFormat)
	accConfidence += result['score']
	predictedLabels.append(result['label'])
	allResults.append(result)
	if result['label'] == actualLabels[i]:
		accCorrectlyPredicted += 1
	strTemp = ' %s (confidence = %.5f)' % (result['label'], result['score'])
	print(str(i + 1) + " of " + str(len(actualLabels)) + " -> " + strTemp)

# Calculate metrics
avgConfidence = accConfidence / len(actualLabels)
avgAccuracy = accCorrectlyPredicted / len(actualLabels)
actualSeries = pd.Series(actualLabels, name='Actual')
predictedSeries = pd.Series(predictedLabels, name='Predicted')
confusionMatrix = pd.crosstab(actualSeries, predictedSeries, rownames=['Actual'], colnames=['Predicted'], margins=True)

# Save metrics in a .txt file
outputFile = open("results.txt", "w")
outputFile.write("Evaluation results for the processed dataset:\n")
outputFile.write("---------------------------------------------\n")
avgConfidenceStr = "%.5f" % (avgConfidence)
outputFile.write("Avg Confidence -> " + avgConfidenceStr + "\n")
avgAccuracyStr = "%.5f" % (avgAccuracy)
outputFile.write("Avg Accuracy -> " + avgAccuracyStr + "\n")
outputFile.write("---------------------------------------------\n")
outputFile.write("Confusion Matrix:\n")
ncmStr = confusionMatrix.to_csv(sep=',')
outputFile.write(ncmStr)
outputFile.write("---------------------------------------------\n")
outputFile.write("Image Predictions (path - prediction - confidence):\n")
for i in range(0, len(allResults)):
	tempStr = ' - %s - %.5f' % (allResults[i]['label'], allResults[i]['score'])
	writeStr = allResults[i]['path'] + tempStr + "\n"
	outputFile.write(writeStr)
outputFile.close()

print("\nExecution has finished!")