# Import required packages
import os
import sys
import cv2 as cv
import tensorflow as tf

# Global variable
data = []

# Constant variables
SCALE = 2
OFFSET = 20
DECIMAL = 6
THICKNESS = 2
COLOR = (0, 0, 255)
FONT = cv.FONT_HERSHEY_PLAIN

# Classify a given image
def classifyImage(imagePath):
    # Access the matrix
    global data

    # Disable tensorflow compilation warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Get image data
    imageData = tf.gfile.FastGFile(imagePath, 'rb').read()

    # Read label lines
    labelLines = [line.rstrip() for line in tf.gfile.GFile('./labels.txt')]

    # Load model
    with tf.gfile.FastGFile('./model.pb', 'rb') as f:
        graphDef = tf.GraphDef()
        graphDef.ParseFromString(f.read())
        _ = tf.import_graph_def(graphDef, name='')

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
            row = [''] * 2
            label = labelLines[nodeID]
            score = predictions[0][nodeID]

            # Check if this new score is larger than the last
            if score > largestScore:
                # Set row's data and add it to matrix
                row[0] = label.upper()
                row[1] = '|  ' + str(round(score * 100, DECIMAL)) + ' %'
                data.append(row)

                # Set newest values to return
                largestScore = score
                largestLabel = label
            else:
                # Set row's data and add it to matrix
                row[0] = label.title()
                row[1] = '|  ' + str(round(score * 100, DECIMAL)) + ' %'
                data.append(row)
    
    # Return most probable candidate
    return { 'label': largestLabel, 'score': largestScore }

# Check CMD arguments
if len(sys.argv) != 2:
    print('ERROR: Invalid number of arguments!')
    print('Usage: python %s <path_to_image>' % (sys.argv[0]))
    print('Where:\n\t<path_to_image> is the path to the image')
    exit()

# Load the image
imagePath = sys.argv[1]
img = cv.imread(imagePath)

# Check if the image really exists
if img is None:
    print('ERROR: The image \'%s\' was not found!' % (sys.argv[1]))
    print('Usage: python %s <path_to_image>' % (sys.argv[0]))
    print('Where:\n\t<path_to_image> is the path to the image')
    exit()

# Classify the image
result = classifyImage(imagePath)

# Parse result into a string
score = str(round(result['score'] * 100, 2))
imgLabel = result['label'] + ' - ' + score + '%'

# Get image's dimensions and text size
imgHeight, imgWidth, imgChannels = img.shape
textWidth, textHeight = cv.getTextSize(imgLabel, FONT, SCALE, THICKNESS)[0]

# Calculate text's position and draw it
org = (int((imgWidth - textWidth) / 2), imgHeight - OFFSET)
cv.putText(img, imgLabel.upper(), org, FONT, SCALE, COLOR, THICKNESS)

# Display final image
cv.imshow(result['label'].title(), img)

# Print all of the predictions
print()
print("Prediction results:")
colWidth = max(len(word) for row in data for word in row) + 2
i = 0
for row in data:
	if i < 5:
		i += 1
		print(''.join(word.ljust(colWidth) for word in row))
		
print("\nPress any key to close window and stop execution!")

# Wait for any key press
cv.waitKey(0)
cv.destroyAllWindows()
