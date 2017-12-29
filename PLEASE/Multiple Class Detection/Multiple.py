# Import required packages
import os
import sys
import time
import imutils
import cv2 as cv
import tensorflow as tf

# Constant variable
ENTER = 13
SIZE = 128
DECIMAL = 2
THICKNESS = 2
TEXT_SCALE = 1.5
RESIZE_SCALE = 1
THRESHOLD = 0.85
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
FONT = cv.FONT_HERSHEY_PLAIN

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load labels and model
labelLines = [line.rstrip() for line in tf.gfile.GFile('./labels.txt')]
with tf.gfile.FastGFile('./model.pb', 'rb') as f:
    graphDef = tf.GraphDef()
    graphDef.ParseFromString(f.read())
    _ = tf.import_graph_def(graphDef, name='')

# User help
def printUsage(arg0):
    print('Usage: python %s <path_to_image> <resize_scale> <sliding_window_size> <threshold>' % (arg0))
    print('Where:')
    print('\t<path_to_image> is the path to the image')
    print('\t<resize_scale> must be a value between 1 and 2 (exclusively)')
    print('\t<sliding_window_size> must be greater than 0')
    print('\t<threshold> must be a value between 0 and 1')

# Slide a window across the image
def slidingWindow(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			yield(x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# Classify a given image
def classifyWindow(window):
    # Get image data
    cv.imwrite('temp.jpg', window)
    imageData = tf.gfile.FastGFile('./temp.jpg', 'rb').read()

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
    return { 'label': largestLabel, 'score': largestScore }

# Check CMD arguments
if len(sys.argv) != 5:
    print('ERROR: Invalid number of arguments!')
    printUsage(sys.argv[0])
    exit()

# Load the image
imagePath = sys.argv[1]
img = cv.imread(imagePath)

# Check if the image really exists
if img is None:
    print('ERROR: The image \'%s\' was not found!' % (sys.argv[1]))
    printUsage(sys.argv[0])
    exit()

# Set user's chosen settings
RESIZE_SCALE = float(sys.argv[2])
SIZE = int(sys.argv[3])
THRESHOLD = float(sys.argv[4])

# Check if settings are valid
if SIZE < 0:
    print('ERROR: The sliding window size must greater than 0!')
    printUsage(sys.argv[0])
    exit()
elif THRESHOLD < 0 or THRESHOLD > 1:
    print('ERROR: The threshold must be a value between 0 and 1!')
    printUsage(sys.argv[0])
    exit()
elif RESIZE_SCALE < 1 or RESIZE_SCALE > 2:
    print('ERROR: The resize scale must be a value between 1 and 2 (exclusively)!')
    printUsage(sys.argv[0])
    exit()
elif RESIZE_SCALE == 1 or RESIZE_SCALE == 2:
    print('ERROR: The resize scale must be a value between 1 and 2 (exclusively)!')
    printUsage(sys.argv[0])
    exit()

# Set the window size
windowWidth, windowHeight = SIZE, SIZE
windowSize = (windowWidth, windowHeight)

# Loop over each sliding window of the image
imgClone = img.copy()
for (x, y, window) in slidingWindow(imgClone, int(SIZE / 2), windowSize):
	# Get window's dimensions
	winW, winH = windowSize

	# Check window's dimensions
	if window.shape[0] != winH or window.shape[1] != winW:
		continue

	# Draw current rectangle
	clone = img.copy()
	cv.rectangle(clone, (x, y), (x + winW, y + winH), GREEN_COLOR, 2)
	cv.imshow("Sliding Window", clone)
	cv.waitKey(1)
	
	# Check for ENTER key input
	if cv.waitKey(1) == ENTER:
		exit()

	# Classify window
	result = classifyWindow(window)

	# Check if result is enough
	if result['score'] > THRESHOLD:
		label = result['label'].title() + ' - ' + str(round(result['score'] * 100, DECIMAL)) + ' %'
		cv.rectangle(img, (x, y), (x + winW, y + winH), RED_COLOR, THICKNESS)
		cv.putText(img, label, (x, y + winH + 25), FONT, TEXT_SCALE, RED_COLOR, THICKNESS)

# Delete temporary sliding window image
os.remove('./temp.jpg')

# Display image
cv.imshow("Sliding Window", img)

# Wait for any key press
cv.waitKey(0)
cv.destroyAllWindows()
