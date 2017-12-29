This script utilizes a train model and its classes
to evaluate its own performance by classifying 
different images from a test dataset and writing
them into a txt file.

Requirements:
It is necessary to have a model named "model.pb" and
its classes file named "labels.txt" in the same directory
of this script to correctly execute.

Usage of this script:
# python Evaluate.py <path_to_test_dataset> <images_format (jpg, png)> <path_to_labels_of_test_dataset>

Example:
# python Evaluate.py ./dataset jpg ./realLabels.txt