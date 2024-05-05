This script utilizes a train model and its classes
to predict multiple classes of features present in an image.

Requirements:
It is necessary to have a model named "model.pb" and
its classes file named "labels.txt" in the same directory
of this script to correctly execute.

Usage of this script:
# python Multiple.py <path_to_image> <sliding_window_step> <sliding_window_size> <minimum_confidence>

Example:
# python Multiple.py ./test1.png 1.9 150 0.97
# python Multiple.py ./test2.png 1.5 180 0.82