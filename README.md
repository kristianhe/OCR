# Optical Character Recognition
## TDT4173 Assignment 5

**Objective:** 	Gain experience with how to set up a pipeline for a non-trivial machine learning problem. Optical Character Recognition (OCR) will be implemented with at least two different machine learning methods to compare performance.

## File description:
*helpers*: 	Contains functions to prepare the dataset\
*cnn*: 		  Implementation of a convolutional neural network to classify datasets\
*smv*:      


## Required packages:
**python3**\
`sudo apt-get install python3-pip python3-dev`

**numpy**\
`sudo apt-get python-numpy`

**OpenCV**\
`sudo apt-get install python-opencv`

**Pillow** *(Formerly PIL)*\
`sudo pip3 install Pillow` 

**TensorFlow**\
`sudo pip3 install tensorflow`

**Keras**\
`sudo pip3 install keras`

## To run:
for each of the methods, input "--image 'path'" to the image you want to try out the sliding window detection on.
"-W ignore" is to ignore warnings, some of the dependencies of the HOG seems to be deprecated.
\
`python -W ignore cnn.py --image detection-images/detection-1.jpg`\
or \
`python -W ignore svm.py --image detection-images/detection-1.jpg`
