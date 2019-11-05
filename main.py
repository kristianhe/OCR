import os
import numpy as np
import pandas as pd
import cv2

dir_dataset = os.path.dirname(os.path.realpath(__file__)) + "/chars74k-lite"

# Find all filenames in the target directory
def loadFilenames(target_dir):
     filenames = []
     for path, _, files in os.walk(target_dir):
          for file in files:
               if file.endswith('.jpg'):
                    file = path + '/' + file
                    filenames.append(file)
     return filenames

# Find all labels in the target directory
def loadLabels(dir):
     labels = []
     for path, _, files in os.walk(dir):
               for file in files:
                    if file.endswith('.jpg'):
                         label = path[-1:]
                         labels.append(label)
     return labels


# Load filenames and labels
filenames = loadFilenames(dir_dataset)
labels = loadLabels(dir_dataset)
numericalLabels = np.arange(26)

# Display the data
print("The number of images is: " + str(len(filenames)))
img = cv2.imread(filenames[0]); img2 = cv2.imread(filenames[5000]); img3 = cv2.imread(filenames[6400])
lbl = labels[0]; lbl2 = labels[5000]; lbl3 = labels[6400]
print("Example image with corresponding pixel intensities within the normalised range (8-bit): \n", img)
cv2.imshow("Image with label '" + lbl + "'", img); cv2.imshow("Image with label '" + lbl2 + "'", img2); cv2.imshow("Image with label '" + lbl3 + "'", img3)
cv2.waitKey(0)