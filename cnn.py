import os
import cv2
import numpy as np
import PIL
from PIL import Image

import helpers

dir_dataset = os.path.dirname(os.path.realpath(__file__)) + "/chars74k-lite"

filenames = helpers.loadFilenames(dir_dataset)
labels = helpers.loadLabels(dir_dataset)

# Display the data
print("The number of images is: " + str(len(filenames)))
img = cv2.imread(filenames[0]); img2 = cv2.imread(filenames[5000]); img3 = cv2.imread(filenames[6400])
lbl = labels[0]; lbl2 = labels[5000]; lbl3 = labels[6400]
#print("Example image with corresponding pixel intensities within the normalised range (8-bit): \n", img)
#cv2.imshow("Image with label '" + lbl + "'", img); cv2.imshow("Image with label '" + lbl2 + "'", img2); cv2.imshow("Image with label '" + lbl3 + "'", img3)
#cv2.waitKey(0)


print("> Preparing data ...")
X, y = helpers.flattenImages(filenames, labels)
print("> Data preparation finished.")
