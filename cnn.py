import os
import cv2
import numpy as np
import PIL
from PIL import Image

import helpers

# Find paths and labels
dir_dataset = helpers.findDataset()
filenames = helpers.loadFilenames(dir_dataset)
labels = helpers.loadLabels(dir_dataset)

# Prepare the data set
print("> Preparing data ...")
X, y = helpers.flattenImages(filenames, labels)
print("> Data preparation finished.")
