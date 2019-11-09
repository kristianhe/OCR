import os
import numpy as np
import PIL
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

alphabet = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10,
            'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20,
            'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}

# Find path to data set
def getPath():
    return os.path.dirname(os.path.realpath(__file__)) + "/chars74k-lite"

# Find all filenames in the target directory
def getFilenames(target_dir):
     filenames = []
     for path, _, files in os.walk(target_dir):
          for file in files:
               if file.endswith('.jpg'):
                    file = path + '/' + file
                    filenames.append(file)
     return filenames

# Find all labels in the target directory
def getLabels(target_dir):
     labels = []
     for path, _, files in os.walk(target_dir):
               for file in files:
                    if file.endswith('.jpg'):
                         label = path[-1:]
                         labels.append(label)
     return labels

# Function for transforming alphabetical letters to numerical labels
def charToNum(char):
    num = alphabet[char]
    return num

# Function for transforming numerical labels to alphabetical letters
def numToChar(num):
    for char in alphabet:
        if alphabet[char] == num:
            return char

# Prepare the data set
def flattenImages(filenames, labels):
     imageData = []
     labelData = []

     for path in filenames:
          flattenedImageData = np.asarray(Image.open(path)).flatten()
          imageData.append(flattenedImageData)

     for label in labels:
          numericalLabels = charToNum(label)
          labelData.append(numericalLabels)

     return np.array(imageData), np.array(labelData)

# (source: https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c)
# Data Generators to augment the dataset
def createDataGenerators(x_train, x_test, y_train, y_test):
    trainDataGenerator = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=0./180,
        vertical_flip=True)

    testDataGenerator = ImageDataGenerator(
        rescale=1./255)

    trainGenerator = trainDataGenerator.flow(x=x_train, y=y_train)
    testGenerator = testDataGenerator.flow(x=x_test, y=y_test)

    return trainGenerator, testGenerator