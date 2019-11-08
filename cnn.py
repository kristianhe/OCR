import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


import helpers

# Find paths and labels
dir_dataset = helpers.findDataset()
filenames = helpers.loadFilenames(dir_dataset)
labels = helpers.loadLabels(dir_dataset)

# Prepare the data set
print("-- Preparing data ...")
X, y = helpers.flattenImages(filenames, labels)
print("-- Data preparation finished.")

# Split into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.67)

# Training parameters
imageRows, imageCols = 20, 20
num_classes = 26  # Size of the alphabet
batch_size = 64
epochs = 20

from keras import backend as K
# Check for correct input data format convention ('channels_first'/'channels_last')
print("-- Images being tranformed to an appropriate shape ...")

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, imageRows, imageCols)
    x_test = x_test.reshape(x_test.shape[0], 1, imageRows, imageCols)
    input_shape = (1, imageRows, imageCols)
else:
    x_train = x_train.reshape(x_train.shape[0], imageRows, imageCols, 1)
    x_test = x_test.reshape(x_test.shape[0], imageRows, imageCols, 1)
    input_shape = (imageRows, imageCols, 1)

print('> Training data shape:', x_train.shape)
print("> Number of samples:", len(x_train), 'train and', len(x_test), 'test')

import keras
# Convert label vectors to binary class matrices (based on the alphabet)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten

# (source: https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c)
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

# Create train and test generators to augment the dataset (for variety)
trainGen, testGen = createDataGenerators(x_train, x_test, y_train, y_test)
print("> Data Generators created.")

# Construct and train the convolutional neural network with Keras
print("-- Initializing training ...")
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(
     loss=keras.losses.categorical_crossentropy,
     optimizer=keras.optimizers.Adadelta(),
     metrics=['accuracy'])
model.fit_generator(
     generator=trainGen,
     steps_per_epoch=10000 // batch_size,
     epochs=epochs,
     verbose=1,
     validation_data=testGen,
     validation_steps=6000 // batch_size)

# Calculate loss and accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print("-- Model has been trained.")
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# RESULTS: LOSS= 113.165, ACC=0.826