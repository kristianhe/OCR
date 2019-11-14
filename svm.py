from sklearn.model_selection import train_test_split
import helpers
import argparse
from sliding_window import svm_localize_and_classify
from PIL import Image
import numpy as np
import os
import cv2

# Find paths and labels
dir_dataset = helpers.getPath()
filenames = helpers.getFilenames(dir_dataset)
labels = helpers.getLabels(dir_dataset)

# Flatten the data set
print("> ------ SVM classifier ------")
print("> Collecting Image data ...")
X, y = helpers.flattenImages(filenames, labels)

print("> Splitting train and test data ...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print("> Preprocess data ...")
X_train_HOG = helpers.SVM_preProcessing(X_train)#[0:5000])
X_test_HOG = helpers.SVM_preProcessing(X_test)#[0:2000])

print("> Creating a model ...")
# hog  pca = 70, kernel='rbf', C=5, gamma=0.16 leads to good results.
svc_clf = helpers.SVM_getModel(X_train_HOG,  y_train)#[0:6000])

# Train the classifier
print("> Predicting ...")
y_pred = svc_clf.predict(X_test_HOG)  

accuracy = helpers.getAccuracy(y_pred, y_test)#[0:2000])
print(f"> The accuracy of the model is {accuracy}")

#print(f"> Plotting the image {y[0]} ...")
#displayImage(X_train[0])

#helpers.plotPredictions(X_test[0:9], y_test[0:9], y_pred[0:9], accuracy, 'SVM')
#helpers.plotPredictions(svm.X_test[6:14], svm.y_test[6:14], svm.y_pred[6:14], svm.accuracy, 'SVM')


# ------------------ Detect test image -----------------------

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
image = np.asarray(Image.open(args["image"]))


ret_img = svm_localize_and_classify(image, svc_clf, 'svm', probLim=0.88, stepSize=2, winW=20, winH=20)
cv2.imwrite('images/svm_pred2.png',ret_img)

