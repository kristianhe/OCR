from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from PIL import Image
import cv2
import helpers
from sliding_window import localize_and_classify

# Find paths and labels
dir_dataset = helpers.getPath()
filenames = helpers.getFilenames(dir_dataset)
labels = helpers.getLabels(dir_dataset)

# Flatten the data set
print("> ------ CNN classifier ------")
print("> Collecting Image data ...")
X, y = helpers.flattenImages(filenames, labels)
num_classes = 26
num_epochs = 20

print("> Splitting train and test data ...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=40)

# Reshaping the array to 4-dims so that it can work with the Keras API
X_train = X_train.reshape(X_train.shape[0], 20, 20, 1)
X_test = X_test.reshape(X_test.shape[0], 20, 20, 1)
input_shape = (20, 20, 1)
# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255.0
X_test /= 255.0


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation=tf.nn.relu, input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3,3), activation=tf.nn.relu))
model.add(Conv2D(128, kernel_size=(3,3), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation=tf.nn.softmax))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=X_train,y=y_train, epochs=num_epochs)

# Get the accuracies
y_pred = []
for pred in model.predict(X_test): 
    y_pred.append(np.argmax(pred))

score = model.evaluate(X_test, y_test, verbose=0)
print("-- Model has been trained.")
print("> Test loss:", score[0])
print("> Test accuracy:", score[1])
accuracy = helpers.getAccuracy(y_pred, y_test)
print(f"> The accuracy of the model is {accuracy}")

#helpers.plotPredictions(X_test[0:9], y_test[0:9], y_pred[0:9], accuracy, 'CNN')


# ------------------ Detect test image -----------------------

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
image = np.asarray(Image.open(args["image"]))

ret_img = localize_and_classify(image, model, 'cnn', probLim=0.99999, stepSize=2, winW=20, winH=20)
cv2.imwrite('images/svm_pred5.png', ret_img)
