import os
import matplotlib.pyplot as plt
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

# ------------------------ CNN ------------------------

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


def plotPredictions(image, true, pred, accuracy, model, original_width=20, original_height=20, zero_one_interval=True):
     ''' Plot 3x3 grid of Chars74K images (image) with the models predictions (pred) '''
     fig, ax = plt.subplots(3, 3)
     # axes are in a two-dimensional array, indexed by [row, col]
     dim = 3
     for i in range(dim):
          for j in range(dim):
               idx = i*dim+j
               ax[i, j].axis('off')
               if zero_one_interval:
                    image[idx] = np.vectorize(lambda p: p*255)(image[idx])
               ax[i, j].imshow( np.reshape(image[idx], (original_width, original_height)), cmap='gray')
               ax[i, j].set_title( f'true: {numToChar(true[idx])} pred: {numToChar(pred[idx])}' )
     fig.suptitle(f'{model} with {np.around(accuracy*100, decimals=2)}% accuracy')
     plt.gray()
     plt.show()

# ------------------------ For SVM ------------------------

from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def displayImage(image, original_width=20, original_height=20, zero_one_interval=True):
    '''Function for displaying the image, either with normal or 0-1 scale '''
    if zero_one_interval:
        image = np.vectorize(lambda p: p*255)(image)
    plt.matshow(np.reshape(image, (original_width, original_height)))
    plt.gray()
    plt.show()

def SVM_preProcessing(X):
    ''' Use the Histogram of Oriented Gradient (HOG) feature descriptor and normalize data '''
    X_out = []
    for img in X:
        # Shape back to 2d image
        img_2d = np.reshape(img, (20, 20))
        #img_filtered = cv2.medianBlur(img_2d,5) # median blur?
        # get HoG
        feature_descriptor, img_hog = hog(img_2d, orientations=9, pixels_per_cell=(4,4), cells_per_block=(1,1), visualise=True)
        # Flatten, normalize and add to return list
        X_out.append(img_hog.flatten()/255.0)
    return X_out

def SVM_getModel(X_train,  y_train):
    ''' Cross Validation With Parameter Tuning Using Grid Search and return best model '''
    pipeline = Pipeline([
            ('clf', SVC(kernel='rbf', gamma=0.01, C=100))
    ])
    parameters = {
            'clf__gamma': (0.1, 0.25, 0.5, 75, 1),
            'clf__C': (1, 1.33, 1.66, 2, 2.5, 3),
    }   
    # Create a classifier object with the classifier and parameter candidates
    #clf_gs = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=3, scoring='accuracy')
    # Create quick classifier witht he optimal hyperparameters
    clf_gs = SVC(kernel='rbf', C=3, gamma=1) # These two are only used if
    clf_gs.fit(X_train, y_train)             # optimal hyperparameters are known
    return clf_gs

def getAccuracy(pred_val, true_val):
    correct = 0
    N = len(true_val)
    for i in range(N):
        if true_val[i] == pred_val[i]:
            correct += 1
    return correct/N