import time
import cv2
from PIL import Image
from skimage.feature import hog
import helpers
import numpy as np

def sliding_window(img, stepSize, windowSize):
    ''' 
    slide window across the image 
    args:
        img       (2d array): 2d image
        stepSize    (int): number of pixels to skip per new window
        windowSize  (int): width and height of window in pixels
    returns:
        int: 
    ''' 
    dims = img.shape
    for y in range(0, dims[0], stepSize):
       for x in range(0, dims[1], stepSize): 
           # yield current window
           yield(x, y, img[y : y+windowSize[1], x : x+windowSize[0]])


def svm_localize_and_classify(image, clf, clf_type='svm', probLim=0.8, stepSize=5, winW=20, winH=20, draw_window=False):
    ''' 
    Print letter classifications and plot their location
    args:
        img         (2d array): image we want to detect and classify
        clf         (classifier): trained ML classifier
        clf_type    (string): 'svm' or 'cnn'
        probLim     (float): model's confidence needs to surpass this limit
        stepSize    (int): stepsize of window, e.g. slide window+stepsize
        winW        (int): width of image
        winH        (int): height of image
        draw_window (bool): if you want to see the sliding window
    returns:
        image with windows with supposedly letters in them 
    '''

    print(f"> Initiating sliding window with for {clf_type} with:\n confidence threshold: {probLim*100}%\n step size: {stepSize}\n window dimension: {winW}x{winH}")

    predictions = []
    display_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    color = (0, 0, 255) # use red as window default, only turn green when detect letter

    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(image, stepSize=stepSize, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        if clf_type == 'svm':
            feature_descriptor, window_hog = hog(window, orientations=9, pixels_per_cell=(4,4), cells_per_block=(1,1), visualise=True)
            window_processed = [window_hog.flatten()/255.0]
            probabilities = clf.predict_proba(window_processed)
        else:
            # Get the accuracies
            window_cnnshape = window.reshape(1, 20, 20, 1)/255.0
            probabilities = clf.predict(window_cnnshape) 
    
        for i, prob in enumerate(probabilities[0]):
            if prob > probLim:
                print(prob)
                predictions.append((window, helpers.numToChar(i), prob))
                color = (0, 255, 0)
                cv2.rectangle(display_image, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            else:
                color = (0, 0, 255)

        # Draw the window
        if draw_window:
            clone = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
            cv2.rectangle(clone*255.0, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)

    print(f"precticted the letters: {[x[1] for x in predictions]}")
    #cv2.imshow("Prediction", display_image)
    #cv2.waitKey(0)

    return display_image


def cnn_slidingWindow(image):
    from itertools import islice

    img_1d = np.asarray(image.flatten())

    def window(seq, n):
        it = iter(seq)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result
            
    slides = []

    # Remove images that only have white pixels
    for w in window(img_1d, 400):
        count_white = w.count(255)
        if count_white < 400:
            slides.append(np.array(w))

    return np.array(slides)