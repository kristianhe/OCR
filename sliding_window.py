import argparse
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

def sliding_window(img, stepSize, windowSize):
    ''' slide window across the image 
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
           yield(x, y, img[y:y + windowSize[1], x:x + windowSize[0]])

(winW, winH) = (20, 20)

# loop over the sliding window for each layer of the pyramid
for (x, y, window) in sliding_window(image, stepSize=10, windowSize=(winW, winH)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != winH or window.shape[1] != winW:
        continue

    # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
    # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
    # WINDOW
    
    print(window)
    pred = svc_clf.predict(window)
    prob = svc_clf.predict_proba(window)

    print(f"Predicted {pred} with {prob*100}% probability")

    # since we do not have a classifier, we'll just draw the window
    clone = image.copy()
    cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
    cv2.imshow("Window", clone)
    cv2.waitKey(1)
    time.sleep(0.025)

