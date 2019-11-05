import os
import numpy as np
import pandas as pd
import cv2

dir_dataset = os.path.dirname(os.path.realpath(__file__)) + "/chars74k-lite"

alphabet = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,
            'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18,'t':19,
            'u':20,'w':22,'x':23,'y':24,'z':25}

labelNames = np.arange(26)

def loadFilenames(dir):
     filenames = []
     for path, _, files in os.walk(dir):
          for file in files:
               if file.endswith('.jpg'):
                    file = path + '/' + file
                    filenames.append(file)
     return filenames

def loadLabels(dir):
     labels = []
     for path, _, files in os.walk(dir):
               for file in files:
                    if file.endswith('.jpg'):
                         label = path[-1:]
                         labels.append(label)
     return labels

filenames = loadFilenames(dir_dataset)
labels = loadLabels(dir_dataset)

