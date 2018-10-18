# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import cv2
import pickle
import face_recognition
import matplotlib.pyplot as plt
%matplotlib inline



df = pd.read_csv('/home/shreeyash/Desktop/ML/faces/training/fer2013.csv')


lst = []
lst.append([np.array(l.split(),dtype='int') for l in df.pixels[:33000]])
arr=np.array(lst)
X_train_prev = arr.reshape((33000,48,48))


lst = []
lst.append([np.array(l.split(),dtype='int') for l in df.pixels[33000:]])
arr=np.array(lst)
X_test_prev = arr.reshape((2887,48,48))



X_train = np.zeros((33000,128))
for i,img in enumerate(X_train_prev):
    cv2.imwrite('temp.jpg',img)
    img = face_recognition.load_image_file('temp.jpg')
    box = face_recognition.face_locations(img)
    e = face_recognition.face_encodings(img,box)
    if len(e)>0:
        X_train[i] = e[0]
np.save('/home/shreeyash/Desktop/ML/faces/training/encodings/X_train_encodings.npy',X_train)
    
X_test = np.zeros((2887,128))
for i,img in enumerate(X_test_prev):
    cv2.imwrite('temp.jpg',img)
    img = face_recognition.load_image_file('temp.jpg')
    box = face_recognition.face_locations(img)
    e = face_recognition.face_encodings(img,box)
    if len(e)>0:
        X_test[i] = e[0]
np.save('/home/shreeyash/Desktop/ML/faces/training/encodings/X_test_encodings.npy',X_test)  
