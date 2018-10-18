#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:09:51 2018

@author: shreeyash
"""

import cv2
import numpy as np
import pickle
import face_recognition
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


df = pd.read_csv('/home/shreeyash/Desktop/ML/faces/training/fer2013.csv')

X_save = np.load('/home/shreeyash/Desktop/ML/faces/training/encodings/X_save.npy')
Y_save = np.load('/home/shreeyash/Desktop/ML/faces/training/encodings/Y_save.npy')


X_train = np.zeros((Y_save.shape[0],128))
for i,img in enumerate(X_save[:,:,:,0]):
    cv2.imwrite('temp.jpg',img)
    img = face_recognition.load_image_file('temp.jpg')
    box = face_recognition.face_locations(img)
    e = face_recognition.face_encodings(img,box)
    if len(e)>0:
        X_train[i] = e[0]
np.save('/home/shreeyash/Desktop/ML/faces/training/encodings/X_train_encodings_new.npy',X_train)
