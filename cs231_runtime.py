#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 00:12:38 2018

@author: shreeyash
"""



import cv2
import dlib
import face_recognition
import imutils
import numpy as np

import keras
from keras.models import load_model
"""
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json

from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

batch_size = 128
epochs = 124

#Main CNN model with four Convolution layer & two fully connected layer
def baseline_model():
    # Initialising the CNN
    model = Sequential()

    # 1 - Convolution
    model.add(Conv2D(64,(3,3), border_mode='same', input_shape=(48, 48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd Convolution layer
    model.add(Conv2D(128,(5,5), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3rd Convolution layer
    model.add(Conv2D(512,(3,3), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 4th Convolution layer
    model.add(Conv2D(512,(3,3), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    # Flattening
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))


    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(7, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


model = baseline_model()
model.load_weights('/home/shreeyash/Desktop/ML/faces/training/model_4layer_2_2_pool.h5')
"""
model = load_model('/home/shreeyash/Desktop/ML/faces/training/trained models/cs231n_model.h5')
emotion_dict = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}

capture = cv2.VideoCapture(0)
no_frames = 0
face_cascade = cv2.CascadeClassifier('/home/shreeyash/opencv-3.4.1/data/haarcascades/haarcascade_frontalcatface.xml')

while True:
    no_frames += 1
    ret, frame = capture.read()

    if no_frames == 5:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for (x,y,w,h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_reshape_48 = cv2.resize(face_img,(48,48))
            face_img_48 = np.reshape(face_reshape_48, (1,48,48,1))
            prediction = model.predict(face_img_48)
            label = emotion_dict[np.argmax(prediction[0])]
            print(label, np.max(prediction[0]))
            
        #prediction = model.predict(face_img_48)
        #label = emotion_dict[np.argmax(prediction[0])]
        
        no_frames=0
    cv2.imshow('frames',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
capture.release()
cv2.destroyAllWindows()            
