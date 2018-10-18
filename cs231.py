#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 23:55:29 2018

@author: shreeyash
"""

import pandas as pd 
import numpy as np
import cv2
import pickle
#import face_recognition
import matplotlib.pyplot as plt
%matplotlib inline



df = pd.read_csv('/home/shreeyash/Desktop/ML/faces/fer2013/fer2013.csv')


lst = []
lst.append([np.array(l.split(),dtype='int') for l in df.pixels[:33000]])
arr=np.array(lst)
X_train_prev = arr.reshape((33000,48,48))


lst = []
lst.append([np.array(l.split(),dtype='int') for l in df.pixels[33000:]])
arr=np.array(lst)
X_test_prev = arr.reshape((2887,48,48))


X_train = np.array(np.reshape(X_train_prev,(33000,48,48,1)))
X_test = np.array(np.reshape(X_test_prev,(2887,48,48,1)))

from keras.utils import to_categorical
emotion = np.array(df.emotion,dtype='int')
emotion_dict = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
Y_train = to_categorical(emotion[:33000],num_classes=7)
Y_test = to_categorical(emotion[33000:],num_classes=7)

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
model.load_weights('model_4layer_2_2_pool.h5')

