#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 17:18:08 2018

@author: shreeyash
"""

import pandas as pd 
import numpy as np
import cv2
import pickle
import pandas as pd
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
%matplotlib inline


import keras
from keras.models import Model,load_model
from keras.layers import Dense, Input,BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
df = pd.read_csv('/home/shreeyash/Desktop/ML/faces/fer2013/fer2013.csv')

diction = {0:'Angry',1:'Happy',2:'Neutral',3:'Surprise'}
#X_h_n_sad_ang = np.concatenate((happy,neutral,surprise,angry),0)

X_train = np.load('/home/shreeyash/Desktop/ML/faces/training/encodings/X_train_encodings_new2.npy')
Y_train = np.load('/home/shreeyash/Desktop/ML/faces/training/encodings/Y_save.npy')
Y_train = to_categorical(Y_train,num_classes=4)
#print(Y_train.shape)

inputs = Input(shape=(128,))  
X = Dense(units = 1024,activation='relu')(inputs)
X = Dense(units = 1024,activation='relu')(X)  
X = Dense(units = 1024,activation='relu')(X)
Y = Dense(units = 4,activation='softmax')(X)

model = Model(inputs = inputs, outputs = Y)

adam = Adam(lr=0.001,decay=0.3)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics = ['accuracy'])
history = model.fit(x=X_train, y=Y_train,epochs = 5, validation_split=0.05,verbose=1)

plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_acc'])

model.save('/home/shreeyash/Desktop/ML/faces/training/trained models/3rd_model.h5')
