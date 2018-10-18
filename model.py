#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 18:28:00 2018

@author: shreeyash
"""

import pandas as pd 
import numpy as np
import cv2
import pickle
import face_recognition
import matplotlib.pyplot as plt
%matplotlib inline


import keras
from keras.models import Model,load_model
from keras.layers import Dense, Input,BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
df = pd.read_csv('/home/shreeyash/Desktop/ML/faces/training/fer2013.csv')


lst = []
lst.append([np.array(l.split(),dtype='int') for l in df.pixels[:33000]])
arr=np.array(lst)
X_train_prev = arr.reshape((33000,48,48))


lst = []
lst.append([np.array(l.split(),dtype='int') for l in df.pixels[33000:]])
arr=np.array(lst)
X_test_prev = arr.reshape((2887,48,48))



emotion = np.array(df.emotion,dtype='int')
emotion_dict = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
Y_train = to_categorical(emotion[:33000],num_classes=7)
Y_test = to_categorical(emotion[33000:],num_classes=7)


X_train = np.load('/home/shreeyash/Desktop/ML/faces/training/encodings/X_train_encodings.npy')
X_test = np.load('/home/shreeyash/Desktop/ML/faces/training/encodings/X_test_encodings.npy')




inputs = Input(shape=(128,))  
X = Dense(units = 1024,activation='relu')(inputs)
X = Dense(units = 1024,activation='relu')(X)
X = BatchNormalization()(X)  
X = Dense(units = 1024,activation='relu')(X)
Y = Dense(units = 7,activation='softmax')(X)

model = Model(inputs = inputs, outputs = Y)
adam = Adam(lr=0.0001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics = ['accuracy'])

#run at last
history = model.fit(x=X_train, y=Y_train,epochs = 10, validation_split=0.1,verbose=1)

#model = load_model('/home/shreeyash/Desktop/ML/faces/1st_model.h5')

loss, accuracy = model.evaluate(x=X_test,y=Y_test)
print('loss = ',loss," and accuracy : ",accuracy)
#at last
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_acc'])

num = 1641
predict = model.predict(X_test[num:num+1])
plt.imshow(X_test_prev[num])
plt.xlabel(emotion_dict[np.argmax(Y_test[num])])
plt.ylabel(emotion_dict[np.argmax(predict[0])])

"""
model.save('/home/shreeyash/Desktop/ML/faces/training/trained models/1st_model.h5')
model.save_weights('/home/shreeyash/Desktop/ML/faces/training/trained models/1st_weights')
"""
