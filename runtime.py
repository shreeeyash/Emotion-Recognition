
import keras 
from keras.models import load_model
import cv2
import dlib
import face_recognition
import imutils
import time
import numpy as np

model = load_model('/home/shreeyash/Desktop/ML/faces/training/trained models/3rd_model.h5')
#emotion_dict = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
#emotion_dict = {0:'Angry',1:'Happy',2:'Neutral',3:'Sad'}
emotion_dict = {0:'Angry',1:'Happy',2:'Neutral',3:'Surprise'}

capture = cv2.VideoCapture(0)
no_frames = 0
while True:
    no_frames += 1
    ret, frame = capture.read()

    if no_frames == 5:
        box = face_recognition.face_locations(frame)
        e = face_recognition.face_encodings(frame, box)
        if len(e)>0:
            encode = np.reshape(e, (-1, 128))
            prediction = model.predict(encode)
            label = emotion_dict[np.argmax(prediction[0])]
            print(label,np.max(prediction[0]))
        no_frames=0
    print('labels are working')	
    cv2.imshow('frames',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
capture.release()
cv2.destroyAllWindows()            






