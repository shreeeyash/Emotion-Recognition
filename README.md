# Emotion-Recognition
This is a deep learning project for facial emotion recognition in real time. I have used [fer2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset for classification.<p>
There are three major parts:</br>
  (1) Preprocessing</br>
  (2) Training CNN model</br>
  (3) Runtime execution</br>
  
## Preprocessing
First part is preprocessing of data in fer2013 dataset. There are about 30,000 labeled images for 7 different emotions.
Image data is given in a (1,48x48) row in .csv file which should be converted to (48,48,1) so that it can be fed to keras model.
Preprocessing includes converting data to image grid form and horizontal flipping for data augmentation.

## Training
I trained my [1st model](https://github.com/Shreeyash-iitr/Emotion-Recognition/blob/master/trained%20models/1st_model.h5) for all 7 emotions. Model extracted [Facenet](https://arxiv.org/abs/1503.03832) 128 feature vector of each face image and then 4 Dense layers over these feature vector learns to classify different emotions: Happy, sad, angry, fear, surprise, disgust, neutral.<p>
  [2nd]() and [3rd]() models are trained only for a set of emotions: happy, neutral, angry and surprise.

  
  
