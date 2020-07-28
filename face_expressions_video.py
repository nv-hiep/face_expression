# -*- coding: utf-8 -*-
"""Face_expressions_video.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1N3A8sM9ZmHGATbfelqBMZmpJ22S3fk3F

# **Import**
"""

import cv2
import numpy      as np
import tensorflow as tf
from tensorflow          import keras
from keras.preprocessing import image

import matplotlib.pyplot as plt

# Define variables
num_classes = 7
classes     = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
batch_size  = 256
epochs      = 10

from google.colab import drive
drive.mount('/content/drive')
# with open('/content/drive/My Drive/glove.6B.50d.txt', encoding='utf-8') as f:
# Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

# opencv initialization
face_cascade = cv2.CascadeClassifier('/content/drive/My Drive/data/haarcascade_frontalface_default.xml')

# For test video
cap = cv2.VideoCapture('/content/drive/My Drive/data/Basic_Emotions_Test.webm')

# For webcam
# cap = cv2.VideoCapture(-1) on linux
# cap = cv2.VideoCapture(0) on windows

"""# **Load model**"""

# Load the trained model
model = keras.models.load_model('/content/drive/My Drive/data/facial_expression_model_weights.h5')

"""# **Make prediction**"""

# For drawing bar chart
def emotion_analysis(emotions):
    y_pos = np.arange(len(classes))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, classes)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()

# prediction for an image

img = image.load_img("/content/drive/My Drive/data/happy.jpeg", grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.gray()
plt.imshow(x)
plt.show()

from google.colab.patches import cv2_imshow

while(True):
  ret, img = cap.read()
  if ret == False:
    break

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = face_cascade.detectMultiScale(gray, 1.3, 5)

  for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)                              # draw rectangle to main image

    detected_face = img[int(y):int(y+h), int(x):int(x+w)]                       # crop detected face
    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)             # to gray scale
    detected_face = cv2.resize(detected_face, (48, 48))                         # to 48x48

    img_pixels = image.img_to_array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis = 0)

    img_pixels /= 255

    predictions = model.predict(img_pixels)                                     # store probabilities of 7 expressions

    # find max indexed array
    # 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
    max_index = np.argmax(predictions[0])

    emotion = classes[max_index]

    # write text above rectangle
    cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

  # end - for

  # cv2.imshow('img',img)
  cv2_imshow(img)

  if cv2.waitKey(1) & 0xFF == ord('q'):                                         # press q to quit
    break

# Close
cap.release()
cv2.destroyAllWindows()