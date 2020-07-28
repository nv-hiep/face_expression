import cv2
import numpy      as np
import tensorflow as tf

from tensorflow          import keras
from keras.preprocessing import image

import matplotlib.pyplot as plt


#------------------------------
#function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
    y_pos = np.arange(len(classes))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, classes)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()
#------------------------------


# Load model
model = keras.models.load_model('model/facial_expression_model.h5')


# Define variables
num_classes = 7
classes     = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
batch_size  = 256
epochs      = 10





if(True):
	# prediction for image
	img = image.load_img("img/happy.jpeg", grayscale=True, target_size=(48, 48))

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









cv2.ocl.setUseOpenCL(False)
# emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	if not ret:
		break
	
	bounding_box = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
	gray_frame   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	num_faces    = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
	
	for (x, y, w, h) in num_faces:
		# The area of faces
		cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
		roi_gray_frame     = gray_frame[y:y + h, x:x + w]
		cropped_img        = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
		
		# Prediction
		emotion_prediction = model.predict(cropped_img)
		maxindex           = int(np.argmax(emotion_prediction))
		cv2.putText(frame, classes[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
	
	cv2.imshow('Video', cv2.resize(frame,(1200,860), interpolation = cv2.INTER_CUBIC))
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()
