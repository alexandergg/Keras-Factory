import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from kerasfactory.imagenet_utils import decode_predictions
from keras.models import model_from_json, load_model
import os
import cv2
import imutils

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
image_path = os.path.abspath(os.path.join(parent_path, os.pardir))
img_path = image_path+"/Keras_Factory/source/cobre.jpg"

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model = load_model('./models/model_network.h5')
prediction = model.predict(x)[0]
idxs = np.argsort(prediction)[::-1]   #[:2] Coger las dos primeras predicciones

classes = ['Cobre', 'Inox', 'Laton', '']

im = cv2.imread(img_path)

for (i, j) in enumerate(idxs):
	label = "{}: {:.2f}%".format(classes[j], prediction[j] * 100)
	cv2.putText(im, label, (10, (i * 30) + 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
for (label, p) in zip(classes, prediction):
	print("{}: {:.2f}%".format(label, p * 100))
 
cv2.imshow("Output", im)
cv2.waitKey(0)