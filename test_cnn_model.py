from tensorflow.keras.models import load_model
from PIL import Image
from imutils import paths
import numpy as np
import json
import time
import cv2

model = load_model('cnn_model.h5')

with open('annotations/instances_train2014.json','r') as file:
    instances = json.load(file)

labels = np.array([category['name'] for category in instances['categories']])
del instances

with open('test_annotations.json','r') as file:
    test_annotations = json.load(file)

file_name = 'test/COCO_train2014_000000527649.jpg'
img = Image.open(file_name)
img = img.resize((128, 128), Image.ANTIALIAS)
img_arr = np.array(img)
img_arr = np.expand_dims(img_arr, 0)
img_arr = img_arr/255
proba = model.predict(img_arr)[0]
proba= (proba*1000).astype(int)
print(proba)
print("Predicted Classes -")
for i in range(len(proba)):
    if proba[i]>50:
        print(labels[i])
print("-------------------------")

actual = [category for category in list(test_annotations[file_name.split('/')[1]]['categories'].keys())]
print("Actual Classes -")
for name in actual:
    print(name)