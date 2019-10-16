import os
import json
import numpy as np
import time
from PIL import Image
from imutils import paths
from tqdm import tqdm
from multiprocessing import Manager, Process, Pool

with open('train_annotations.json', 'r') as file:
    train_annotations = json.load(file)

with open('validation_annotations.json', 'r') as file:
    validation_annotations = json.load(file)

target_size = (128,128)

trainImagePaths = list(paths.list_images('train'))
shape = (len(trainImagePaths),)+target_size+(3,)
train_x = np.zeros(shape=shape, dtype=np.float16)
train_y = []
for i in tqdm(range(len(trainImagePaths))):
    imagePath = trainImagePaths[i]
    image = Image.open(imagePath)
    image = image.resize(size=target_size, resample=Image.LANCZOS)
    image = np.array(image)
    if (len(image.shape) == 2):
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    train_x[i]=image
    labels = set()
    for category in train_annotations[imagePath.split(os.path.sep)[-1]]['categories']: labels.add(category)
    train_y.append(list(labels))

train_y = np.array(train_y)
np.save('train_x_128.npy', train_x)
np.save('train_y.npy', train_y)

validationImagePaths = list(paths.list_images('validation'))
shape = (len(validationImagePaths),)+target_size+(3,)
validation_x = np.zeros(shape=shape, dtype=np.float16)
validation_y = []
for i in tqdm(range(len(validationImagePaths))):
    imagePath = validationImagePaths[i]
    image = Image.open(imagePath)
    image = image.resize(size=target_size, resample=Image.LANCZOS)
    image = np.array(image)
    if (len(image.shape) == 2):
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    validation_x[i]=image
    labels = set()
    for category in validation_annotations[imagePath.split(os.path.sep)[-1]]['categories']: labels.add(category)
    validation_y.append(list(labels))

validation_y = np.array(validation_y)
np.save('validation_x_128.npy', validation_x)
np.save('validation_y.npy', validation_y)