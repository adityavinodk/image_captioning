#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib
matplotlib.use("Agg")
import json
import os
os.environ["CUDA_DEVICE_ORDER"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from cnn_model import ImageClassificationModel

from PIL import Image
# import matplotlib.pyplot as plt
# import pickle as pkl
import dask.array as da
import numpy as np


# In[3]:


# with open('train_annotations.json', 'r') as file:
#     train_annotations = json.load(file)

# # with open('validation_annotations.json', 'r') as file:
# #     validation_annotations = json.load(file)
# cb = Callback()
# from tensorflow.python.client import device_lib
# with tf.device('/gpu:0'):
#     print(device_lib.list_local_devices())
# print(tf.test.gpu_device_name())
# del train_annotations
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
sess.list_devices()


# In[3]:


# train_annotations['COCO_train2014_000000000025.jpg']


# In[4]:


target_size = (128,128)
train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = len(os.listdir(train_data_dir))
nb_validation_samples = len(os.listdir(validation_data_dir))
epochs = 50
batch_size = 60


# In[5]:


# with open('annotations/instances_train2014.json','r') as file:
#     instances = json.load(file)

# labels = np.array([category['name'] for category in instances['categories']])
# labels = da.from_array(labels, chunks = (20))
# labels.to_zarr('labels.zarr')
labels = da.from_zarr('labels.zarr')
labels
# del instances


# In[6]:


# imagePaths = list(paths.list_images('train'))
# shape = (len(imagePaths),)+target_size+(3,)
# train_x = np.zeros(shape=shape, dtype=np.float16)
# train_y = []
# for i in tqdm(range(len(imagePaths))):
#     imagePath = imagePaths[i]
#     image = Image.open(imagePath)
#     image = image.resize(size=target_size, resample=Image.LANCZOS)
#     image = np.array(image)
#     if (len(image.shape) == 2):
#         image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
#     # print(image.shape)
#     train_x[i]=image
#     labels = set()
#     for category in train_annotations[imagePath.split(os.path.sep)[-1]]['categories']: labels.add(category)
#     train_y.append(list(labels))


# In[7]:


with tf.device('/cpu:0'):
#     train_x = np.load('train_x_128.npy', allow_pickle=True).astype('uint8')
#     train_x = train_x.astype(int)
#     train_x = train_x.astype('uint8')
#     train_x = da.from_array(train_x, chunks = (827,128,128,3))
#     train_x.to_zarr("train_x_128_dask.zarr")
    train_x = da.from_zarr("train_x_128_dask.zarr")
#     train_y = np.load('train_y.npy', allow_pickle=True)
#     train_y = np.array(train_y)
#     mlb = MultiLabelBinarizer(labels)
#     train_y = mlb.fit_transform(train_y)
    train_y = da.from_zarr("train_y_dask.zarr")


# In[8]:


# imagePaths = list(paths.list_images('validation'))
# shape = (len(imagePaths),)+target_size+(3,)
# validation_x = np.zeros(shape=shape, dtype=np.float16)
# validation_y = []
# for i in tqdm(range(len(imagePaths))):
#     imagePath = imagePaths[i]
#     image = Image.open(imagePath)
#     image = image.resize(size=target_size, resample=Image.LANCZOS)
#     image = np.array(image)
#     if (len(image.shape) == 2):
#         image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
#     # print(image.shape)
#     validation_x[i]=image
#     labels = set()
#     for category in validation_annotations[imagePaths[i].split(os.path.sep)[-1]]['categories']: labels.add(category)
#     validation_y.append(list(labels))
# train_x.to_zarr("train_x_244_dask.zarr")


# In[9]:


with tf.device('/cpu:0'):
#     validation_x = np.load('validation_x_128.npy', allow_pickle=True)#).astype(int)).astype('uint8')
#     validation_x = da.from_array(validation_x, chunks = (400,128,128,3))
#     validation_x.to_zarr('validation_x_128_dask.zarr')
#     validation_y = np.load('validation_y.npy', allow_pickle=True)
#     validation_y = np.array(validation_y)
#     mlb = MultiLabelBinarizer(labels)
#     validation_y = mlb.fit_transform(validation_y)
    validation_x = da.from_zarr("validation_x_128_dask.zarr")
    validation_y = da.from_zarr("validation_y_dask.zarr")


# In[10]:


train_x


# In[11]:


train_datagen = ImageDataGenerator(
        rotation_range=25, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        shear_range=0.2,
        fill_mode='nearest',
        zoom_range=0.2,
        horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

model = ImageClassificationModel.build(target_size[0], target_size[1], len(labels), 'softmax')
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.95), metrics=['accuracy'])


# In[12]:


with tf.device('/gpu:0'):
    stop_early = EarlyStopping(monitor='val_loss', patience=20)
    reduceLR = ReduceLROnPlateau(monitor='val_loss', paitence = 20, factor=0.2, min_lr = 0.0001)
    ModelCheck    =ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    callbacks = [stop_early, reduceLR, ModelCheck]
    model.fit_generator(
        train_datagen.flow(train_x, train_y, batch_size= batch_size),
        steps_per_epoch = nb_train_samples//batch_size,
        epochs = epochs,
        validation_data=validation_datagen.flow(validation_x, validation_y, batch_size=batch_size),
        validation_steps = nb_validation_samples//batch_size,
        callbacks = callbacks)


# In[ ]:


model.save("best_model.h5")
f = open('mlb.pickle', 'wb')
f.write(pkl.dumps(mlb))
f.close()


# In[2]:


model = load_model('best_model.h5')
# type(model_1)


# In[15]:


img = Image.open('test/COCO_train2014_000000543689.jpg')# image extension *.png,*.jpg
img.show()
img = img.resize((128,128), Image.ANTIALIAS)
arr = np.array(img)
arr=np.expand_dims(arr,0)
arr=arr/255
proba = model.predict(arr)[0]
proba= (proba*1000).astype(int)
for x in range(len(proba)):
    if proba[x] >100:
        proba[x]=1
    elif proba[x] > 50:
        proba[x] = 0.5
    else:
        proba[x]=0
for i in range(len(proba)):
    if(proba[i]==1):
        print(labels[i])
    elif proba[i]==0.5:
        print("Probably -> ", labels[i])


# In[ ]:




