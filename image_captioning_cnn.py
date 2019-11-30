import json
import os
os.environ["CUDA_DEVICE_ORDER"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras
import tensorflow as tf
import numpy as np
from PIL import Image
import dask.array as da

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import load_model
from cnn_model import ImageClassificationModel

if 'labels.npy' not in os.listdir():
    with open('annotations/instances_train2014.json','r') as file:
        instances = json.load(file)

    labelsList = np.array([category['name'] for category in instances['categories']])
    np.save('labels.npy', labelsList)
    del instances

labelsList = np.load('labels.npy',allow_pickle=True)

if "train_x_128_dask.zarr" in os.listdir('.') and "train_y_dask.zarr" in os.listdir('.'):
    with tf.device('/cpu:0'):
        train_x = da.from_zarr("train_x_128_dask.zarr")
        train_y = da.from_zarr("train_y_dask.zarr")
else: 
    print("Train dask arrays haven't been saved, please run extract_labels.py")
    exit()

if "validation_x_128_dask.zarr" in os.listdir('.') and "validation_y_dask.zarr" in os.listdir('.'):
    with tf.device('/cpu:0'):
        validation_x = da.from_zarr("validation_x_128_dask.zarr")
        validation_y = da.from_zarr("validation_y_dask.zarr")
else: 
    print("Train dask arrays haven't been saved, please run extract_labels.py")
    exit()

target_size = (128,128)
train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = len(os.listdir(train_data_dir))
nb_validation_samples = len(os.listdir(validation_data_dir))
epochs = 50
batch_size = 60

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

if 'best_model.h5' in os.listdir() and 'best_ modelw.h5' in os.listdir():
    model = load_model('best_model.h5')
else:
    model = ImageClassificationModel.build(target_size[0], target_size[1], len(labelsList), 'softmax')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.9, beta_2=0.95), metrics=['accuracy'])

print(model.summary())

with tf.device('/gpu:0'):
    stop_early = EarlyStopping(monitor='val_loss',patience = 20)
    reduceLR = ReduceLROnPlateau(monitor='val_loss',paitence = 20, factor=0.2, min_lr = 0.0001)
    ModelCheck = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    callbacks = [stop_early, reduceLR, ModelCheck]
    model.fit_generator(
        train_datagen.flow(train_x, train_y, batch_size= batch_size),
        steps_per_epoch = nb_train_samples//batch_size,
        epochs = epochs,
        validation_data=validation_datagen.flow(validation_x, validation_y, batch_size=batch_size),
        validation_steps = nb_validation_samples//batch_size,
        callbacks = callbacks)

# testModel = load_model("best_model.h5")

# img = Image.open('test/COCO_train2014_000000543689.jpg')# image extension *.png,*.jpg
# img = img.resize((128,128), Image.ANTIALIAS)
# arr = np.array(img)
# arr=np.expand_dims(arr,0)
# arr=arr/255
# proba = testModel.predict(arr)[0]
# # print(proba)
# proba= (proba*1000).astype(int)
# for x in range(len(proba)):
#     if proba[x] >100:
#         proba[x]=1
#     elif proba[x] > 50:
#         proba[x] = 0.5
#     else:
#         proba[x]=0

# print('List of classes present')
# for i in range(len(proba)):
#     if(proba[i]==1):
#         print(labels[i])
#     elif proba[i]==0.5:
#         print("Probably -> ", labels[i])