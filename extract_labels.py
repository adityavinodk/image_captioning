import os
import json
import numpy as np
import time
from PIL import Image
from imutils import paths
from tqdm import tqdm
import dask.array as da
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
target_size = (128,128)

if 'labels.npy' not in os.listdir():
    with open('annotations/instances_train2014.json','r') as file:
        instances = json.load(file)

    labelsList = np.array([category['name'] for category in instances['categories']])
    np.save('labels.npy', labelsList)
    del instances

labelsList = np.load('labels.npy', allow_pickle=True)
mlb = MultiLabelBinarizer(labelsList)

if 'train_x_128_dask.zarr' not in os.listdir('.') or 'train_y_dask.zarr' not in os.listdir('.'):
    print('Train dask arrays not found, saving now...')
    with open('train_annotations.json', 'r') as file:
        train_annotations = json.load(file)

    print("Storing train Images into Dask Array format...")
    trainImagePaths = list(paths.list_images('train'))
    shape = (len(trainImagePaths),)+target_size+(3,)
    train_x = np.zeros(shape=shape, dtype=np.float16)
    train_y = []
    with tf.device('/gpu:0'):
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

    train_x = train_x.astype('uint8')
    train_x = da.from_array(train_x, chunks = (827,128,128,3))
    train_x.to_zarr("train_x_128_dask.zarr")

    train_y = np.array(train_y)
    train_y = mlb.fit_transform(train_y)
    
    train_y = train_y.astype('uint8')
    train_y = da.from_array(train_y, chunks = (827))
    train_y.to_zarr("train_y_dask.zarr")

    print('Train dask arrays have been successfully stored')
    del train_annotations; del train_x; del train_y; del trainImagePaths

if 'validation_x_128_dask.zarr' not in os.listdir('.') or 'validation_y_dask.zarr' not in os.listdir('.'):
    print('Validation dask arrays not found, saving now...')
    with open('validation_annotations.json', 'r') as file:
        validation_annotations = json.load(file)

    print("Storing validation Images into Dask Array format...")
    validationImagePaths = list(paths.list_images('validation'))
    shape = (len(validationImagePaths),)+target_size+(3,)
    validation_x = np.zeros(shape=shape, dtype=np.float16)
    validation_y = []
    with tf.device('/gpu:0'):
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

    validation_x = validation_x.astype('uint8')
    validation_x = da.from_array(validation_x, chunks = (827,128,128,3))
    validation_x.to_zarr("validation_x_128_dask.zarr")

    validation_y = np.array(validation_y)
    validation_y = mlb.fit_transform(validation_y)
    
    validation_y = validation_y.astype('uint8')
    validation_y = da.from_array(validation_y, chunks = (827))
    validation_y.to_zarr("validation_y_dask.zarr")

    print('Validation dask arrays have been successfully stored')
    del validation_annotations; del validation_x; del validation_y; del validationImagePaths

if 'test_x_128_dask.zarr' not in os.listdir('.') or 'test_y_dask.zarr' not in os.listdir('.'):
    print('Test dask arrays not found, saving now...')
    with open('test_annotations.json', 'r') as file:
        test_annotations = json.load(file)

    print("Storing Test Images into Dask Array format...")
    testImagePaths = list(paths.list_images('test'))
    shape = (len(testImagePaths),)+target_size+(3,)
    test_x = np.zeros(shape=shape, dtype=np.float16)
    test_y = []
    with tf.device('/gpu:0'):
        for i in tqdm(range(len(testImagePaths))):
            imagePath = testImagePaths[i]
            image = Image.open(imagePath)
            image = image.resize(size=target_size, resample=Image.LANCZOS)
            image = np.array(image)
            if (len(image.shape) == 2):
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            test_x[i]=image
            labels = set()
            for category in test_annotations[imagePath.split(os.path.sep)[-1]]['categories']: labels.add(category)
            test_y.append(list(labels))

    test_x = test_x.astype('uint8')
    test_x = da.from_array(test_x, chunks = (827,128,128,3))
    test_x.to_zarr("test_x_128_dask.zarr")

    test_y = np.array(test_y)
    test_y = mlb.fit_transform(test_y)
    
    test_y = test_y.astype('uint8')
    test_y = da.from_array(test_y, chunks = (827))
    test_y.to_zarr("test_y_dask.zarr")

    print('Test dask arrays have been successfully stored')
    del test_annotations; del test_x; del test_y; del testImagePaths    