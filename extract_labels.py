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

if 'labels.npy' not in os.listdir('data'):
    with open('data/image_data/instances.json','r') as file:
        instances = json.load(file)

    labelsList = np.array([category['name'] for category in instances['categories']])
    np.save('data/labels.npy', labelsList)
    del instances

labelsList = np.load('data/labels.npy', allow_pickle=True)
mlb = MultiLabelBinarizer(labelsList)

if 'train_x_128_dask.zarr' not in os.listdir('data') or 'train_y_dask.zarr' not in os.listdir('data'):
    print('Train dask arrays not found, saving now...')
    with open('data/train_annotations.json', 'r') as file:
        train_annotations = json.load(file)

    print("Storing train Images into Dask Array format...")
    trainImagePaths = list(paths.list_images('data/train'))
    trainImagePathsLength = len(trainImagePaths)
    
    shape = (1000,)+target_size+(3,)
    train_x = np.zeros(shape=shape, dtype=np.float16)
    train_x_dask = da.from_array(np.zeros(shape=(0,)+target_size+(3,)), chunks=827)
    train_y = []
    done_val = 0

    with tf.device('/gpu:0'):
        for i in tqdm(range(trainImagePathsLength)):
            imagePath = trainImagePaths[i]
            image = Image.open(imagePath)
            image = image.resize(size=target_size, resample=Image.LANCZOS)
            image = np.array(image)
            if (len(image.shape) == 2):
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            train_x[i-done_val] = image
            if i-done_val==999:
                done_val = i+1
                train_x_dask = da.concatenate([train_x, train_x_dask], axis = 0)
                train_x = np.zeros(shape=shape, dtype=np.float16)
            labels = set()
            for category in train_annotations[imagePath.split(os.path.sep)[-1]]['categories']: labels.add(category)
            train_y.append(list(labels))

    train_x_dask.to_zarr("data/train_x_128_dask.zarr")

    train_y = np.array(train_y)
    train_y = mlb.fit_transform(train_y)
    
    train_y = train_y.astype('uint8')
    train_y = da.from_array(train_y, chunks = (827))
    train_y.to_zarr("data/train_y_dask.zarr")

    print('Train dask arrays have been successfully stored')
    del train_annotations; del train_x; del train_y; del trainImagePaths

if 'validation_x_128_dask.zarr' not in os.listdir('data') or 'validation_y_dask.zarr' not in os.listdir('data'):
    print('Validation dask arrays not found, saving now...')
    with open('data/validation_annotations.json', 'r') as file:
        validation_annotations = json.load(file)

    print("Storing Validation Images into Dask Array format...")
    validationImagePaths = list(paths.list_images('data/validation'))
    validationImagePathsLength = len(validationImagePaths)
    
    shape = (1000,)+target_size+(3,)
    validation_x = np.zeros(shape=shape, dtype=np.float16)
    validation_x_dask = da.from_array(np.zeros(shape=(0,)+target_size+(3,)), chunks=827)
    validation_y = []
    done_val = 0

    with tf.device('/gpu:0'):
        for i in tqdm(range(validationImagePathsLength)):
            imagePath = validationImagePaths[i]
            image = Image.open(imagePath)
            image = image.resize(size=target_size, resample=Image.LANCZOS)
            image = np.array(image)
            if (len(image.shape) == 2):
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            validation_x[i-done_val] = image
            if i-done_val==999:
                done_val = i+1
                validation_x_dask = da.concatenate([validation_x, validation_x_dask], axis = 0)
                validation_x = np.zeros(shape=shape, dtype=np.float16)
            labels = set()
            for category in validation_annotations[imagePath.split(os.path.sep)[-1]]['categories']: labels.add(category)
            validation_y.append(list(labels))

    validation_x_dask.to_zarr("data/validation_x_128_dask.zarr")

    validation_y = np.array(validation_y)
    validation_y = mlb.fit_transform(validation_y)
    
    validation_y = validation_y.astype('uint8')
    validation_y = da.from_array(validation_y, chunks = (827))
    validation_y.to_zarr("data/validation_y_dask.zarr")

    print('validation dask arrays have been successfully stored')
    del validation_annotations; del validation_x; del validation_y; del validationImagePaths

if 'test_x_128_dask.zarr' not in os.listdir('data') or 'test_y_dask.zarr' not in os.listdir('data'):
    print('Test dask arrays not found, saving now...')
    with open('data/test_annotations.json', 'r') as file:
        test_annotations = json.load(file)

    print("Storing Test Images into Dask Array format...")
    testImagePaths = list(paths.list_images('data/test'))
    testImagePathsLength = len(testImagePaths)
    
    shape = (1000,)+target_size+(3,)
    test_x = np.zeros(shape=shape, dtype=np.float16)
    test_x_dask = da.from_array(np.zeros(shape=(0,)+target_size+(3,)), chunks=827)
    test_y = []
    done_val = 0

    with tf.device('/gpu:0'):
        for i in tqdm(range(testImagePathsLength)):
            imagePath = testImagePaths[i]
            image = Image.open(imagePath)
            image = image.resize(size=target_size, resample=Image.LANCZOS)
            image = np.array(image)
            if (len(image.shape) == 2):
                image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            test_x[i-done_val] = image
            if i-done_val==999:
                done_val = i+1
                test_x_dask = da.concatenate([test_x, test_x_dask], axis = 0)
                test_x = np.zeros(shape=shape, dtype=np.float16)
            labels = set()
            for category in test_annotations[imagePath.split(os.path.sep)[-1]]['categories']: labels.add(category)
            test_y.append(list(labels))

    test_x_dask.to_zarr("data/test_x_128_dask.zarr")

    test_y = np.array(test_y)
    test_y = mlb.fit_transform(test_y)
    
    test_y = test_y.astype('uint8')
    test_y = da.from_array(test_y, chunks = (827))
    test_y.to_zarr("data/test_y_dask.zarr")

    print('test dask arrays have been successfully stored')
    del test_annotations; del test_x; del test_y; del testImagePaths