import os
import json
import numpy as np
import time
from PIL import Image
from imutils import paths
from tqdm import tqdm
<<<<<<< HEAD
import dask.array as da
import tensorflow as tf
target_size = (128,128)

if 'train_x_128_dask.zarr' not in os.listdir('.'):
    with open('train_annotations.json', 'r') as file:
        train_annotations = json.load(file)

    print("Storing train Images into Numpy Array format...")
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

    train_y = np.array(train_y)
    train_x = train_x.astype(int)
    train_x = train_x.astype('uint8')
    train_x = da.from_array(train_x, chunks = (827,128,128,3))
    train_x.to_zarr("train_x_128_dask.zarr")
    # np.save('train_x_128.npy', train_x)
    np.save('train_y.npy', train_y)

    del train_annotations; del train_x; del train_y; del trainImagePaths;

if 'validation_x_128_dask.zarr' not in os.listdir('.'):
    with open('validation_annotations.json', 'r') as file:
        validation_annotations = json.load(file)

    print("Storing validation Images into Numpy Array format...")
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

    validation_y = np.array(validation_y)
    validation_x = validation_x.astype(int)
    validation_x = validation_x.astype('uint8')
    validation_x = da.from_array(validation_x, chunks = (827,128,128,3))
    validation_x.to_zarr("validation_x_128_dask.zarr")
    # np.save('validation_x_128.npy', validation_x)
    np.save('validation_y.npy', validation_y)

    del validation_annotations; del validation_x; del validation_y; del validationImagePaths; 

if 'test_x_128_dask.zarr' not in os.listdir('.'):
    with open('test_annotations.json', 'r') as file:
        test_annotations = json.load(file)

    print("Storing Test Images into Numpy Array format...")
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

    test_y = np.array(test_y)
    test_x = test_x.astype(int)
    test_x = test_x.astype('uint8')
    test_x = da.from_array(test_x, chunks = (827,128,128,3))
    test_x.to_zarr("test_x_128_dask.zarr")
    # np.save('test_x_128.npy', test_x)
    np.save('test_y.npy', test_y)

=======

target_size = (128,128)

if 'train_x_128.npy' not in os.listdir('.'):
    with open('train_annotations.json', 'r') as file:
        train_annotations = json.load(file)

    print("Storing Train Images into Numpy Array format...")
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

    del train_annotations; del train_x; del train_y; del trainImagePaths; 

if 'validation_x_128.npy' not in os.listdir('.'):
    with open('validation_annotations.json', 'r') as file:
        validation_annotations = json.load(file)

    print("Storing Validation Images into Numpy Array format...")
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

    del validation_annotations; del validation_x; del validation_y; del validationImagePaths; 

if 'test_x_128.npy' not in os.listdir('.'):
    with open('test_annotations.json', 'r') as file:
        test_annotations = json.load(file)

    print("Storing Test Images into Numpy Array format...")
    testImagePaths = list(paths.list_images('test'))
    shape = (len(testImagePaths),)+target_size+(3,)
    test_x = np.zeros(shape=shape, dtype=np.float16)
    test_y = []

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

    test_y = np.array(test_y)
    np.save('test_x_128.npy', test_x)
    np.save('test_y.npy', test_y)

>>>>>>> 53fa87cc674829694f016daadfa6ca4ce5855718
    del test_annotations; del test_x; del test_y; del testImagePaths; 