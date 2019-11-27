import json as js
from tqdm import tqdm
import string
import sys
import os
from PIL import Image
import numpy as np
from imutils import paths
import keras
import tensorflow as tf
import dask.array as da
from tensorflow.keras.models import load_model

def convert_embedding(embedding):
    embedding = (embedding*1000).astype(int)
    for x in range(len(embedding)):
        if embedding[x]>100:
            embedding[x] = 1
        elif embedding[x] > 50:
            embedding[x] = 0.5
        else:
            embedding[x]=0
    return embedding

with open("train_captions.json") as file:
    train_captions = js.load(file)
with open("test_captions.json") as file:
    test_captions = js.load(file)
with open("validation_captions.json") as file:
    validation_captions = js.load(file)
    
table = str.maketrans('', '', string.punctuation)
for k in [train_captions, test_captions, validation_captions]:
    caption_list = list(k.keys())
    for i in tqdm(range(len(caption_list))):
        cap = k[caption_list[i]]
        for j in range(cap[0]):
            desc = cap[1][j]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            cap[1][j] =  ' '.join(desc)
        k[caption_list[i]] = cap

with open("train_captions.json",'w') as file:
    js.dump(train_captions, file)
with open("test_captions.json",'w') as file:
    js.dump(test_captions, file)
with open("validation_captions.json",'w') as file:
    js.dump(validation_captions, file)

if "best_model.h5" in os.listdir():
    cnn_model = load_model("best_model.h5")
else: 
    print("Please run CNN model in Image_captioning_CNN.ipynb")
    exit()

if 'all_descriptions_train.zarr' not in os.listdir() or 'embedding_trains_128.zarr' not in os.listdir():
    trainImagePaths = list(paths.list_images('train'))
    labelsList = np.load('labels.npy', allow_pickle = True)
    embedding_train = np.zeros(shape = (len(trainImagePaths),)+labelsList.shape)
    all_descriptions_train = []

    print("Not found embeddings for train, saving it now along with descriptions...")
    for i in tqdm(range(len(trainImagePaths))):
        imagePath = trainImagePaths[i]
        imageName = imagePath.split(os.path.sep)[-1]
        image = Image.open(imagePath)
        image = image.resize(size = (128,128), resample = Image.ANTIALIAS)
        arr = np.array(image)
        arr = np.expand_dims(arr,0)
        arr = arr/255
        embedding = cnn_model.predict(arr)[0]
        embedding = convert_embedding(embedding)
        for j in range(train_captions[imageName][0]):
            embedding_train[i+j] = embedding
            image_desc = ''.join(train_captions[imageName][1][j])
            updated_desc = 'startseq '+image_desc+' endseq'
            all_descriptions_train.append(updated_desc)

    embedding_train = da.from_array(embedding_train, chunks = (827))
    embedding_train.to_zarr('embedding_train_128.zarr')

    all_descriptions_train = np.array(all_descriptions_train)
    all_descriptions_train = da.from_array(all_descriptions_train, chunks = (827))
    all_descriptions_train.to_zarr('all_descriptions_train.zarr')

    del all_descriptions_train; del embedding_train; del train_captions; del trainImagePaths

if 'all_descriptions_validation.zarr' not in os.listdir() or 'embedding_validations_128.zarr' not in os.listdir():
    validationImagePaths = list(paths.list_images('validation'))
    labelsList = np.load('labels.npy', allow_pickle = True)
    embedding_validation = np.zeros(shape = (len(validationImagePaths),)+labelsList.shape)
    all_descriptions_validation = []
    
    print("Not found embeddings for validation, saving it now along with descriptions...")
    for i in tqdm(range(len(validationImagePaths))):
        imagePath = validationImagePaths[i]
        imageName = imagePath.split(os.path.sep)[-1]
        image = Image.open(imagePath)
        image = image.resize(size = (128,128), resample = Image.ANTIALIAS)
        arr = np.array(image)
        arr = np.expand_dims(arr,0)
        arr = arr/255
        embedding = cnn_model.predict(arr)[0]
        embedding = convert_embedding(embedding)
        for j in range(validation_captions[imageName][0]):
            embedding_validation[i+j] = embedding
            image_desc = ''.join(validation_captions[imageName][1][j])
            updated_desc = 'startseq '+image_desc+' endseq'
            all_descriptions_validation.append(updated_desc)

    embedding_validation = da.from_array(embedding_validation, chunks = (827))
    embedding_validation.to_zarr('embedding_validation_128.zarr')

    all_descriptions_validation = np.array(all_descriptions_validation)
    all_descriptions_validation = da.from_array(all_descriptions_validation, chunks = (827))
    all_descriptions_validation.to_zarr('all_descriptions_validation.zarr')

    del all_descriptions_train; del embedding_train; del train_captions; del validationImagePaths

if 'all_descriptions_test.zarr' not in os.listdir() or 'embedding_tests_128.zarr' not in os.listdir():
    testImagePaths = list(paths.list_images('test'))
    labelsList = np.load('labels.npy', allow_pickle = True)
    embedding_test = np.zeros(shape = (len(testImagePaths),)+labelsList.shape)
    all_descriptions_test = []

    print("Not found embeddings for test, saving it now along with descriptions...")
    for i in tqdm(range(len(testImagePaths))):
        imagePath = testImagePaths[i]
        imageName = imagePath.split(os.path.sep)[-1]
        image = Image.open(imagePath)
        image = image.resize(size = (128,128), resample = Image.ANTIALIAS)
        arr = np.array(image)
        arr = np.expand_dims(arr,0)
        arr = arr/255
        embedding = cnn_model.predict(arr)[0]
        embedding = convert_embedding(embedding)
        for j in range(test_captions[imageName][0]):
            embedding_test[i+j] = embedding
            image_desc = ''.join(test_captions[imageName][1][j])
            updated_desc = 'startseq '+image_desc+' endseq'
            all_descriptions_test.append(updated_desc)

    embedding_test = da.from_array(embedding_test, chunks = (827))
    embedding_test.to_zarr('embedding_test_128.zarr')

    all_descriptions_test = np.array(all_descriptions_test)
    all_descriptions_test = da.from_array(all_descriptions_test, chunks = (827))
    all_descriptions_test.to_zarr('all_descriptions_test.zarr')

    del all_descriptions_train; del embedding_train; del train_captions; del testImagePaths