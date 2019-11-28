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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

class textProcessing:
    @staticmethod
    def create_sequences(tokenizer, max_length, description, photo, vocab_size):
        X1, X2, Y = [], [], []
        seq = tokenizer.texts_to_sequences([(description.compute()).tolist()])[0]
        for i in range(1,len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo.compute())
            X2.append(in_seq)
            Y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(Y)
    
    @staticmethod
    def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
        while 1:
            for i in range(len(descriptions)):
                in_img, in_seq, out_word = textProcessing.create_sequences(tokenizer, max_length, descriptions[i], photos[i], vocab_size)
                nb = in_img.shape[0]
                yield [[in_img, in_seq], out_word, nb]
    
    @staticmethod
    def word_for_id(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    @staticmethod
    def generate_desc(rnn_model, tokenizer, photo, max_length):
        in_text = 'startseq'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # pad input
            sequence = pad_sequences([sequence], maxlen=max_length)
            # predict next word
            probs = rnn_model.predict([photo, sequence], verbose=0)
            # convert probability to integer
            probs_max = np.argmax(probs)
            # map integer to word
            word = textProcessing.word_for_id(probs_max, tokenizer)
            # stop if we cannot map the word
            if word is None:
                break
            # append as input for generating the next word
            in_text += ' ' + word
            if word == 'endseq':
                break
        return in_text


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
        list_cap = []
        for j in range(len(cap)):
            desc = cap[j]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            desc = [word for word in desc if word.isalpha()]
            list_cap.append(' '.join(desc))
        k[caption_list[i]] = list_cap

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

if 'all_descriptions_train.zarr' not in os.listdir('.') and 'embedding_trains_128.zarr' not in os.listdir('.'):
    # Add a new folder which has the train images for training RNN model
    trainImagePaths = list(paths.list_images('train'))
    # Make sure the length is set as this for a particular folder
    train_length = sum(len(train_captions[key.split(os.path.sep)[-1]]) for key in trainImagePaths)
    
    labelsList = np.load('labels.npy', allow_pickle = True)
    embedding_train = np.zeros(shape = (train_length+1,)+labelsList.shape)
    all_descriptions_train = []

    print("Not found embeddings for train, saving it now along with descriptions...")
    for i in tqdm(range(train_length)):
        imagePath = trainImagePaths[i]
        imageName = imagePath.split(os.path.sep)[-1]
        image = Image.open(imagePath)
        image = image.resize(size = (128,128), resample = Image.ANTIALIAS)
        arr = np.array(image)
        if (len(arr.shape) == 2):
            arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
        arr = np.expand_dims(arr,0)
        arr = arr/255
        embedding = cnn_model.predict(arr)[0]
        embedding = convert_embedding(embedding)
        for j in range(len(train_captions[imageName])):
            embedding_train[i+j] = embedding
            image_desc = ''.join(train_captions[imageName][j])
            updated_desc = 'startseq '+image_desc+' endseq'
            all_descriptions_train.append(updated_desc)
            break

    embedding_train = da.from_array(embedding_train, chunks = (827))
    embedding_train.to_zarr('embedding_train_128.zarr')

    all_descriptions_train = np.array(all_descriptions_train)
    all_descriptions_train = da.from_array(all_descriptions_train, chunks = (827))
    all_descriptions_train.to_zarr('all_descriptions_train.zarr')

    del all_descriptions_train; del embedding_train; del train_captions; del trainImagePaths

if 'all_descriptions_validation.zarr' not in os.listdir() and 'embedding_validations_128.zarr' not in os.listdir():
    # Add a new folder which has the validation images for training RNN model
    validationImagePaths = list(paths.list_images('validation'))
    # Make sure the length is set as this for a particular folder
    validation_length = sum(len(validation_captions[key.split(os.path.sep)[-1]]) for key in validationImagePaths)
    
    labelsList = np.load('labels.npy', allow_pickle = True)
    embedding_validation = np.zeros(shape = (validation_length+1,)+labelsList.shape)
    all_descriptions_validation = []
    
    print("Not found embeddings for validation, saving it now along with descriptions...")
    for i in tqdm(range(validation_length)):
        imagePath = validationImagePaths[i]
        imageName = imagePath.split(os.path.sep)[-1]
        image = Image.open(imagePath)
        image = image.resize(size = (128,128), resample = Image.ANTIALIAS)
        arr = np.array(image)
        if (len(arr.shape) == 2):
            arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
        arr = np.expand_dims(arr,0)
        arr = arr/255
        embedding = cnn_model.predict(arr)[0]
        embedding = convert_embedding(embedding)
        for j in range(len(validation_captions[imageName])):
            embedding_validation[i+j] = embedding
            image_desc = ''.join(validation_captions[imageName][j])
            updated_desc = 'startseq '+image_desc+' endseq'
            all_descriptions_validation.append(updated_desc)
            break

    embedding_validation = da.from_array(embedding_validation, chunks = (827))
    embedding_validation.to_zarr('embedding_validation_128.zarr')

    all_descriptions_validation = np.array(all_descriptions_validation)
    all_descriptions_validation = da.from_array(all_descriptions_validation, chunks = (827))
    all_descriptions_validation.to_zarr('all_descriptions_validation.zarr')

    del all_descriptions_validation; del embedding_validation; del validation_captions; del validationImagePaths

if 'all_descriptions_test.zarr' not in os.listdir() and 'embedding_tests_128.zarr' not in os.listdir():
    # Add a new folder which has the test images for training RNN model
    testImagePaths = list(paths.list_images('test'))
    # Make sure the length is set as this for a particular folder
    test_length = sum(len(test_captions[key.split(os.path.sep)[-1]]) for key in testImagePaths)
    
    labelsList = np.load('labels.npy', allow_pickle = True)
    embedding_test = np.zeros(shape = (test_length+1,)+labelsList.shape)
    all_descriptions_test = []

    print("Not found embeddings for test, saving it now along with descriptions...")
    for i in tqdm(range(test_length)):
        imagePath = testImagePaths[i]
        imageName = imagePath.split(os.path.sep)[-1]
        image = Image.open(imagePath)
        image = image.resize(size = (128,128), resample = Image.ANTIALIAS)
        arr = np.array(image)
        if (len(arr.shape) == 2):
            arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
        arr = np.expand_dims(arr,0)
        arr = arr/255
        embedding = cnn_model.predict(arr)[0]
        embedding = convert_embedding(embedding)
        for j in range(len(test_captions[imageName])):
            embedding_test[i+j] = embedding
            image_desc = ''.join(test_captions[imageName][j])
            updated_desc = 'startseq '+image_desc+' endseq'
            all_descriptions_test.append(updated_desc)
            break

    embedding_test = da.from_array(embedding_test, chunks = (827))
    embedding_test.to_zarr('embedding_test_128.zarr')

    all_descriptions_test = np.array(all_descriptions_test)
    all_descriptions_test = da.from_array(all_descriptions_test, chunks = (827))
    all_descriptions_test.to_zarr('all_descriptions_test.zarr')

    del all_descriptions_test; del embedding_test; del test_captions; del testImagePaths