import json as js
from tqdm import tqdm
import string
import sys
import keras
import tensorflow as tf
import json
import numpy as np
import dask.array as da
import math
import os
# os.environ["CUDA_DEVICE_ORDER"]="0"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from rnn_bimodel import RNNBimodel
from text_processing import textProcessing
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer as Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

if "all_descriptions_train.zarr" in os.listdir('.') and "embedding_train_128.zarr" in os.listdir('.'):
    with tf.device('/cpu:0'):
        embedding_train = da.from_zarr("embedding_train_128.zarr")
        desc_train = da.from_zarr("all_descriptions_train.zarr")
else: 
    print("Embedding and Descriptions dask array haven't been saved, please run text_preprocessing.py")
    exit()

if "all_descriptions_validation.zarr" in os.listdir('.') and "embedding_validation_128.zarr" in os.listdir('.'):
    with tf.device('/cpu:0'):
        embedding_validation = da.from_zarr("embedding_validation_128.zarr")
        desc_validation = da.from_zarr("all_descriptions_validation.zarr")
else: 
    print("Embedding and Descriptions dask array haven't been saved, please run text_preprocessing.py")
    exit()

if "all_descriptions_test.zarr" in os.listdir('.') and "embedding_test_128.zarr" in os.listdir('.'):
    with tf.device('/cpu:0'):
        embedding_test = da.from_zarr("embedding_test_128.zarr")
        desc_test = da.from_zarr("all_descriptions_test.zarr")
else: 
    print("Embedding and Descriptions dask array haven't been saved, please run text_preprocessing.py")
    exit()

allText = []
for desc_array in [desc_train.compute(), desc_train.compute(), desc_validation.compute()]:
    for descs in desc_array:
        for desc in descs:
            allText.append(desc)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(allText)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(d.split()) for d in allText)

def count_length(tokenizer, descriptions):
    Y = 0
    for i in tqdm(range(len(descriptions))):
        for j in range(len(descriptions[i])):
            seq = tokenizer.texts_to_sequences([(descriptions[i][j].compute()).tolist()])[0]
            Y+=len(seq)-1
    return Y

if 'best_model_rnn.h5' in os.listdir():
    model = load_model('best_model_rnn.h5')
    initial_epoch=initial_epoch = 0
else:
    model = RNNBimodel.build(vocab_size, max_length)
    initial_epoch = 0

print(model.summary())

epochs = 50
batch_size = 256

# nb_train_samples = count_length(tokenizer, desc_train)
# nb_validation_samples = count_length(tokenizer, desc_validation)
nb_train_samples = 472734
nb_validation_samples = 236630
print(nb_train_samples, nb_validation_samples)
print('max_length: {}\nvocab_size: {}\nepochs: {}'.format(max_length,vocab_size,epochs))
with tf.device('/gpu:0'):
    stop_early = EarlyStopping(monitor='val_loss',patience = 10)
    ModelCheck = ModelCheckpoint('best_model_rnn.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    callbacks = [stop_early, ModelCheck]
    train_generator = textProcessing.data_generator(desc_train, embedding_train, tokenizer, max_length, vocab_size)
    validation_generator = textProcessing.data_generator(desc_validation, embedding_validation, tokenizer, max_length, vocab_size)
    model.fit_generator(
        train_generator, 
        epochs=epochs, 
        steps_per_epoch=nb_train_samples//batch_size,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples//batch_size,
        callbacks = callbacks,
        initial_epoch=6,
        verbose=1)