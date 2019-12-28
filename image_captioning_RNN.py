from tqdm import tqdm
import keras
import tensorflow as tf
import dask.array as da
import os
import pickle as pkl
from rnn_bimodel import RNNBimodel
from text_processing import textProcessing
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer as Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

if "all_descriptions_train.zarr" in os.listdir('data') and "embedding_train_128.zarr" in os.listdir('data'):
    with tf.device('/cpu:0'):
        embedding_train = da.from_zarr("data/embedding_train_128.zarr")
        desc_train = da.from_zarr("data/all_descriptions_train.zarr")
else: 
    print("Embedding and Descriptions dask array haven't been saved for train, please run text_preprocessing.py")
    exit()

if "all_descriptions_validation.zarr" in os.listdir('data') and "embedding_validation_128.zarr" in os.listdir('data'):
    with tf.device('/cpu:0'):
        embedding_validation = da.from_zarr("data/embedding_validation_128.zarr")
        desc_validation = da.from_zarr("data/all_descriptions_validation.zarr")
else: 
    print("Embedding and Descriptions dask array haven't been saved for validation, please run text_preprocessing.py")
    exit()

if "all_descriptions_test.zarr" in os.listdir('data') and "embedding_test_128.zarr" in os.listdir('data'):
    with tf.device('/cpu:0'):
        embedding_test = da.from_zarr("data/embedding_test_128.zarr")
        desc_test = da.from_zarr("data/all_descriptions_test.zarr")
else: 
    print("Embedding and Descriptions dask array haven't been saved for testing, please run text_preprocessing.py")
    exit()

allText = []
for desc_array in [desc_train.compute(), desc_train.compute(), desc_validation.compute()]:
    for descs in desc_array:
        for desc in descs:
            allText.append(desc)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(allText)
with open('tokenizer.pickle', 'wb') as handle:
    pkl.dump(tokenizer, handle, protocol=pkl.HIGHEST_PROTOCOL)
print("Saved tokenizer file....")
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(d.split()) for d in allText)

def count_length(tokenizer, descriptions):
    try:
        with tf.device('/gpu:0'):
            Y = 0
            for i in tqdm(range(len(descriptions))):
                for j in range(len(descriptions[i])):
                    seq = tokenizer.texts_to_sequences([(descriptions[i][j].compute()).tolist()])[0]
                    Y+=len(seq)-1
            return Y
    except:
        with tf.device('/cpu:0'):
            Y = 0
            for i in tqdm(range(len(descriptions))):
                for j in range(len(descriptions[i])):
                    seq = tokenizer.texts_to_sequences([(descriptions[i][j].compute()).tolist()])[0]
                    Y+=len(seq)-1
            return Y

if 'best_model_rnn.h5' in os.listdir('data'):
    model = load_model('data/best_model_rnn.h5')
    initial_epoch=initial_epoch = 0
else:
    model = RNNBimodel.build(vocab_size, max_length)
    initial_epoch = 0

print(model.summary())

epochs = 50
batch_size = 512

nb_train_samples = count_length(tokenizer, desc_train)
nb_validation_samples = count_length(tokenizer, desc_validation)
print(nb_train_samples, nb_validation_samples)
print('max_length: {}\nvocab_size: {}\nepochs: {}'.format(max_length,vocab_size,epochs))

with tf.device('/gpu:0'):
    stop_early = EarlyStopping(monitor='val_loss',patience = 10)
    ModelCheck = ModelCheckpoint('data/best_model_rnn.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    callbacks = [stop_early, ModelCheck]
    train_generator = textProcessing.data_generator(desc_train, embedding_train, tokenizer, max_length, vocab_size)
    validation_generator = textProcessing.data_generator(desc_validation, embedding_validation, tokenizer, max_length, vocab_size)
    history = model.fit_generator(
        train_generator, 
        epochs=6, 
        steps_per_epoch=nb_train_samples//batch_size,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples//batch_size,
        callbacks = callbacks,
        initial_epoch=0,
        verbose=1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()