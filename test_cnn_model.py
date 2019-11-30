import json
import os
import numpy as np
import dask.array as da
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import MultiLabelBinarizer

if 'labels.npy' not in os.listdir():
    with open('annotations/instances_train2014.json','r') as file:
        instances = json.load(file)

    labelsList = np.array([category['name'] for category in instances['categories']])
    np.save('labels.npy', labelsList)
    del instances

labelsList = np.load('labels.npy', allow_pickle=True)

target_size = (128,128)
test_data_dir = 'test'
nb_test_samples = len(os.listdir(test_data_dir))
epochs = 50
batch_size = 60

if "best_model.h5" in os.listdir():
    model = load_model("best_model.h5")
else: 
    print("Please run CNN model in Image_captioning_CNN.ipynb")
    exit()

if "test_x_128_dask.zarr" in os.listdir('.') and "test_y_dask.zarr" in os.listdir('.'):
    with tf.device('/cpu:0'):
        test_x = da.from_zarr("test_x_128_dask.zarr")
        test_y = da.from_zarr("test_y_dask.zarr")
else:
    print("Test Dask Arrays haven't been set. Please run extract_labels.py beefore running this file.")
    exit()

test_datagen = ImageDataGenerator(rescale=1./255)

with tf.device('/gpu:0'):
    stop_early = EarlyStopping(monitor='val_loss',patience = 20)
    reduceLR = ReduceLROnPlateau(monitor='val_loss',paitence = 20, factor=0.2, min_lr = 0.0001)
    callbacks = [stop_early, reduceLR]
    scores = model.evaluate_generator(
        test_datagen.flow(test_x, test_y, batch_size=batch_size),
        steps = nb_test_samples//batch_size,
        callbacks = callbacks
    )

print('Accuracy of CNN model is - ', scores[1])