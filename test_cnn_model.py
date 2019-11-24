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

labels = da.from_zarr('labels.zarr')

target_size = (128,128)
test_data_dir = 'test'
nb_test_samples = len(os.listdir(test_data_dir))
epochs = 50
batch_size = 60

if "best_model.h5" in os.listdir():
    model = load_model("best_model.h5")

if "test_x_128_dask.zarr" in os.listdir('.') and "test_y_dask.zarr" in os.listdir('.'):
    with tf.device('/cpu:0'):
        test_x = da.from_zarr("test_x_128_dask.zarr")
        test_y = da.from_zarr("test_y_dask.zarr")

else: 
    test_x = np.load('test_x_128.npy', allow_pickle=True)
    test_x = test_x.astype('uint8')
    test_x = da.from_array(test_x, chunks = (827,128,128,3))
    test_x.to_zarr("test_x_128_dask.zarr")
    
    test_y = np.load('test_y.npy', allow_pickle=True)
    test_y = np.array(test_y)
    
    mlb = MultiLabelBinarizer(labels)
    test_y = mlb.fit_transform(test_y)
    
    test_y = test_y.astype('uint8')
    test_y = da.from_array(test_y, chunks = (827,128,128,3))
    test_y.to_zarr("test_y_128_dask.zarr")

test_datagen = ImageDataGenerator(rescale=1./255)

with tf.device('/gpu:0'):
    stop_early = EarlyStopping(monitor='val_loss',patience = 20)
    reduceLR = ReduceLROnPlateau(monitor='val_loss',paitence = 20, factor=0.2, min_lr = 0.0001)
    ModelCheck = ModelCheckpoint('best_model_loss.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    callbacks = [stop_early, reduceLR, ModelCheck]
    model.evaluate_generator(
        test_datagen.flow(test_x, test_y, batch_size=batch_size),
        steps = nb_test_samples//batch_size,
        callbacks = callbacks,
        use_multiprocessing=True
    )