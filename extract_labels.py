import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from six.moves import cPickle as pickle
import cv2
import numpy as np
import time
from imutils import paths
from multiprocessing import Manager, Process, Pool

target_size = (224,224)
train_data_dir = 'train'
validation_data_dir = 'validation'
nb_train_samples = os.listdir('train')
nb_validation_samples = os.listdir('validation')
epochs = 50
batch_size = 32

with open('train_annotations.json', 'r') as file:
    train_annotations = json.load(file)

with open('validation_annotations.json', 'r') as file:
    validation_annotations = json.load(file)

def label_worker(i,directory,annotations,data,labels):
    print("Process running - ",os.getpid())
    data_tmp = {}
    labels_tmp = {}
    os.chdir(directory)
    length = i+827
    for fc in range(i,i+20):
        if fc>=length: return
        file_name = os.listdir()[fc]
        image = cv2.imread(file_name)
        image = cv2.resize(image, (target_size[1], target_size[0]))
        data_tmp[file_name] = img_to_array(image)
        if(fc%100 == 0):
            print("remaining -> ",length-fc)
        labels = set()
        for category in annotations[file_name]['categories']:
            labels.add(category)
        labels_tmp[file_name] = list(labels)
        if(fc%400 == 0):
            labels.update(labels_tmp)
            data.update(data_tmp)
            data_tmp = {}
            labels_tmp = {}
    data.update(data_tmp)
    labels.update(labels_tmp)
    print("Process stopping - ",os.getpid())

if __name__ == '__main__':
    train_data = Manager().dict()
    train_labels = Manager().dict()
    validation_data = Manager().dict()
    validation_labels = Manager().dict()
    processPool = Pool(6)
    if 'train_labels.json' not in os.listdir() and 'train_data.pkl' not in os.listdir():
        # processPool.starmap(label_worker, [(i,'train',train_annotations,train_data,train_labels) for i in range(0,100,10)])
        label_worker(0, 'train', train_annotations, train_data, train_labels)
        print(type(train_data))
        with open('train_data.pkl','wb') as fp:
            pickle.dump(train_data, fp)
        with open('train_labels.json','w') as fp:
            json.dump(train_labels.copy(),fp)
    # if 'validation_labels.json' not in os.listdir() and 'validation_data.pkl' not in os.listdir():
    #     processPool.starmap(label_worker, [(i,'validation',validation_annotations,validation_data,validation_labels) for i in range(0,4962,827)])
    #     with open('validation_data.pkl','wb') as fp:
    #         pickle.dump(validation_data, fp)
    #     with open('validation_labels.json','w') as fp:
    #         json.dump(validation_labels.copy(),fp)