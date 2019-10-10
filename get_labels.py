import os
import json

from keras.preprocessing.image import img_to_array

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
from imutils import paths
from PIL import Image
from tqdm import tqdm
import shutil
import tarfile
from imutils import paths
from multiprocessing import Manager, Process, Pool

with open('validation_annotations.json', 'r') as file:
    validation_annotations = json.load(file)

validation_data = Manager().list()
validation_labels = Manager().list()
target_size = (224,224)
    
def validation_label_worker(i, validation_annotations, length):
    for fc in range(i,i+int(length/4)):
        if fc>=length: return
        file_name = list(paths.list_images('validation'))[fc]
        image = cv2.imread(file_name)
        image = cv2.resize(image, (target_size[1], target_size[0]))
        validation_data.append(img_to_array(image))
        labels = set()
        for category in validation_annotations[file_name.split(os.path.sep)[-1]]['categories']: labels.add(category)
        validation_labels.append(list(labels))

processPool = Pool(4)
processPool.starmap(validation_label_worker, [(i,validation_annotations,len(os.listdir('validation'))) for i in [0,4135,8270,12405]])

print(validation_labels)