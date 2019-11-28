import json
import sys
import os
from tqdm import tqdm
from timeit import default_timer as timer
import tensorflow as tf

train,val,test = [os.listdir('train'), os.listdir('validation'), os.listdir('test')]

if __name__ == "__main__":
    with tf.device('/cpu:0'):
        st = timer()
        print('Loading Captions ...')
        with open("./annotations/captions_train2014.json", 'r') as file:
            captions = json.load(file)['annotations']
        print("Done Loading Captions, in Time :", timer() - st)
        st = timer()
        print('Loading Image Ids ...')
        with open("./annotations/instances_train2014.json", 'r') as file:
            raw_images = json.load(file)['images']
        print("Done Loading Image Ids, in Time :", timer() - st)
    # print(len(captions))

    images = {}
    for i in range(len(raw_images)):
        images[raw_images[i]['id']] = raw_images[i]['file_name']

    print(images[116100] , '\n\n' , captions[0])

    train_cap = {}
    val_cap = {}
    test_cap = {}
    with tf.device('/gpu:0'):
        calc = 0
        st = timer()
        print('Preprocessing...')
        for i in tqdm(range(len(captions))):
            id = captions[i]['image_id']
            if images[id] in train:
                if images[id] not in train_cap:
                    train_cap[images[id]] = [captions[i]['caption']]
                else:
                    train_cap[images[id]].append(captions[i]['caption'])
            elif images[id] in val:
                if images[id] not in val_cap:
                    val_cap[images[id]] = [captions[i]['caption']]
                else:
                    val_cap[images[id]].append(captions[i]['caption'])
            elif images[id] in test:
                if images[id] not in test_cap:
                    test_cap[images[id]] = [captions[i]['caption']]
                else:
                    test_cap[images[id]].append(captions[i]['caption'])
            else:
                calc +=1
        print("Done Preprocessing Part 1, in Time :", timer() - st)
    
    with open ('validation_captions.json', 'w') as outfile:
        json.dump(val_cap, outfile)
    with open ('train_captions.json', 'w') as outfile:
        json.dump(train_cap, outfile)
    with open ('test_captions.json', 'w') as outfile:
        json.dump(test_cap, outfile)