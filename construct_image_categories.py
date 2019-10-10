import json
import sys
import os
from tqdm import tqdm
from multiprocessing import Manager, Process, Pool
# import time

def worker(i,annotations,categories,mydict):
    print("Process running - ",os.getpid())
    os.chdir('validation')
    for fc in range(i,i+5000):
        if fc>=len(os.listdir()):
            return
        file_name = os.listdir()[fc]
        if file_name.split('.')[1]!='jpg': continue
        if file_name not in mydict:
            fid = int(file_name.split('_')[2].split('.')[0])
            objects = {}
            for annotation in annotations:
                if fid == annotation['image_id']:
                    for category in categories:
                        if annotation['category_id'] == category[0]:
                            if category[1] not in objects:
                                objects[category[1]] = {'id':category[0], 'count':1, 'boundings':[annotation['bbox']]}
                            else:
                                objects[category[1]]['count']+=1
                                objects[category[1]]['boundings'].append(annotation['bbox'])
                            break
            im_an = {'id': fid, 'categories': objects}
            mydict[file_name] = im_an
    print("Process stopping - ",os.getpid())

if __name__ == "__main__":
    manager = Manager()
    categories = []
    mydict = manager.dict()
    with open("annotations/instances_train2014.json", 'r') as file:
        instances = json.load(file)
    for category in instances['categories']:
        categories.append([category['id'],category['name']])
    annotations = instances['annotations']
    p = Pool(4)
    p.starmap(worker, [(i,annotations,categories,mydict) for i in [0,4135,8270,12405]])
    with open ('validation_annotations.json', 'w') as outfile:
        json.dump(mydict.copy(), outfile)
    print(mydict)
    print("Operation Done.")