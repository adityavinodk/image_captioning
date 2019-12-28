import os
import shutil
import sys
from multiprocessing import Pool

if 'images' not in os.listdir('data'):
    print("Add a directory of all images under data named as 'images'")
    exit()

totalLength = len(os.listdir('data/images'))
train_limit = int(0.6*totalLength)
validation_limit = train_limit+int(0.2*totalLength)

def worker(i):
    file_count = i
    length = len(os.listdir())
    while file_count<i+999 and file_count<length:
        file_name = os.listdir()[file_count]
        if file_name.split('.')[1]!='jpg': continue
        if file_count<=train_limit:
            shutil.copy(file_name, '../train')
        elif file_count>train_limit and file_count<=validation_limit:
            shutil.copy(file_name, '../validation')
        else:
            shutil.copy(file_name, '../test')
        print(file_count)
        file_count+=1

if __name__  == '__main__':
    if 'train' not in os.listdir('data'):
        os.makedirs('data/train')
    if 'test' not in os.listdir('data'):
        os.makedirs('data/test')
    if 'validation' not in os.listdir('data'):
        os.makedirs('data/validation')
    os.chdir('data/images')
    p=Pool(processes = 10)
    p.map(worker,[i for i in range(0,totalLength,1000)])
    p.close()