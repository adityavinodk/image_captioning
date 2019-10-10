import os
import shutil
import sys
from multiprocessing import Pool
​
def worker(i):
    file_count = i
    length = len(os.listdir())
    while file_count<i+999 and file_count<length:
        file_name = os.listdir()[file_count]
        if file_name.split('.')[1]!='jpg': continue
        if file_count<49669:
            shutil.copy(file_name, '../train')
        elif file_count>49669 and file_count<66227:
            shutil.copy(file_name, '../validation')
        else:
            shutil.copy(file_name, '../test')
        print(file_count)
        file_count+=1
​
if __name__  == '__main__':
    if 'train' not in os.listdir():
        os.makedirs('train')
    if 'test' not in os.listdir():
        os.makedirs('test')
    if 'validation' not in os.listdir():
        os.makedirs('validation')
    os.chdir('train2017')
    p=Pool(processes = 10)
    p.map(worker,[i for i in range(0,120000,1000)])
    p.close()