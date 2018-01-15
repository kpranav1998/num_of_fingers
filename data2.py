import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflearn
import tensorflow as tf
from PIL import Image
#for writing text files
import glob
import os
import random
#reading images from a text file
from tflearn.data_utils import image_preloader
from tqdm import tqdm
import math

TRAIN_DATA5='/home/kpranav1998/PycharmProjects/num_fingers/images/test/FIVE'
TRAIN_DATA4='/home/kpranav1998/PycharmProjects/num_fingers/images/test/FOUR'
TRAIN_DATA3='/home/kpranav1998/PycharmProjects/num_fingers/images/test/THREE'
TRAIN_DATA2='/home/kpranav1998/PycharmProjects/num_fingers/images/test/TWO'
TRAIN_DATA1='/home/kpranav1998/PycharmProjects/num_fingers/images/test/ONE'
TRAIN_DATA0='/home/kpranav1998/PycharmProjects/num_fingers/images/test/NONE'

fr = open('train_data.txt', 'w')
files=[]
file_names=os.listdir(TRAIN_DATA0)
for filename in tqdm(file_names):
    path=os.path.join(TRAIN_DATA0,filename)
    files.append([path,' 0'])

file_names=os.listdir(TRAIN_DATA1)

for filename in tqdm(file_names):
    path=os.path.join(TRAIN_DATA1,filename)
    files.append([path, ' 1'])

file_names=os.listdir(TRAIN_DATA2)
for filename in tqdm(file_names):
    path=os.path.join(TRAIN_DATA2,filename)
    files.append([path, ' 2'])

file_names=os.listdir(TRAIN_DATA3)

for filename in tqdm(file_names):
    path=os.path.join(TRAIN_DATA3,filename)
    files.append([path,' 3'])

file_names=os.listdir(TRAIN_DATA4)

for filename in tqdm(file_names):
    path=os.path.join(TRAIN_DATA4,filename)
    files.append([path, ' 4'])

file_names=os.listdir(TRAIN_DATA5)

for filename in tqdm(file_names):
    path=os.path.join(TRAIN_DATA5,filename)
    files.append([path, ' 5'])

random.shuffle(files)

for file in files:
    fr.write(file[0]+file[1]+'\n')



