import numpy as np
from PIL import Image
import time
import os
import sys
import datetime
import keras
import math
from keras.models import Model
import random
from keras.datasets import mnist
from numpy import arange
import argparse
from keras.applications import vgg19,resnet50
from datautils import get_data,get_model,data_proprecessing
from keras.applications.vgg19 import preprocess_input
import tensorflow as tf


    
def generate_ratio_vector(num,ratio):
    import math
    perturbate_num = math.ceil(num * ratio)
    non_perturbate_num = num - perturbate_num
    a = np.zeros(perturbate_num)+1
    b = np.zeros(non_perturbate_num)
    a_b = np.concatenate((a,b), axis=0)
    np.random.shuffle(a_b)
    return a_b
    
def black(image,i=0,j=0):
  image = np.array(image, dtype=float)
  image[0+2*i:2+2*i,0+2*j:2+2*j]=0
  return image.copy()
  
def white(image,i=0,j=0):
  image = np.array(image, dtype=float)
  image[0+2*i:2+2*i,0+2*j:2+2*j]=255
  return image.copy()
  
def reverse_color(image,i=0,j=0):
    image = np.array(image, dtype=float)
    part = image[0+2*i:2+2*i,0+2*j:2+2*j].copy()
    reversed_part = 255-part
    image[0+2*i:2+2*i,0+2*j:2+2*j] = reversed_part
    return image
    
def gauss_noise(image,i=0,j=0,mean=0, var=0.1,ratio=1.0):
  image = np.array(image, dtype=float)
  image = image.astype('float32') / 255
  part = image[0+2*i:2+2*i,0+2*j:2+2*j].copy()
  ratio_vector = generate_ratio_vector(len(part.ravel()),ratio).reshape(part.shape)
  noise = np.random.normal(mean, var ** 0.5, part.shape)
  noise = noise * ratio_vector
  image[0+2*i:2+2*i,0+2*j:2+2*j] += noise
  image = np.clip(image, 0, 1)
  image *= 255
  return image.copy()


def shuffle_pixel(image,i=0,j=0):
    image = np.array(image, dtype=float)
    # image /= 255
    part = image[0+2*i:2+2*i,0+2*j:2+2*j].copy()
    part_r = part.reshape(-1,1)
    np.random.shuffle(part_r)
    part_r = part_r.reshape(part.shape)
    image[0+2*i:2+2*i,0+2*j:2+2*j] = part_r
    return image
    

exp_id = sys.argv[1]
perturbate_type = sys.argv[2]

if __name__ == '__main__':
    #2
    import time
    start = time.time()
    x,y = get_data(exp_id)

    # making needed directory
    basedir = os.path.dirname(__file__)
    
    if not os.path.exists(os.path.join(basedir, 'input')):
        os.mkdir(os.path.join(basedir, 'input'))
    basedir = os.path.join(basedir, 'input')
    basedir = os.path.join(basedir, exp_id)
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    basedir = os.path.join(basedir, exp_id)
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    if not os.path.exists(os.path.join(basedir, 'gauss')):
        os.mkdir(os.path.join(basedir, 'gauss'))
    if not os.path.exists(os.path.join(basedir, 'reverse')):
        os.mkdir(os.path.join(basedir, 'reverse'))
    if not os.path.exists(os.path.join(basedir, 'black')):
        os.mkdir(os.path.join(basedir, 'black'))
    if not os.path.exists(os.path.join(basedir, 'white')):
        os.mkdir(os.path.join(basedir, 'white'))
    if not os.path.exists(os.path.join(basedir, 'shuffle')):
        os.mkdir(os.path.join(basedir, 'shuffle'))


    image_id = 0 
    for image in x:
        #gausstemp = []
        #reversetemp = []
        #blacktemp = []
        #whitetemp = []
        #shuffletemp = []
        tt_temp = []
        for i in range(16):
            for j in range(16):
                # temp.append(gauss_noise(image,i,j,ratio=1.0,var=0.8)-X_train_mean)
                # temp.append(gauss_noise(image+X_train_mean,i,j,ratio=1.0,var=0.8)-X_train_mean)
                #temp.append(white(image+X_train_mean,i,j)-X_train_mean)
                #gausstemp.append(preprocess_input(gauss_noise(image,i,j,ratio=1.0,var=0.8)*255))
                #reversetemp.append(preprocess_input(reverse_color(image,i,j)*255))
                #blacktemp.append(preprocess_input(black(image,i,j)*255))
                #whitetemp.append(preprocess_input(white(image,i,j)*255))
                #shuffletemp.append(preprocess_input(shuffle_pixel(image,i,j)*255))
                #gt = (gauss_noise(image,i,j,ratio=1.0,var=0.8))
                #rt = (reverse_color(image,i,j))
                #bt = (black(image,i,j))
                #wt = (white(image,i,j))
                #st = (shuffle_pixel(image,i,j))
                #gausstemp.append(data_proprecessing(exp_id)(gt))
                #reversetemp.append(data_proprecessing(exp_id)(rt))
                #blacktemp.append(data_proprecessing(exp_id)(bt))
                #whitetemp.append(data_proprecessing(exp_id)(wt))
                #shuffletemp.append(data_proprecessing(exp_id)(st))
                
                if perturbate_type == 'gauss':
                   tt = gauss_noise(image,i,j,ratio=1.0,var=0.01)
                elif perturbate_type == 'white':
                   tt = white(image,i,j)
                elif perturbate_type == 'black':
                   tt = black(image,i,j)
                elif perturbate_type == 'reverse':
                   tt = reverse_color(image,i,j)
                elif perturbate_type == 'shuffle':
                   tt = shuffle_pixel(image,i,j)
                tt_temp.append(data_proprecessing(exp_id)(tt))
                    
                    
                    
        #np.save(os.path.join(basedir, 'gauss',str(image_id) + '.npy'), np.array(gausstemp).reshape(-1,32,32,3))
        #np.save(os.path.join(basedir, 'reverse', str(image_id) + '.npy'), np.array(reversetemp).reshape(-1,32,32,3))
        #np.save(os.path.join(basedir, 'black',str(image_id) + '.npy'), np.array(blacktemp).reshape(-1,32,32,3))
        #np.save(os.path.join(basedir, 'white', str(image_id) + '.npy'), np.array(whitetemp).reshape(-1,32,32,3))
        #np.save(os.path.join(basedir, 'shuffle', str(image_id) + '.npy'), np.array(shuffletemp).reshape(-1,32,32,3))

        np.save(os.path.join(basedir, str(perturbate_type), str(image_id) + '.npy'), np.array(tt_temp).reshape(-1, 32, 32, 3))
        image_id += 1
        #del gausstemp
        #del reversetemp
        #del blacktemp
        #del whitetemp
        #del shuffletemp
        if image_id%1000 == 0:
            print(str(image_id))
            print(time.time()-start)
    print(time.time()-start)
    print('finish generating...')
