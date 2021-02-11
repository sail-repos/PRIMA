import time
import os
import sys
import datetime
import numpy as np
import keras
import math
import tensorflow as tf
from keras.models import Model
import random
from keras.datasets import mnist
from numpy import arange
import argparse
from keras.applications import vgg19,resnet50
import pickle
from datautils import get_data,get_model,data_proprecessing
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
#from keras.backend.tensorflow_backend import set_session
#import random
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
#tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

def walkFile(file):
    file_list = []
    for root, dirs, files in os.walk(file):
        for f in files:
            file_list.append(os.path.join(root, f))
    return file_list

def count_wrong_prediction(given_list):
    set01 = set(given_list)
    dict01 = {}
    for item in set01:
        dict01.update({item: given_list.count(item)})
    return dict01


def save_dict(filename,dictionary):
    dictfile = open(filename + '.dict', 'wb')
    pickle.dump(dictionary, dictfile)
    dictfile.close()


if __name__=="__main__":
    # exp_id = 'vgg19'
    import time
    start = time.time()
    basedir = os.path.abspath(os.path.dirname(__file__))
    """Parser of command args"""
    parse = argparse.ArgumentParser()
    parse.add_argument("--exp_id", type=str, help="exp_identifiers")
    parse.add_argument("--ptype", type=str, help="GF//NEB//NAI//WS")
    parse.add_argument("--mutants_num", type=int, help="100")
    exp_id = sys.argv[1]
    ptype = sys.argv[2]
    sample = len(get_data(exp_id)[0])
    file_name = 'model_perturbation_'+str(exp_id)+'_'+str(ptype)
    basedir = os.path.dirname(__file__)
    basedir = os.path.join(basedir, 'model')
    basedir = os.path.join(basedir, exp_id)
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    file_name = os.path.join(basedir, file_name)
    kill_num_dict = {i: 0 for i in range(int(float(sample)))}
    save_dict(dictionary=kill_num_dict,filename=file_name)
    
    model_save_path = 'perturbated_'+ptype
    model_save_path = os.path.join(basedir, model_save_path)
    file_list = [model_save_path+'/'+str(i)+'.h5' for i in range(3)]
    file_id = 0
    for file in file_list:
        # print(file_name,file)
        os.system("python core_prioritization_unit.py %s %s %s %s %s" %(file_name,file,str(file_id),exp_id,ptype))
        file_id += 1
    print(time.time()-start)
