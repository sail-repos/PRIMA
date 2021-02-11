import os
import time
import sys
import datetime
import numpy as np
import keras
from keras.applications.vgg19 import VGG19
import math
from keras.models import Model
import random
from keras.datasets import mnist
from numpy import arange
import argparse
from keras.applications import vgg19,resnet50
from keras.applications.vgg19 import preprocess_input
import re
from datautils import get_data,get_model,data_proprecessing

import pandas as pd
exp_id = sys.argv[1]
ptype = sys.argv[2]
samples = len(get_data(exp_id)[0])
# print(samples)
# samples = int(float(sys.argv[3]))
import tensorflow as tf


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from keras.backend.tensorflow_backend import set_session
import random
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
if __name__=="__main__":
    # 3
    import time
    start_ = time.time()
    basedir = os.path.dirname(__file__)
    basedir = os.path.join(basedir, 'input')
    basedir = os.path.join(basedir, exp_id)
        
    predicting_file_path = os.path.join(basedir, 'predict_probability_vector_'+str(exp_id)+'.npy')
    X_test,Y_test = get_data(exp_id)
    X_test = data_proprecessing(exp_id)(X_test)
    origin_model = get_model(exp_id)
    if not os.path.exists(predicting_file_path):
        a = origin_model.predict(X_test)
        # a = np.argmax(a, axis=1)
        np.save(predicting_file_path,a)
        ori_prob = a
    else:
        ori_prob = np.load(predicting_file_path)
        
    file_name = 'image_perturbation_'+exp_id+'_'+ptype
    file_name = os.path.join(basedir, file_name)
    result_recording_file = open(file_name+'.txt', 'w')
    origin_model_temp_result = ori_prob
    origin_model_result = np.argmax(origin_model_temp_result, axis=1)
    print('origin_prediction:',origin_model_result)
    result_recording_file.write(str(origin_model_result))
    kill_num_dict = {}
    eu_distance_dict,cos_distance_dict,mahat_distance_dict,qube_distance_dict={},{},{},{}
    abs_eu_distance_dict, abs_cos_distance_dict, abs_mahat_distance_dict, abs_qube_distance_dict = {}, {}, {}, {}
    
    perturbate_image_path = os.path.join(basedir,exp_id,ptype)
    nlp_tasks = ['imdb_bilstm','sst5_bilstm','trec_bilstm','spam_bilstm']
    form_tasks = ['kddcup99']
    if exp_id in nlp_tasks+form_tasks:
        file_list = [os.path.join(perturbate_image_path,str(i)+'.csv') for i in range(samples)]
    else:
        file_list = [os.path.join(perturbate_image_path,str(i)+'.npy') for i in range(samples)]
    image_id = 0
    for file in file_list:
        start = datetime.datetime.now()
        
        if exp_id in nlp_tasks:
            x_all = pd.read_csv(file,header=None,names=['review'])
            x_all = x_all.review
            x_all = data_proprecessing(exp_id)(x_all)
        elif exp_id in form_tasks:
            x_all = pd.read_csv(file).values
            x_all = data_proprecessing(exp_id)(x_all)
        else:
            x_all = np.load(file)

        my_model = origin_model
        temp_result = my_model.predict(x_all)
        spath = str(exp_id)+'_'+ptype+'_prob'
        if not os.path.exists(os.path.join(basedir,spath)):
            os.mkdir(os.path.join(basedir,spath))
        spath = os.path.join(basedir, spath)
        np.save(os.path.join(spath,str(image_id)+'.npy'),temp_result)
        result = np.argmax(temp_result, axis=1)
        kill_num = 0
        for r in result:
            if r != origin_model_result[image_id]:
                kill_num += 1

        kill_num_dict.update({image_id: kill_num})
        print('image_id:'+str(image_id))
        print('kill_rate:',kill_num)
        result_recording_file.write('image_id:'+str(image_id))
        result_recording_file.write('\n')
        result_recording_file.write('kill_num:'+str(kill_num))
        result_recording_file.write('\n')
        image_id+=1


    d2 = sorted(kill_num_dict.items(), key=lambda x: x[1], reverse=True)
    kill_num_dict = {score: letter for score, letter in d2}
    import pickle
    dictfile = open(file_name+'.dict', 'wb')
    pickle.dump(kill_num_dict, dictfile)
    dictfile.close()

    result_recording_file.close()
    print(time.time()-start_)