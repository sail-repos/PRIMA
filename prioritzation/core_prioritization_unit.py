import keras
import datetime
import tensorflow as tf
import numpy as np
from keras.applications import vgg19,resnet50
import math
import pickle
import sys
import os
from datautils import get_data,get_model,data_proprecessing
from keras.applications.vgg19 import preprocess_input
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    
def save_dict(filename,dictionary):
    dictfile = open(filename + '.dict', 'wb')
    pickle.dump(dictionary, dictfile)
    dictfile.close()


def load_dict(filename):
    dictfile = open(filename + '.dict', 'rb')
    a = pickle.load(dictfile)
    dictfile.close()
    return a

file_name = sys.argv[1]
file = sys.argv[2]
file_id = int(float(sys.argv[3]))
exp_id = sys.argv[4]
ptype = sys.argv[5]
from keras import backend as K
basedir = os.path.dirname(__file__)
basedir = os.path.join(basedir, 'model')
basedir = os.path.join(basedir, exp_id)

predicting_file_path = os.path.join(basedir, 'predict_probability_vector_'+str(exp_id)+'.npy')
X_test,Y_test = get_data(exp_id)
X_test = data_proprecessing(exp_id)(X_test)
origin_model = get_model(exp_id)
if not os.path.exists(predicting_file_path):
  a = origin_model.predict(X_test)
  np.save(predicting_file_path,a)
  origin_model_result = a
else:
  origin_model_result = np.load(predicting_file_path)
  
origin_model_result = np.argmax(origin_model_result, axis=1)

kill_num_dict = load_dict(file_name)
result_recording_file = open(file_name + '.txt', 'a')
start = datetime.datetime.now()
my_model = keras.models.load_model(file)
print('file:', file)
result_recording_file.write(str('file:' + str(file)))
result_recording_file.write('\n')
temp_result = my_model.predict(X_test)
new_name = str(file_id)
savepath = os.path.join(basedir, exp_id+'_temp_result_'+ptype)
if not os.path.exists(savepath):
    os.mkdir(savepath)
np.save(savepath+'/'+new_name+'.npy',temp_result)
result = np.argmax(temp_result, axis=1)
my_model = keras.models.load_model(file)
wrong_predict = []
for count in range(len(X_test)):
    if result[count] != origin_model_result[count]:
        kill_num_dict[count] += 1
        wrong_predict.append(count)

result_recording_file.write('diff_num:' + str(len(wrong_predict)))
result_recording_file.write('\n')
result_recording_file.write('different_pred:' + str(wrong_predict))
result_recording_file.write('\n')


K.clear_session()

elapsed = (datetime.datetime.now() - start)
print("Time used: ", elapsed)
result_recording_file.close()
save_dict(dictionary=kill_num_dict,filename=file_name)