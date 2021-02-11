import pandas as pd
import numpy as np
import pickle
from datautils import get_data,get_model,data_proprecessing
import os
import sys

def read_kill_rate_dict(file_name):
    dictfile = open(file_name + '.dict', 'rb')
    kill_rate_file = pickle.load(dictfile)
    if type(kill_rate_file) == dict:
        kill_rate_dict = kill_rate_file
    else:
        kill_rate_dict = {score: letter for score, letter in kill_rate_file}
    return kill_rate_dict

exp_id = sys.argv[1]
sample = len(get_data(exp_id)[0])
ptypes =  sys.argv[2]

if __name__ == '__main__':

    if ptypes == 'input': 
        input_types = ['gauss','reverse','black','white','shuffle']
    elif ptypes == 'model':
        input_types = ['GF','NAI','NEB','WS']
    elif ptypes == 'nlp':
        input_types = ['vs','vr','vrp']
        ptypes = 'input'
    elif ptypes == 'form':
        input_types = ['ad']
        ptypes = 'input'
        
    
    all = []
    all_title = []
    basedir = os.path.dirname(__file__)
    basedir = os.path.join(basedir, ptypes)
    # basedir = ptypes
    basedir = os.path.join(basedir, exp_id)
    for mt in input_types:
        types = ptypes+'_'+str(mt)+'_'
        f = open(os.path.join(basedir,exp_id+'_'+mt+'_feature.txt'), 'r')
        if ptypes == 'input':
            kill_rate_dict = read_kill_rate_dict(os.path.join(basedir,'image_perturbation_'+exp_id+'_'+mt))
        else:
            kill_rate_dict = read_kill_rate_dict(os.path.join(basedir,'model_perturbation_'+exp_id+'_'+mt))
        a = f.readlines()
        sh = []
        cos = []
        difference = []
        wrong_class_num = []
        max_class_num = []
        cos_distribution = []
        for i in a:
            if i[0] == 'c':
                x = float(i[i.find(':') + 1:-1].strip())
                cos.append(x)
            elif i[0] == 'd':
                x = float(i[i.find(':') + 1:-1].strip())
                difference.append(x)
            elif i[0] == 'n':
                x = int(i[i.find(':') + 1:-1].strip())
                max_class_num.append(x)
            elif i[0] == 'w':
                x = int(i[i.find(':') + 1:-1].strip())
                wrong_class_num.append(x)
            elif i[0] == 'f':
                x = eval(i[i.find(':') + 1:-1].strip())
                cos_distribution.append(x)

        kill_num_list = []
        for i in range(sample):
            kill_num_list.append(kill_rate_dict[i])

        all_vector = []
        all_vector.append(kill_num_list)
        all_vector.append(cos)
        all_vector.append(difference)
        all_vector.append(max_class_num)
        all_vector.append(wrong_class_num)
        cd = list(np.asarray(cos_distribution).T)
        for i in range(10):
            all_vector.append(cd[i].tolist())

        title = [str(types)+'kill_num',str(types)+'cos',\
                 str(types)+'difference',\
                 str(types)+'max_class_num',\
                 str(types)+'wrong_class_num']
        title.extend([str(types)+'cos_distribution'+str(i) for i in range(10)])
        
        all_title.extend(title)
        all.extend(all_vector)
    for i in all:
        print(len(i))
    pd_data_all = pd.DataFrame(np.asarray(all).T,columns=all_title)
    
    X_test, Y_test = get_data(exp_id)
    y_predict_prob = np.load(os.path.join(basedir,'predict_probability_vector_'+str(exp_id)+'.npy'))
    y_predict = np.argmax(y_predict_prob,axis=1)
    right_or_wrong = []
    for i in range(sample):
        if Y_test[i] !=  y_predict[i]:
            right_or_wrong.append(0)
        else:
            right_or_wrong.append(1)
    rightness_pd = pd.DataFrame(np.array(right_or_wrong), columns=['rightness'])

    
    result = pd.concat( [pd_data_all,rightness_pd], axis=1 )
    result.to_csv(ptypes+'_'+str(exp_id)+'_feature.csv')