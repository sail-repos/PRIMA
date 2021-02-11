import numpy as np
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
import os
import keras
import sys
from datautils import get_data,get_model,data_proprecessing
def cos_distribution(cos_array):
    cos_distribute = [0 for i in range(10)]
    for i in cos_array:
        if i >= 0 and i < 0.1:
            cos_distribute[0] += 1
        elif i >= 0.1 and i < 0.2:
            cos_distribute[1] += 1
        elif i >= 0.2 and i < 0.3:
            cos_distribute[2] += 1
        elif i >= 0.3 and i < 0.4:
            cos_distribute[3] += 1
        elif i >= 0.4 and i < 0.5:
            cos_distribute[4] += 1
        elif i >= 0.5 and i < 0.6:
            cos_distribute[5] += 1
        elif i >= 0.6 and i < 0.7:
            cos_distribute[6] += 1
        elif i >= 0.7 and i < 0.8:
            cos_distribute[7] += 1
        elif i >= 0.8 and i < 0.9:
            cos_distribute[8] += 1
        elif i >= 0.9 and i <= 1.0:
            cos_distribute[9] += 1
    return cos_distribute

exp_id = sys.argv[1]
ptype = sys.argv[2]
samples = len(get_data(exp_id)[0])

if __name__ == '__main__':
    #2
    #origin_model = get_model(exp_id)
    #X_test,_ = get_data(exp_id)
    #ori_prob = origin_model.predict(X_test)
    
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
    
    # ori_prob = np.load('predict_prob_resnet20_cifar10.npy')
    # ori_prob = np.load('origin_model_temp_result.npy')
    result = np.argmax(ori_prob, axis=1)
    # np.save('vgg19_random_predict.npy',ori_prob)
    file_name = exp_id+'_'+ptype+'_feature'
    file_name = os.path.join(basedir, file_name)
    prob_path = exp_id+'_'+ptype+'_prob'
    prob_path = os.path.join(basedir,prob_path)
    for i in range(0,samples):
        a = ori_prob[i]
        max_value = np.max(a)
        max_value_pos = np.argmax(a)
        file_path = os.path.join(prob_path,str(i)+'.npy')
        #if not os.path.exists(file_path):
            #continue
        perturbated_prediction = np.load(file_path)
        result_recording_file = open(file_name + '.txt', 'a+')
        euler = 0
        mahat = 0
        qube = 0
        cos = 0
        difference = 0
        different_class = []
        cos_list = []
        for pp in perturbated_prediction:
            pro = pp
            opro = a
            # if np.argmax(ii) != result[i]:
            difference += abs(max_value - pp[max_value_pos])
            euler += np.linalg.norm(pro - opro)
            mahat += np.linalg.norm(pro - opro, ord=1)
            qube += np.linalg.norm(pro - opro, ord=np.inf)
            co = (1 - (np.dot(pro, opro.T) / (np.linalg.norm(pro) * (np.linalg.norm(opro)))))
            if co < 0:
                co = 0
            elif co > 1:
                co = 1
            cos += co
            cos_list.append(co)
            if np.argmax(pp) != max_value_pos:
                different_class.append(np.argmax(pp))
        cos_dis = cos_distribution(cos_list)
        # euler /= 256
        # mahat /= 256
        # qube /= 256
        # cos /= 256
        dic = {}
        for key in different_class:
            dic[key] = dic.get(key, 0) + 1
        wrong_class_num = len(dic)
        if len(dic)>0:
            max_class_num = max(dic.values())
        else :
            max_class_num = 0
        print('id:',i)
        print('euler:', euler)
        print('mahat:', mahat)
        print('qube:', qube)
        print('cos:', cos)
        print('difference:',difference)
        print('wnum:',wrong_class_num)
        print('num_mc:', max_class_num)
        print('fenbu:',cos_dis)
        result_recording_file.write('image_id:' + str(i))
        result_recording_file.write('\n')
        result_recording_file.write('euler:' + str(euler))
        result_recording_file.write('\n')
        result_recording_file.write('mahat:' + str(mahat))
        result_recording_file.write('\n')
        result_recording_file.write('qube:' + str(qube))
        result_recording_file.write('\n')
        result_recording_file.write('cos:' + str(cos))
        result_recording_file.write('\n')
        result_recording_file.write('difference:' + str(difference))
        result_recording_file.write('\n')
        result_recording_file.write('wnum:' + str(wrong_class_num))
        result_recording_file.write('\n')
        result_recording_file.write('num_mc:' + str(max_class_num))
        result_recording_file.write('\n')
        result_recording_file.write('fenbu:' + str(cos_dis))
        result_recording_file.write('\n')
        result_recording_file.close()