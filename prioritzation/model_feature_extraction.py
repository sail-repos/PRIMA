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

    
if __name__ == '__main__':

    exp_id = sys.argv[1]
    ptype = sys.argv[2]
    sample = len(get_data(exp_id)[0])

    basedir = os.path.dirname(__file__)
    basedir = os.path.join(basedir, 'model')
    basedir = os.path.join(basedir, exp_id)
    
    predicting_file_path = os.path.join(basedir, 'predict_probability_vector_'+str(exp_id)+'.npy')
    X_test,Y_test = get_data(exp_id)
    origin_model = get_model(exp_id)
    X_test = data_proprecessing(exp_id)(X_test)
    if not os.path.exists(predicting_file_path):
        a = origin_model.predict(X_test)
        # a = np.argmax(a, axis=1)
        np.save(predicting_file_path,a)
        ori_prob = a
    else:
        ori_prob = np.load(predicting_file_path)

    file_name = 'image_perturbation_'+exp_id+'_'+ptype
    result = np.argmax(ori_prob, axis=1)

    samples = int(float(sample))
    file_name = exp_id+'_'+ptype+'_feature'
    file_name = os.path.join(basedir, file_name)
    euler = [0 for i in range(samples)]
    mahat = [0 for i in range(samples)]
    qube = [0 for i in range(samples)]
    cos = [0 for i in range(samples)]
    difference = [0 for i in range(samples)]
    cos_list = [[] for i in range(samples)]
    different_class = [[] for i in range(samples)]

    for i in range(0, 3):

        file_path = exp_id+'_temp_result_'+ptype+'/' + str(i) + '.npy'
        file_path = os.path.join(basedir, file_path)

        perturbated_prediction = np.load(file_path)
        for ii in range(samples):
            pro = perturbated_prediction[ii]
            opro = ori_prob[ii]
            max_value_pos = np.argmax(opro)
            max_value = np.max(opro)
            difference[ii] += abs(max_value - pro[max_value_pos])
            euler[ii] += np.linalg.norm(pro - opro)
            mahat[ii] += np.linalg.norm(pro - opro, ord=1)
            qube[ii] += np.linalg.norm(pro - opro, ord=np.inf)
            co = (1 - (np.dot(pro, opro.T) / (np.linalg.norm(pro) * (np.linalg.norm(opro)))))
            if co < 0:
                co = 0
            elif co > 1:
                co = 1
            cos[ii] += co
            cos_list[ii].append(co)

            if np.argmax(pro) != max_value_pos:
                different_class[ii].append(np.argmax(pro))

    result_recording_file = open(file_name + '.txt', 'a+')
    for i in range(samples):

        dic = {}
        for key in different_class[i]:
            dic[key] = dic.get(key, 0) + 1
        wrong_class_num = len(dic)
        if len(dic)>0:
            max_class_num = max(dic.values())
        else :
            max_class_num = 0
        cos_dis = cos_distribution(cos_list[i])
        print('id:', i)
        #print('euler:', euler[i])
        #print('mahat:', mahat[i])
        #print('qube:', qube[i])
        #print('cos:', cos[i])
        result_recording_file.write('image_id:' + str(i))
        result_recording_file.write('\n')
        result_recording_file.write('euler:' + str(euler[i]))
        result_recording_file.write('\n')
        result_recording_file.write('mahat:' + str(mahat[i]))
        result_recording_file.write('\n')
        result_recording_file.write('qube:' + str(qube[i]))
        result_recording_file.write('\n')
        result_recording_file.write('cos:' + str(cos[i]))
        result_recording_file.write('\n')
        result_recording_file.write('difference:' + str(difference[i]))
        result_recording_file.write('\n')
        result_recording_file.write('wnum:' + str(wrong_class_num))
        result_recording_file.write('\n')
        result_recording_file.write('num_mc:' + str(max_class_num))
        result_recording_file.write('\n')
        result_recording_file.write('fenbu:' + str(cos_dis))
        result_recording_file.write('\n')
    result_recording_file.close()