import keras
import numpy as np
#from driving_utils import load_test_data
from keras.applications.vgg19 import VGG19
import pandas as pd

    

def get_combine_cifar10(**kwargs):
    ft = np.load('../data/cifar10_combined_bim_validation_vgg16.npz')
    X_test = ft['arr_0']
    Y_test = ft['arr_1']
    return X_test, Y_test


def get_combine_cifar100(**kwargs):
    ft = np.load('../data/cifar100_combined_jsma_test.npz')
    X_test = ft['arr_0']
    Y_test = ft['arr_1']
    return X_test, Y_test


def get_cifar10(**kwargs):
    from keras.datasets import cifar10
    subtract_pixel_mean = False
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    # Normalize data.
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean
    Y_test = keras.utils.to_categorical(Y_test, 10)
    return X_test, Y_test


def get_cifar10_vgg16(**kwargs):
    from keras.datasets import cifar10
    subtract_pixel_mean = False
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Normalize data.
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean
    Y_test = keras.utils.to_categorical(Y_test, 10)
    return X_test, Y_test


def get_cifar100(**kwargs):
    from keras.datasets import cifar100
    subtract_pixel_mean = False
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()

    # Normalize data.
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean
    Y_test = keras.utils.to_categorical(Y_test, 100)
    return X_test, Y_test


def get_imagenet(**kwargs):
    ft = np.load('../data/imagenet5000_test.npz')
    X_test = ft['arr_0']
    Y_test = ft['arr_1']
    return X_test, Y_test


def get_adversarial_cifar10(**kwargs):
    adversarial = 'fgsm'
    X_adv = np.load('data/cifar10_combined_10000_image_' + str(adversarial) + '.npy')
    Y_adv = np.load('data/cifar10_combined_10000_label_' + str(adversarial) + '.npy')
    for i in range(len(Y_adv)):
        if type(Y_adv[i]) is np.ndarray:
            Y_adv[i] = int(Y_adv[i][0])
        else:
            Y_adv[i] = int(Y_adv[i])
    return X_adv, Y_adv


def get_mnist(**kwargs):
    from keras.datasets import mnist
    (_, _), (X_test, Y_test) = mnist.load_data()
    X_test = X_test.astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test /= 255
    # Y_test = keras.utils.to_categorical(Y_test, 10)
    return X_test, Y_test


def get_combine_mnist(**kwargs):
    ft = np.load('data/mnist_combined_bim_validation.npz')
    X_test = ft['arr_0']
    Y_test = ft['arr_1']
    # Y_test = keras.utils.to_categorical(Y_test, 10)
    return X_test, Y_test


def get_usps(**kwargs):
    X_test = np.load('../data/usps_x.npy')
    X_test = X_test.astype('float32') / 255
    Y_test = np.load('../data/usps_y.npy')
    Y_test -= 1
    return X_test, Y_test


def get_mnist_16(**kwargs):
    from keras.datasets import mnist
    from PIL import Image
    (_, _), (X_test, Y_test) = mnist.load_data()
    X_test = X_test.astype('float32')
    new_X_test = []
    for i in range(10000):
        img = Image.fromarray(np.uint8(X_test[i]))
        new_X_test.append(np.asarray(img.resize((16, 16))))
    new_X_test = np.asarray(new_X_test)
    new_X_test = new_X_test.reshape(X_test.shape[0], 16, 16, 1)
    new_X_test = new_X_test.astype('float32')
    new_X_test /= 255
    Y_test = keras.utils.to_categorical(Y_test, 10)
    return new_X_test, Y_test
    
def get_driving(**kwargs):
    #data,length = load_test_data(batch_size=5614)
    ft=np.load('data/driving_test.npz')
    data=ft['arr_0']
    length=ft['arr_1']

    return data,length
    
def get_cifar100_vgg19(**kwargs):
    from keras.datasets import cifar100
    (_, _), (X_test, Y_test) = cifar100.load_data()
    X_test = X_test.astype('float32')
    X_test /= 255.
    # X_test = X_test.reshape(-1,32,32,3)
    return X_test, Y_test    


def get_sst5(**kwargs):
    X_test = np.load('../data/sst5_validation_x_char.npy')
    Y_test = np.load('../data/sst5_validation_y_char.npy')
    return X_test, Y_test 
    

def get_imdb(**kwargs):
    X_test = np.load('../data/imdb_test_x.npy')
    Y_test = np.load('../data/imdb_test_y.npy')
    return X_test, Y_test 

def get_combine_cifar100_vgg19(**kwargs):
    ft = np.load('../data/cifar100_validation_combined_jsma_vgg19.npz')
    X_test = ft['arr_0']
    Y_test = ft['arr_1']
    return X_test, Y_test 
    
def get_kddcup99(**kwargs):
    X_test = pd.read_csv('../data/test_x3.csv').values
    Y_test = pd.read_csv('../data/test_y3.csv', header=None).values
    Y_test = np.argmax(Y_test,axis=1)
    return X_test,Y_test

def get_T(**kwargs):
    X_test = np.load('../data/T_test_x.npy')
    Y_test = np.load('../data/T_test_y.npy')
    return X_test,Y_test


def get_pie9(**kwargs):
    X_test = np.load('../data/x_pie9_test.npy')
    Y_test = np.load('../data/y_pie9_test.npy')
    X_test = X_test.reshape(-1, 32, 32, 1)
    return X_test,Y_test
    
def get_pie5(**kwargs):
    X_test = np.load('../data/x_pie5_test.npy')
    Y_test = np.load('../data/y_pie5_test.npy')
    X_test = X_test.reshape(-1, 32, 32, 1)
    return X_test,Y_test
    
    
def get_coil(**kwargs):
    X_test = np.load('../data/x_coil2_test.npy')
    Y_test = np.load('../data/y_coil2_test.npy')
    X_test = X_test.reshape(-1, 32, 32, 1)
    return X_test,Y_test
    
    
def get_Tcl4(**kwargs):
    X_test = np.load('../data/T_4cl_validation_x.npy')
    Y_test = np.load('../data/T_4cl_validation_y.npy')
    return X_test,Y_test


def get_Tcl8(**kwargs):
    X_test = np.load('../data/T_8cl_validation_x.npy')
    Y_test = np.load('../data/T_8cl_validation_y.npy')
    return X_test,Y_test


def get_cifar10_vgg16_example_test(**kwargs):
    from keras.datasets import cifar10
    (_, _), (X_test, Y_test) = cifar10.load_data()
    # Normalize data.
    #X_test = X_test.astype('float32') / 255
    #Y_test = keras.utils.to_categorical(Y_test, 10)
    return X_test[0:100], Y_test[0:100]


def get_cifar10_vgg16_example_validation(**kwargs):
    from keras.datasets import cifar10
    (_, _), (X_test, Y_test) = cifar10.load_data()
    # Normalize data.
    #X_test = X_test.astype('float32') / 255
    #Y_test = keras.utils.to_categorical(Y_test, 10)
    return X_test[0:100], Y_test[0:100]


def get_data(exp_id):
    exp_model_dict = {
                      'cifar10_vgg16_example_test':get_cifar10_vgg16_example_test,
                      'cifar10_vgg16_example_validation': get_cifar10_vgg16_example_validation,
                      'lenet1': get_mnist,
                      'lenet4': get_mnist,
                      'lenet5': get_mnist,
                      'cifar10': get_cifar10,
                      'cifar100': get_cifar100,
                      'adv_cifar10': get_adversarial_cifar10,
                      'combine_cifar10': get_combine_cifar10,
                      'combine_cifar100': get_combine_cifar100,
                      'imagenet': get_imagenet,
                      'vgg16_cifar10': get_cifar10_vgg16,
                      'mnist_m1': get_mnist,
                      'mnist_m2': get_mnist,
                      'mnist_m3': get_mnist,
                      'combine_mnist': get_combine_mnist,
                      'usps': get_usps,
                      'mnist': get_mnist,
                      'mnist_transfer': get_mnist_16,
                      'driving':get_driving,
                      'cifar100_vgg19':get_cifar100_vgg19,
                      'sst5_bilstm':get_sst5,
                      'imdb_bilstm':get_imdb,
                      'combine_cifar100_vgg19':get_combine_cifar100_vgg19,
                      'kddcup99':get_kddcup99,
                      'T':get_T,
                      'pie9':get_pie9,
                      'pie5':get_pie5,
                      'coil':get_coil,
                      'Tcl4':get_Tcl4,
                      'Tcl8':get_Tcl8}
    return exp_model_dict[exp_id](exp_id=exp_id)


def get_model(exp_id):
    if exp_id == 'cifar10':
        origin_model_path = 'models/model_cifar10_resnet20.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'cifar10_vgg16_example_test' or 'cifar10_vgg16_example_validation':
        origin_model_path = 'models/cifar10_vgg16.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'adv_cifar10':
        origin_model_path = 'models/model_cifar10_resnet20.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'imagenet':
        origin_model = VGG19(weights='imagenet', include_top=True)
    elif exp_id == 'lenet5':
        origin_model_path = 'models/LeNet-5.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'cifar100':
        origin_model_path = 'models/model_cifar100.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'combine_cifar10':
        origin_model_path = 'models/cifar10_vgg16.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'combine_cifar100':
        origin_model_path = 'models/model_cifar100.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'vgg16_cifar10':
        origin_model_path = 'models/cifar10_vgg16.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'mnist_m1':
        origin_model_path = 'models/mutant1.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'mnist_m2':
        origin_model_path = 'model/mutant2.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'mnist_m3':
        origin_model_path = 'models/mutant3.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'combine_mnist':
        origin_model_path = 'models/LeNet-5.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'usps':
        origin_model_path = 'models/trans_lenet5_usps.hdf5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'mnist_transfer':
        origin_model_path = 'models/trans_lenet5_usps.hdf5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'cifar100_vgg19':
        origin_model_path = 'models/vgg19_cifar100_new.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'driving':
        #origin_model_path = 'models/model1_driving_orig.h5'
        from driving_models import Dave_orig
        origin_model = Dave_orig(input_tensor=None, load_weights=True)
        #origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'sst5_bilstm':
        origin_model_path = 'models/sst5_bilstm_char.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'imdb_bilstm':
        origin_model_path = 'models/imdb_bilstm.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'combine_cifar100_vgg19':
        origin_model_path = 'models/cifar100_vgg19.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'kddcup99':
        origin_model_path = 'models/kddcup99_model.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'T':
        origin_model_path = 'models/T_lstm_model.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'pie9':
        origin_model_path = 'models/pie27-9_vgg.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'pie5':
        origin_model_path = 'models/pie27-5_vgg0.72.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'coil':
        origin_model_path = 'models/coil_vgg0.87.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'Tcl4':
        origin_model_path = 'models/T_4cl_lstm_model.h5'
        origin_model = keras.models.load_model(origin_model_path)
    elif exp_id == 'Tcl8':
        origin_model_path = 'models/T_8cl_lstm_model.h5'
        origin_model = keras.models.load_model(origin_model_path)
    return origin_model
    
def preprocess_cifar100_vgg19(test_images):
    mean = [0.5073615, 0.48668972, 0.44108823]
    std = [0.2674881, 0.25659335, 0.27630848]
    test_images = test_images.reshape(-1,32,32,3)
    # test_images = (test_images - mean) / std
    for i in range(test_images.shape[-1]):
        test_images[ :,:, :, i] = (test_images[ :,:, :, i] - mean[i]) / std[i]
    # test_images = test_images.reshape(32,32,3)
    return test_images

def preprocess_kddcup99(test_images):
    test_images = test_images.reshape(test_images.shape[0], 41, 1)
    return test_images

def return_test_images(test_images):
    test_images = test_images.astype('float32') / 255
    return test_images
    
def data_proprecessing(exp_id):
    if exp_id == 'cifar100_vgg19':
        return preprocess_cifar100_vgg19
    elif exp_id == 'kddcup99':
        return preprocess_kddcup99
    else:
        return return_test_images
    
