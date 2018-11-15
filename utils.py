
from keras.models import model_from_json
from keras.models import load_model
import numpy as np
import cv2

def import_data(file_path):
    full_array = np.load('image_array.npy')
    return full_array

def bgr_to_gray(arr):
    gray_array = np.zeros([arr.shape[0], arr.shape[1], arr.shape[2], 1])
    for i in range(arr.shape[0]):
        gray_array[i,:,:,:] = cv2.cvtColor(arr[i,:,:,:], cv2.COLOR_BGR2GRAY)
    return gray_array

def rescale_to_1(arr):
    return arr.astype('float32') / 255.

def test_train_split(arr, test_fraction=0.1):
    num_images = arr.shape[0]
    n_test_images = round(num_images(test_fraction))
    n_train_images = num_images - n_test_images
    idx = random.sample(range(0, num_images, num_images))
    x_train = full_array[idx[:n_train_images], :, :, :]
    x_test = full_array[idx[n_train_images:], :, :, :]
    return x_train, x_test

def reshape_4d_to_2d(arr):
    return arr.reshape(arr.shape[0], arr.shape[1]*arr.shape[2]*arr.shape[3])

def save_model(model, save_name='model'):
    # serialize model to JSON
    #  the keras model which is trained is defined as 'model' in this example
    model_json = model.to_json()
    with open(save_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(save_name + ".h5")
    print('Saved model')

def load_model(model_name):
    # load json and create model
    json_file = open(model_name+'.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(model_name + ".h5")
    print("Loaded model from disk")
    return loaded_model
