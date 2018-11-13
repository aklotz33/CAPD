import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
import cPickle

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import cifar10


# from here
#https://github.com/chaitanya100100/VAE-for-Image-Generation/blob/master/src/cifar10_train.py

# import parameters
from capd_image_params import *

# tensorflow uses channels_last
# theano uses channels_first
if K.image_data_format() == 'channels_first':
    original_img_size = (IMG_CHANNELS, IMG_ROWS, IMG_COLS)
else:
    original_img_size = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)


# encoder architecture
x = Input(shape=original_img_size)
conv_1 = Conv2D(IMG_CHANNELS,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(FILTERS,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(FILTERS,
                kernel_size=NUM_CONV,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(FILTERS,
                kernel_size=NUM_CONV,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(INTERMEDIATE_DIM, activation='relu')(flat)

# mean and variance for latent variables
z_mean = Dense(LATENT_DIM)(hidden)
z_log_var = Dense(LATENT_DIM)(hidden)

# sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], LATENT_DIM),
                              mean=0., stddev=EPSILON_STD)
    return z_mean + K.exp(z_log_var) * epsilon

z = Lambda(sampling, output_shape=(LATENT_DIM,))([z_mean, z_log_var])


# decoder architecture
decoder_hid = Dense(INTERMEDIATE_DIM, activation='relu')
decoder_upsample = Dense(FILTERS * IMG_ROWS / 2 * IMG_COLS / 2, activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (BATCH_SIZE, FILTERS, IMG_ROWS / 2, IMG_COLS / 2)
else:
    output_shape = (BATCH_SIZE, IMG_ROWS / 2, IMG_COLS / 2, FILTERS)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(FILTERS,
                                   kernel_size=NUM_CONV,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(FILTERS,
                                   kernel_size=NUM_CONV,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_3_upsamp = Conv2DTranspose(FILTERS,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(IMG_CHANNELS,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = IMG_ROWS * IMG_COLS * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        return x

y = CustomVariationalLayer()([x, x_decoded_mean_squash])

# entire model
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()


# load dataset
""" (x_train, _), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)) """

full_array = np.load('image_array.npy')
num_images = full_array.shape[0]
# 90 10 train val split
n_train_images = round(num_images(0.9))
idx = random.sample(range(0, num_images, num_images))
x_train = full_array[idx[:n_train_images], :, :, :]
x_test = full_array[idx[n_train_images:], :, :, :]


# training
history = vae.fit(x_train,
        shuffle=True,
        EPOCHS=EPOCHS,
        BATCH_SIZE=BATCH_SIZE,
        validation_data=(x_test, None))

# encoder from learned model
encoder = Model(x, z_mean)

# generator / decoder from learned model
decoder_input = Input(shape=(LATENT_DIM,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)

# save all 3 models for future use - especially generator
vae.save('../models/cifar10_ld_%d_conv_%d_id_%d_e_%d_vae.h5' % (LATENT_DIM, NUM_CONV, INTERMEDIATE_DIM, EPOCHS))
encoder.save('../models/cifar10_ld_%d_conv_%d_id_%d_e_%d_encoder.h5' % (LATENT_DIM, NUM_CONV, INTERMEDIATE_DIM, EPOCHS))
generator.save('../models/cifar10_ld_%d_conv_%d_id_%d_e_%d_generator.h5' % (LATENT_DIM, NUM_CONV, INTERMEDIATE_DIM, EPOCHS))

# save training history
fname = '../models/cifar10_ld_%d_conv_%d_id_%d_e_%d_history.pkl' % (LATENT_DIM, NUM_CONV, INTERMEDIATE_DIM, EPOCHS)
with open(fname, 'wb') as file_pi:
    cPickle.dump(history.history, file_pi)