#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:33:13 2019

@author: akshay
"""
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
# from keras.datasets import mnist
from vae_model import encoder_decoder, encoder_decoder_conv
import matplotlib.pyplot as plt
import numpy as np
from loss_vae import loss, recon_loss
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from plot_results import plot_results
from skimage.transform import resize
from tensorflow.keras.callbacks import ReduceLROnPlateau
# from tensorflow.keras import objectives
from sklearn.utils import class_weight
import os
from unittest.mock import MagicMock

from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from numpy.random import seed

seed(1)


# MNIST dataset
def scheduler(epoch, n_ep, init_lr):
    m = init_lr * pow((1 - (epoch / n_ep)), 1.5)
    print(m)
    return m


def schedule_lr(n_ep, init_lr):
    def scheduler(epoch):
        m = init_lr * pow((1 - (epoch / n_ep)), 1.5)
        print(m)
        return m

    return scheduler


size_image = (224, 224, 1)

# build the model
encoder_model, decoder_model, encoder_decoder_model = encoder_decoder(size_image, 2)

# fix the loss
encoder_decoder_model.summary()

opt = Adam(lr=0.001)
batch_size = 64
encoder_decoder_model.compile(loss=[recon_loss((160, 160)), loss((batch_size,) + size_image)], optimizer=opt)

x_train = np.load('x_train.py')
x_val = np.load('x_val.py')
x_test = np.load('x_test.py')
y_train = np.load('y_train.py')
y_val = np.load('y_val.py')
y_test = np.load('y_test.py')

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(np.argmax(y_train, axis=-1)),
                                                  np.argmax(y_train, axis=-1))
print('num labels is ', y_train.shape)
print('the mean is ', np.mean(x_train))
print('the number of nans is ', np.count_nonzero(np.isnan(x_train)))

z_dummy = np.zeros((x_train.shape[0], 512, 2))
z_dummy_val = np.zeros((x_val.shape[0], 512, 2))
z_dummy_test = np.zeros((x_test.shape[0], 512, 2))

"""
early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=1e-2,
                           patience=10,
                           mode='auto',
                           restore_best_weights=True)
"""


for i in range(10):
    print('the learning rate is ', K.eval(encoder_decoder_model.optimizer.lr))
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(schedule_lr(200, K.eval(encoder_decoder_model.optimizer.lr)))
    encoder_decoder_model.fit(x_train, [x_train, z_dummy], epochs=20, batch_size=batch_size,
                              validation_data=[x_val, [x_val, z_dummy_val]], shuffle=True, callbacks=[reduce_lr])

    encoder_decoder_model.save_weights('vae_chest.h5')
    decoder_model.save_weights('vae_decoder_chest.h5')
    encoder_model.save_weights('vae_encoder_chest.h5')

    plot_results((encoder_model, decoder_model),
                 (x_test, np.argmax(y_test, axis=-1), x_train, np.argmax(y_train, axis=-1)), batch_size=128,
                 model_name="vae_mlp", epoch=i)
