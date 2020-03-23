#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 09:36:41 2019

@author: akshay
"""

import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Input, Activation, Dense, Lambda, Reshape, BatchNormalization, Flatten, Dropout, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import regularizers


# From keras-team/keras/blob/master/examples/variational_autoencoder.py
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_var) * epsilon

def encoder_decoder(input_size,d=3):
    
    '''
    input_size size of the input should include channels in the end
    
    '''

    cf = 0.75
    
    if d==2:
        from tensorflow.keras.layers import Conv2D as ConvND
        from tensorflow.keras.layers import Conv2DTranspose as DeconvolutionND
        
    elif d==3:
        from tensorflow.keras.layers import Conv3D as ConvND
        from tensorflow.keras.layers import Conv3DTranspose as DeconvolutionND
        
    else:
        
        raise Exception('d should be 2 or 3')
        
    
    
    inputs = Input(shape=input_size)
    
    x = ConvND(int(32*cf),kernel_size=3,strides=2,padding='same')(inputs)
    x1 = Activation('relu', name='ac1')(x)
    #x1 = BatchNormalization()(x1)

    x2 = ConvND(int(64*cf),kernel_size=3,strides=2,padding='same')(x1)
    x2 = Activation('relu', name='ac2')(x2)
    #x2 = BatchNormalization()(x2)

    x3 = ConvND(int(128*cf),kernel_size=3,strides=2,padding='same')(x2)
    x3 = Activation('relu', name='ac3')(x3)
    #x3 = BatchNormalization()(x3)

    x4 = ConvND(int(256*cf),kernel_size=3,strides=2,padding='same')(x3)
    x5 = Activation('relu', name='ac4')(x4) 
    
    #x5 = ConvND(int(512*cf),kernel_size=3,strides=2,padding='same')(x4)
    #x5 = Activation('relu', name='ac5')(x5) 
    #x = Dropout(0.5)(x)
    
    #flatten part
    if d==3:
        _,H,W,D,c = K.int_shape(x5)
    elif d==2:
        _,H,W,c = K.int_shape(x5)
        
    
    x_up = Flatten()(x5)
    x_up = Dense(int(1024*cf),activation='relu')(x_up)
    #x_up = BatchNormalization()(x_up)
    #x_up = Dense(128,activation='relu')(x_up)
    
    z_mean = Dense(int(512*cf), name='z_mean')(x_up)
    z_log_var = Dense(int(512*cf), name='z_log_var')(x_up)
    z = Lambda(sampling, name='Dec_VAE_VDraw_Sampling')([z_mean, z_log_var])


    z_mean_reshape = Reshape((int(512*cf),1),name='r1')(z_mean)
    z_log_var_reshape = Reshape((int(512*cf),1),name='r2')(z_log_var)
    z_concatenate = Concatenate(axis=-1)([z_mean_reshape, z_log_var_reshape])
    
    encoder_model = Model(inputs,[z,z_concatenate])
    encoder_model.summary()
    
    
    ##decoder part
    latent_inputs = Input(shape=(int(512*cf),))
    if d==3:
        x_up = Dense((c) * (H) * (W) * (D))(latent_inputs)
    elif d==2:
        x_up = Dense((c) * (H) * (W))(latent_inputs)
    
    
    if d==3:
        x_up = Reshape((H,W,D,c),name='reshape')(x_up)
    elif d==2:
        x_up = Reshape((H,W,c),name='reshape')(x_up)
        
    
    x_up = DeconvolutionND(int(512*cf),kernel_size=4,strides=2,padding='same')(x_up)
    x_up = Activation('relu')(x_up)
    #x_up = Concatenate(axis=-1)([x_up,x3])
    #x_up = BatchNormalization()(x_up)
    
    x_up = DeconvolutionND(int(256*cf),kernel_size=4,strides=2,padding='same')(x_up)
    x_up = Activation('relu')(x_up)
    #x_up = Concatenate(axis=-1)([x_up,x2])
    #x_up = BatchNormalization()(x_up)
    
    x_up = DeconvolutionND(int(128*cf),kernel_size=4,strides=2,padding='same')(x_up)
    x_up = Activation('relu')(x_up)
    #x_up = Concatenate(axis=-1)([x_up,x1])
    #x_up = BatchNormalization()(x_up)

    x_up = DeconvolutionND(int(64*cf),kernel_size=4,strides=2,padding='same')(x_up)
    x_up = Activation('relu')(x_up)
    #x_up = BatchNormalization()(x_up)
    
    #x_up = DeconvolutionND(int(32*cf),kernel_size=4,strides=2,padding='same')(x_up)
    #x_up = Activation('relu')(x_up)
    
    x_up = DeconvolutionND(input_size[-1],kernel_size=4,strides=1,padding='same')(x_up)
    recon = Activation('sigmoid', name='recon')(x_up)

    print(recon)

    decoder_model = Model([inputs,latent_inputs],recon)
    encoder_decoder_model = Model(inputs,[decoder_model([inputs,encoder_model(inputs)[0]]), encoder_model(inputs)[1]])
    
    
    return encoder_model,decoder_model,encoder_decoder_model
    #return encoder_decoder_model
    
    
    
def encoder_decoder_conv(input_size,d=3):
    
    '''
    input_size size of the input should include channels in the end
    
    '''

    cf = 0.75
    
    if d==2:
        from tensorflow.keras.layers import Conv2D as ConvND
        from tensorflow.keras.layers import Conv2DTranspose as DeconvolutionND
        from tensorflow.keras.layers import MaxPooling2D as MaxPoolingND
        from tensorflow.keras.layers import UpSampling2D as UpSamplingND
        
    elif d==3:
        from tensorflow.keras.layers import Conv3D as ConvND
        from tensorflow.keras.layers import Conv3DTranspose as DeconvolutionND
        from tensorflow.keras.layers import MaxPooling3D as MaxPoolingND
        from tensorflow.keras.layers import UpSampling3D as UpSamplingND
        
    else:
        
        raise Exception('d should be 2 or 3')
        
    
    
    in_tensor = Input(shape=input_size)
    
    #encoder
    padding = 'same'
    trainable = True
    depth = 4
    kernel_size = 3
    filters = 32
    kr = regularizers.l2(1e-6)
    maxPool = (2,2)
    residual_connections = []
    
    #add a conv block
    in_ = ConvND(int(filters*cf), kernel_size,padding=padding,kernel_regularizer=kr,trainable=trainable)(in_tensor)
    in_ = LeakyReLU(alpha=1e-2)(in_)
    in_ = Dropout(0.2)(in_)
    
    for i in range(depth):
        print(i)
        #conv = ConvND(int(filters*cf), kernel_size,padding=padding, kernel_regularizer=kr,trainable=trainable,strides=1,activation='relu')(in_)
        #conv = BatchNormalization()(conv)
        in_ = ConvND(int(filters * cf), kernel_size,padding=padding, kernel_regularizer=kr,trainable=trainable,strides=2,activation='relu')(in_)
        #bn = BatchNormalization()(conv)
        #in_ = MaxPoolingND(pool_size=maxPool)(conv)
        #residual_connections.append(conv)

    
    x_up = Flatten()(in_)
    x_up = Dense(int(1024*cf),activation='relu')(x_up)
    #x_up = BatchNormalization()(x_up)
    #x_up = Dense(128,activation='relu')(x_up)
    
    z_mean = Dense(int(512*cf), name='z_mean')(x_up)
    z_log_var = Dense(int(512*cf), name='z_log_var')(x_up)
    z = Lambda(sampling, name='Dec_VAE_VDraw_Sampling')([z_mean, z_log_var])


    z_mean_reshape = Reshape((int(512*cf),1),name='r1')(z_mean)
    z_log_var_reshape = Reshape((int(512*cf),1),name='r2')(z_log_var)
    z_concatenate = Concatenate(axis=-1)([z_mean_reshape,z_log_var_reshape])
    
    encoder_model = Model(in_tensor,[z,z_concatenate])
    encoder_model.summary()

    
    
    #flatten part
    if d==3:
        _,H,W,D,c = K.int_shape(in_)
    elif d==2:
        _,H,W,c = K.int_shape(in_)
    #Sampler
    latent_inputs = Input(shape=(int(512*cf),))
    if d==3:
        x_up = Dense((c) * (H) * (W) * (D))(latent_inputs)
    elif d==2:
        x_up = Dense((c) * (H) * (W))(latent_inputs)
    
    
    if d==3:
        x_up = Reshape((H,W,D,c),name='reshape')(x_up)
    elif d==2:
        x_up = Reshape((H,W,c),name='reshape')(x_up)
        
    
    #decoder part
    #residual_connections = residual_connections[::-1]
    for i in range(depth):
        # Reduce filter count
        filters /= 2

        # Up-sampling block
        # Note: 2x2 filters used for backward comp, but you probably
        # want to use 3x3 here instead.
        #up = UpSamplingND(size=maxPool)(x_up)
        #conv = DeconvolutionND(int(filters * cf), kernel_size,
        #              padding=padding,
        #              kernel_regularizer=kr,trainable=trainable,strides=1,activation='relu')(x_up)
        #conv = Concatenate(axis=-1)([residual_connections[i], conv])
        #conv = BatchNormalization()(conv)
        x_up = DeconvolutionND(int(filters * cf), kernel_size,
                      padding=padding,
                      kernel_regularizer=kr,trainable=trainable,strides=2,activation='relu')(x_up)
        #x_up = BatchNormalization()(conv)
    
   
    
    x_up = DeconvolutionND(input_size[-1],kernel_size=4,strides=1,padding='same')(x_up)
    recon = Activation('sigmoid', name='recon')(x_up)

    print(recon)

    decoder_model = Model([in_tensor,latent_inputs],recon)
    encoder_decoder_model = Model(in_tensor,[decoder_model([in_tensor,encoder_model(in_tensor)[0]]), encoder_model(in_tensor)[1]])
    
    
    return encoder_model,decoder_model,encoder_decoder_model
    #return encoder_decoder_model
    
    
    
    
    
    
    
    
    
    
    
    
    
