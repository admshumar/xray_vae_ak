#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:28:52 2019

@author: akshay
"""
from tensorflow.keras.utils import plot_model
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from save_tSNE import LatentSpaceTSNE
from scipy.stats import norm
from logistic_regression import logistic_regression
import pickle
#code from https://keras.io/examples/variational_autoencoder/


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist",epoch=1):
    """Plots labels and MNIST digits as a function of the 2D latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test, x_train, y_train = data
    os.makedirs(model_name, exist_ok=True)
    

    #filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _= encoder.predict(x_test, batch_size=batch_size)

    z_mean_train, _= encoder.predict(x_train, batch_size=batch_size)

    print('Logisitic regression for testing ....')
    log_reg = logistic_regression(z_mean_train, y_train, z_mean, y_test)
    pickle.dump(log_reg, open('log_reg.sav', 'wb'))
    #np.save('latents_train_epoch_'+str(epoch)+'.npy',z_mean_train)

    print(z_mean.shape, y_test.shape)
    lp = LatentSpaceTSNE(z_mean, y_test, 'tSNE_epoch_'+str(epoch)+'.png', './', number_of_tnse_components=2)
    lp.save_tsne()
    #plt.figure(figsize=(12, 10))
    #plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    #plt.colorbar()
    #plt.xlabel("z[0]")
    #plt.ylabel("z[1]")
    #plt.savefig(filename)
    #plt.show()
    plt.close()
    
    # display a 30x30 2D manifold of digits
    #n = 10
    #digit_size = 160
    #figure = np.zeros((n, digit_size, digit_size))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    #z1 = norm.ppf(np.linspace(0.01, 0.99, n))
    #z2 = norm.ppf(np.linspace(0.01, 0.99, n))

    #z_grid = np.dstack(np.meshgrid(z1, z2))
    
    #x_pred_grid = decoder.predict(z_grid.reshape(n*n, 2)) \
    #                               .reshape(n, n, digit_size, digit_size)

    x_pred_grid = decoder.predict([x_test,z_mean],batch_size=2)

    #plt.figure(figsize=(10, 10))
    #plt.imshow(np.block(list(map(list, x_pred_grid))), cmap='gray')

    for i in range(x_pred_grid.shape[0]):
        im = np.squeeze(x_pred_grid[i,...])
        plt.imshow(im)
        plt.savefig('digits'+'_'+str(i)+'.png')


        if i==10:
            break


                        
        #plt.show()
    plt.close()
