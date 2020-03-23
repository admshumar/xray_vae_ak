from vae_model import encoder_decoder
import numpy as np

size_image = (224, 224, 1)

def extract_latent_means(latent_parameters):
    latent_means = latent_parameters[:, 0:latent_parameters.shape[1]//2]
    return latent_means

def predict_means(data):
    latent_data = encoder_model.predict(data)
    latent_parameters = latent_data[1]
    return extract_latent_means(latent_parameters)

# build the model
encoder_model, _, _ = encoder_decoder(size_image, 2)
encoder_model.summary()
encoder_model.load_weights('vae_encoder_chest.h5')

x_train = np.load('x_train.npy')
x_val = np.load('x_val.npy')
x_test = np.load('x_test.npy')

"""
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')
"""

x_train_latent = predict_means(x_train)
x_val_latent = predict_means(x_val)
x_test_latent = predict_means(x_test)

np.save('x_train_latent.npy', x_train_latent)
np.save('x_val_latent.npy', x_val_latent)
np.save('x_test_latent.npy', x_test_latent)