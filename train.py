import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from variational_autoencoder.py import create_encoder, create_decoder
from VAE import VAE


if __name__ == 'main':
  (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
  mnist_digits = np.concatenate([x_train, x_test], axis=0)
  mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

  encoder = create_encoder()
  decoder = create_decoder(latent_dim=2)

  vae = VAE(encoder, decoder)
  vae.compile(optimizer=keras.optimizers.Adam())
  vae.fit(mnist_digits, epochs=30, batch_size=128)
