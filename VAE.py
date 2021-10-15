import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from variational_autoencoder create_decoder create_encoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        # encoder and decoder as defined in the module : variational_autoencoder
        self.encoder = create_encoder
        self.decoder = create_decoder(latent_dim = 1024)
        
        # composite (= total) loss
        # NELBO loss
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        # reconstruction loss (fidelity of the reconstructed image with the original image)
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        #  distribution loss (fidelity of P(Z/X) to follow N(mu, var)
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        # metrics to keep track of (total loss
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # 3 variables we want to "learn"
            # 
            z_mean, z_log_var, z = self.encoder(data)
            reconstructed_data = self.decoder(z)
            
            # loss which optimize the P(c, x) 
            # averaged over the batch
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstructed_data), axis=(1, 2)
                )
            )
            # loss which forces Z to follow a distribution of type normal of dimensionality 1
            # averaged over batch 
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            #composite loss ( - binary cross entropy + KL = - (binary cross entropy - KL) = NELBO)
            total_loss = reconstruction_loss + kl_loss
        
        # define what/how to backpropagate
        # define what gradient to compute d[total_loss]/d[trainable weights]
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        # apply the gradients to the trainable weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # update each loss tracker with the current value 
        # stacking of the total loss value
        self.total_loss_tracker.update_state(total_loss)
        
        # stacking of the reconstruction loss
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        
        # stacking of the reconstruction loss
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
      
