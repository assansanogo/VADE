# The first step is to build the sampling layer
class Sampling(layers.Layer):
    ''' Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
     (the latent_dimension is not impacting this sampling function !)
     Parameters:
                    z_mean (float): A decimal integer
                    z_log_var (int): Another decimal integer

            Returns:
                    the probability of z given z_mean and z_log_var 
    '''

    def call(self, inputs):
        # we use the mean and the logvariance 
        z_mean, z_log_var = inputs
        # z_mean is a tensor of size (batch, latent dimension)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # epsilon is N(0,1) 
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        # this expression comes from the reparameterization trick and allows to do back propagation
        # based on mu and/or sigma while keeping stochasticity in epsilon N(0,1)
        
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_sampler(x, latent_dim):
  '''
  wrapper function to create a sampler
  
   Parameters:
                x (float): A decimal integer


        Returns:
                z_mean, z_log_var and z (where z = z_mean + eps*z_log_var)
  '''
  latent_dimensions = latent_dim

  z_mean = layers.Dense(latent_dimensions, name="z_mean")(x)
  z_log_var = layers.Dense(latent_dimensions, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])
  
  return z_mean, z_log_var, z


# The second step is to build an encoder layer
def create_encoder():
  '''
  encoder plus sampler
  
 Parameters:
            None

    Returns:
            encoder (keras.Model)
  '''
  encoder_inputs = keras.Input(shape=(28, 28, 1))
  x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
  x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Flatten()(x)
  x = layers.Dense(16, activation="relu")(x)
  Z_mean, Z_log_var, Z = create_sampler(x, latent_dim)
  encoder = keras.Model(encoder_inputs, [Z_mean, Z_log_var, Z], name="encoder")
  print(encoder.summary())
  return encoder

  
# the third step is to build the decoder layer

def create_decoder(latent_dim):
  '''
  decoder part
  
   Parameters:
            latent_dim : (dimension of the latent representation)

    Returns:
            decoder (keras.Model)
  '''
  latent_inputs = keras.Input(shape=(latent_dim,))
  # dense layer which concentrates computation
  x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
  # reshape into a squared output
  x = layers.Reshape((7, 7, 64))(x)
  # upscale x2
  x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
  # upscale x2
  x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
  # keep 1 feature map
  decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
  # model with latent inputs (INPUT) & latent inputs (OUTPUT)
  decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
  print(decoder.summary())
  return decoder
