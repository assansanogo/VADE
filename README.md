# VADE
I. Definitions

  a Variational Autoencoders is a generative model. 
  a VAE is a particular autoencoder :
  - which  aims to capture the latent space of our input (by optimizing the reconstruction from the latent space to the input).
  - whose encodings distribution is regularised during the training (in order to ensure that its latent space follows a given distribution.)
  - uses variational inference method(approximation of the theoretical loss by the NELBO/ELBO)

  As such, this process allows us to generate some new data. 

II. Variational models:
 1. VAE (Variational auto encoder).  :▶️:  VAE.ipynb [VAE.ipynb](./notebooks/VAE.ipynb)
 2. VADE (Variational deep encoding) :▶️: VADE.ipynb [VADE.ipynb](./notebooks/VADE.ipynb)
