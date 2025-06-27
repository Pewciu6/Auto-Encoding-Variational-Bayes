Simple Variational Autoencoder (VAE) Implementation

[https://vae_picture.png](https://github.com/Pewciu6/Auto-Encoding-Variational-Bayes/blob/main/vae_picture.png)
Basic VAE Architecture - Encoder, Latent Space, and Decoder

This project implements a simple Variational Autoencoder (VAE) using PyTorch. VAEs are powerful generative models that can learn to recreate input data and generate new similar samples.
Key Components

    Encoder: Compresses input data into a latent space representation

    Latent Space: Represents data as Gaussian distributions (μ, σ) instead of fixed points

    Decoder: Reconstructs input data from latent space representations

    Loss Function: Combination of reconstruction loss and KL divergence

How VAEs Work

    Encoder processes input data

    Creates parameters for latent distribution (μ = mean, σ = standard deviation)

    Sampling from the distribution using the reparameterization trick:
    z = μ + σ ⊙ ε where ε ~ N(0, I)

    Decoder reconstructs input from the latent sample

    Model learns by minimizing:
    Loss = Reconstruction Loss + KL Divergence

Key Features

    Generates new data samples similar to training data

    Learns compressed representations of input data

    Continuous latent space allows smooth interpolations

    Simple PyTorch implementation

Original Paper

Variational Autoencoders were introduced in:
Auto-Encoding Variational Bayes
Kingma & Welling, 2013
Basic Usage
python

from vae import VAE
import torch

# Initialize VAE
vae = VAE(input_size=784, hidden_size=400, latent_size=20)

# Pass data through VAE
data = torch.randn(32, 784)  # Batch of 32 samples
reconstructed, mu, logvar = vae(data)

# Calculate loss
loss = vae.loss_function(reconstructed, data, mu, logvar)

Files

    vae.py - VAE implementation

    vae_picture.png - Architecture diagram

    README.md - This documentation

Requirements

    Python 3.x

    PyTorch (install with pip install torch)
