VAE Project: Simple Variational Autoencoder

Basic VAE Architecture


This project implements a simple Variational Autoencoder (VAE) using PyTorch. VAEs are generative models that can learn to recreate input data and generate new similar samples.

Key Components

    Encoder: Compresses input to latent space

    Latent Space: Gaussian distribution (μ, σ)

    Decoder: Reconstructs input from latent code

    Loss: Reconstruction + KL divergence
    
Original paper in which VAE's were introduced: https://arxiv.org/pdf/1312.6114 
