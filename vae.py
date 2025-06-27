import torch.nn as nn
import torch


class VAE(nn.Module):
    """Variational Autoencoder (VAE) class."""
    
    def __init__(self, hidden_size, latent_size, input_size):
        """ Initializes the VAE model."""
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.input_size = input_size
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
        )

        self.mu_layer = nn.Linear(hidden_size//2, latent_size)
        self.logvar_layer = nn.Linear(hidden_size//2, latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encodes the input into mean and log variance."""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from the latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decodes the latent variable back to the input space."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

    def loss_function(self, reconstructed_x, x, mu, logvar):
        """Calculates the VAE loss function."""
        BCE = nn.functional.binary_cross_entropy(reconstructed_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
def train_vae(model, data_loader, optimizer, epochs):
    """Trains the VAE model."""
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            reconstructed_x, mu, logvar = model(batch)
            loss = model.loss_function(reconstructed_x, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')