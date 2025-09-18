import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Variational Autoencoder
# --------------------------
class Encoder(nn.Module):
    """Encoder network: downsamples image -> latent distribution parameters (μ, logσ²)."""
    def __init__(self, latent_dim=32, base_filters=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, base_filters, 3, stride=2, padding=1)   # -> H/2
        self.bn1   = nn.BatchNorm2d(base_filters)
        self.conv2 = nn.Conv2d(base_filters, base_filters*2, 3, stride=2, padding=1) # -> H/4
        self.bn2   = nn.BatchNorm2d(base_filters*2)
        self.conv3 = nn.Conv2d(base_filters*2, base_filters*4, 3, stride=2, padding=1) # -> H/8
        self.bn3   = nn.BatchNorm2d(base_filters*4)

        # Will be flattened before linear layers
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(base_filters*4*8*8, latent_dim)     # mean
        self.fc_logvar = nn.Linear(base_filters*4*8*8, latent_dim) # log variance

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
class Decoder(nn.Module):
    """Decoder network: latent vector -> reconstructed image."""
    def __init__(self, latent_dim=32, base_filters=8):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_filters*4*8*8)
        self.deconv1 = nn.ConvTranspose2d(base_filters*4, base_filters*2, 4, stride=2, padding=1) # -> H/4
        self.deconv2 = nn.ConvTranspose2d(base_filters*2, base_filters, 4, stride=2, padding=1)   # -> H/2
        self.deconv3 = nn.ConvTranspose2d(base_filters, 1, 4, stride=2, padding=1)               # -> H

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), -1, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # output in [0,1]
        return x
    
class VAE(nn.Module):
    """Full VAE model (Encoder + reparam + Decoder)."""
    def __init__(self, latent_dim=32, base_filters=8):
        super().__init__()
        self.encoder = Encoder(latent_dim, base_filters)
        self.decoder = Decoder(latent_dim, base_filters)

    def reparameterize(self, mu, logvar):
        """Sample z ~ N(mu, sigma^2) using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

