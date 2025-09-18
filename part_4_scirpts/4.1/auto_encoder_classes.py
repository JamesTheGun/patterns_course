import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# Encoder
# --------------------------
class Encoder(nn.Module):
    """Encoder network: downsamples image -> latent distribution parameters (μ, logσ²)."""
    def __init__(self, latent_dim=32, base_filters=8, input_shape=(1, 256, 256)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, base_filters, 3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(base_filters)
        self.conv2 = nn.Conv2d(base_filters, base_filters*2, 3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(base_filters*2)
        self.conv3 = nn.Conv2d(base_filters*2, base_filters*4, 3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(base_filters*4)
        self.flatten = nn.Flatten()

        # Dynamically infer shape after conv stack
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self._forward_conv(dummy)
            self.start_shape = out.shape[1:]       # e.g. (32, 32, 32)
            self.feature_dim = out.numel()

        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

    def _forward_conv(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# --------------------------
# Decoder
# --------------------------
class Decoder(nn.Module):
    """Decoder network: latent vector -> reconstructed image."""
    def __init__(self, latent_dim, start_shape, base_filters=8):
        super().__init__()
        self.start_shape = start_shape
        self.fc = nn.Linear(latent_dim, int(torch.prod(torch.tensor(start_shape))))
        self.deconv1 = nn.ConvTranspose2d(start_shape[0], base_filters*2, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(base_filters*2, base_filters, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(base_filters, 1, 4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), *self.start_shape)  # reshape to (B, C, H, W)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


# --------------------------
# VAE
# --------------------------
class VAE(nn.Module):
    """Full variational autoencoder."""
    def __init__(self, latent_dim=32, base_filters=8, input_shape=(1, 256, 256)):
        super().__init__()
        self.encoder = Encoder(latent_dim, base_filters, input_shape)
        # Pass encoder's discovered shape to the decoder
        self.decoder = Decoder(latent_dim, self.encoder.start_shape, base_filters)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
