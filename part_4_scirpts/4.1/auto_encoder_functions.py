import torch
import torch.nn as nn
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar):
    """
    Standard VAE loss = reconstruction loss + KL divergence.
    Uses BCE for reconstruction since inputs are in [0,1].
    """
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_div) / x.size(0)