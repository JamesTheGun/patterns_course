import torch
from torch.utils.data import DataLoader
from auto_encoder_classes import VAE
from auto_encoder_functions import vae_loss

# --------------------------
# Minimal Dataset Definition
# --------------------------
import os
import numpy as np
from glob import glob
from PIL import Image

class MRISliceDataset(torch.utils.data.Dataset):
    """Simple dataset for MRI slices (no masks)."""
    def __init__(self, image_dir):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.png")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)  # shape (1,H,W)
        return img

# --------------------------
# Config
# --------------------------
TRAIN_DIR = "../4.2/keras_png_slices_train"
VAL_DIR = "../4.2/keras_png_slices_validate"
BATCH_SIZE = 32
LATENT_DIM = 32
EPOCHS = 10
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = MRISliceDataset(TRAIN_DIR)
val_dataset = MRISliceDataset(VAL_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

vae = VAE(latent_dim=LATENT_DIM, input_shape=(1, 256, 256)).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=LR)

best_val_loss = float("inf")
patience, max_patience = 0, 3  # early stopping

for epoch in range(EPOCHS):
    # --------------------------
    # Training
    # --------------------------
    vae.train()
    total_train_loss = 0
    for imgs in train_loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = vae(imgs)
        loss = vae_loss(recon, imgs, mu, logvar)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # --------------------------
    # Validation
    # --------------------------
    vae.eval()
    total_val_loss = 0
    with torch.no_grad():
        for imgs in val_loader:
            imgs = imgs.to(device)
            recon, mu, logvar = vae(imgs)
            total_val_loss += vae_loss(recon, imgs, mu, logvar).item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --------------------------
    # Checkpoint Saving
    # --------------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(vae.state_dict(), f"vae_epoch{epoch+1}_loss{best_val_loss:.4f}.pth")
        print(f"✅ Saved new best model (val loss {best_val_loss:.4f})")
        patience = 0
    else:
        patience += 1
        if patience >= max_patience:
            print("⏹ Early stopping triggered.")
            break
