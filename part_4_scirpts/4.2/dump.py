#!/usr/bin/env python
"""
Optimized UNet Training Pipeline for MRI Segmentation (Lab 4.2)
- Modularized into Dataset, Model, Training Loop, Utilities
- Includes AMP (new API), cosine LR, early stopping, and DSC tracking
- Dataset scan + ignore_index handling for safety
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better error reporting

import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# ============================
# 1. Dataset
# ============================
class MRISliceDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx])
        img = np.array(img, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.int64)
        img = torch.tensor(img).unsqueeze(0)  # (1,H,W)
        mask = torch.tensor(mask, dtype=torch.long)
        return img, mask

# ============================
# 2. Model Components
# ============================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_classes, base_filters=8):
        super().__init__()
        self.down1 = DoubleConv(1, base_filters)
        self.down2 = DoubleConv(base_filters, base_filters*2)
        self.down3 = DoubleConv(base_filters*2, base_filters*4)
        self.down4 = DoubleConv(base_filters*4, base_filters*8)

        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(base_filters*8, base_filters*8, 2, stride=2)
        self.conv_up3 = DoubleConv(base_filters*8+base_filters*4, base_filters*4)
        self.conv_up2 = DoubleConv(base_filters*4+base_filters*2, base_filters*2)
        self.conv_up1 = DoubleConv(base_filters*2+base_filters, base_filters)
        self.out_conv = nn.Conv2d(base_filters, n_classes, 1)

    def forward(self, x):
        c1 = self.down1(x)
        x = self.pool(c1)
        c2 = self.down2(x)
        x = self.pool(c2)
        c3 = self.down3(x)
        x = self.pool(c3)
        x = self.down4(x)

        x = self.up4(x)
        x = torch.cat([x, c3], dim=1)
        x = self.conv_up3(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, c2], dim=1)
        x = self.conv_up2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, c1], dim=1)
        x = self.conv_up1(x)
        return self.out_conv(x)

# ============================
# 3. Utility Functions
# ============================
def dice_coef(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)
    dice = 0.0
    for c in range(num_classes):
        p_c = (pred == c).float()
        t_c = (target == c).float()
        intersection = (p_c * t_c).sum()
        union = p_c.sum() + t_c.sum()
        dice += (2 * intersection + 1e-6) / (union + 1e-6)
    return dice / num_classes

def scan_labels(ds, cap=2000):
    """Scan a dataset for min/max/unique labels."""
    mn, mx = 10**9, -10**9
    seen = set()
    for i in range(min(len(ds), cap)):
        _, m = ds[i]
        m = m.long()
        mn = min(mn, int(m.min()))
        mx = max(mx, int(m.max()))
        if len(seen) < 256:
            seen.update(int(x) for x in torch.unique(m).tolist())
    return mn, mx, sorted(seen)

# ============================
# 4. Training Loop
# ============================
def train_unet(train_loader, val_loader, num_classes, ignore_index=None, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05, ignore_index=ignore_index if ignore_index else -100)
    scaler = GradScaler('cuda')

    # One-batch sanity check
    imgs, masks = next(iter(train_loader))
    print(f"Sanity check: imgs={imgs.shape}, masks={masks.shape}, mask dtype={masks.dtype}")
    with torch.no_grad():
        out = model(imgs.to(device))
    print(f"Model output shape: {out.shape} (expected channels={num_classes})")

    best_dice, patience = 0, 0
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar.set_postfix({"loss": loss.item()})

        # Validation
        model.eval()
        dice_score = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(imgs)
                dice_score += dice_coef(outputs, masks, num_classes).item()
        dice_score /= len(val_loader)
        print(f"Epoch {epoch+1}: Dice={dice_score:.4f}")

        if dice_score > best_dice:
            best_dice = dice_score
            torch.save(model.state_dict(), "best_unet.pth")
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                print("Early stopping triggered.")
                break

# ============================
# 5. Main Entrypoint
# ============================
if __name__ == "__main__":
    torch.cuda.empty_cache()

    train_dataset = MRISliceDataset("keras_png_slices_train", "keras_png_slices_seg_train")
    val_dataset   = MRISliceDataset("keras_png_slices_validate", "keras_png_slices_seg_validate")

    # Scan labels across train + val
    mn_tr, mx_tr, uniq_tr = scan_labels(train_dataset)
    mn_va, mx_va, uniq_va = scan_labels(val_dataset)
    uniq = sorted(set(uniq_tr) | set(uniq_va))
    print(f"Label stats -> min={min(mn_tr,mn_va)}, max={max(mx_tr,mx_va)}, uniques(sample)={uniq}")

    HAS_255 = 255 in uniq
    valid_labels = [u for u in uniq if u != 255]
    num_classes = max(valid_labels) + 1 if valid_labels else 1

    print(f"Using num_classes={num_classes}, ignore_index={255 if HAS_255 else None}")

    train_loader = DataLoader(train_dataset, batch_size=22, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=22, shuffle=False, num_workers=0, pin_memory=True)

    train_unet(train_loader, val_loader, num_classes, ignore_index=255 if HAS_255 else None)