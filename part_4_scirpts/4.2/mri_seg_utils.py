import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

#util functions

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
