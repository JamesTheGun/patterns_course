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