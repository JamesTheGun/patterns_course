import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler


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