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