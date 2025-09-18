import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from mri_seg_utils import *
from mri_segmentation_classes import *

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