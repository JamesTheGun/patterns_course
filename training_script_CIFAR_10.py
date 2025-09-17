#!/usr/bin/env python
import math
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler

# ----------------------------
# Speed knobs for A100
# ----------------------------
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ----------------------------
# ResNet-18 (from scratch)
# ----------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    data_dir: str = os.environ.get("DATA_DIR", "./data")
    epochs: int = int(os.environ.get("EPOCHS", 60))
    batch_size: int = int(os.environ.get("BATCH_SIZE", 512))
    lr: float = float(os.environ.get("LR", 0.4))  # with large batch, higher base LR
    weight_decay: float = float(os.environ.get("WD", 5e-4))
    warmup_epochs: int = int(os.environ.get("WARMUP", 5))
    num_workers: int = int(os.environ.get("WORKERS", 8))
    amp: bool = os.environ.get("AMP", "1") == "1"
    label_smoothing: float = float(os.environ.get("LS", 0.1))
    channels_last: bool = os.environ.get("CHANNELS_LAST", "1") == "1"
    seed: int = int(os.environ.get("SEED", 42))

cfg = Config()

# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CosineLRSchedule:
    def __init__(self, optimizer, base_lr, steps, warmup=0):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.steps = steps
        self.warmup = warmup
        self.step_num = 0
    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup:
            lr = self.base_lr * self.step_num / max(1, self.warmup)
        else:
            t = (self.step_num - self.warmup) / max(1, self.steps - self.warmup)
            lr = 0.5 * self.base_lr * (1 + math.cos(math.pi * t))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

# ----------------------------
# Data
# ----------------------------
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# ----------------------------
# Train / Eval
# ----------------------------
def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def main():
    set_seed(cfg.seed)
    assert torch.cuda.is_available(), "CUDA required on Rangpur GPU node"
    device = torch.device('cuda')

    # Data
    train_set = datasets.CIFAR10(root=cfg.data_dir, train=True, download=True, transform=train_transform)
    test_set  = datasets.CIFAR10(root=cfg.data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=1024, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)

    # Model
    model = resnet18(num_classes=10)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.to(device)

    # Optimizer / Loss / LR schedule
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    steps_per_epoch = math.ceil(len(train_loader))
    scheduler = CosineLRSchedule(opt, base_lr=cfg.lr, steps=cfg.epochs * steps_per_epoch, warmup=cfg.warmup_epochs * steps_per_epoch)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = GradScaler(enabled=cfg.amp)

    best_acc = 0.0
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if cfg.channels_last:
                x = x.to(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).step(opt)
            scaler.update()
            lr_now = scheduler.step()

            running_loss += loss.item()
            global_step += 1
            if (i+1) % 50 == 0:
                avg = running_loss / 50
                print(f"Epoch {epoch:03d}/{cfg.epochs} | Step {i+1:04d}/{len(train_loader)} | loss {avg:.4f} | lr {lr_now:.5f}")
                running_loss = 0.0

        # Eval
        acc = accuracy(model, test_loader, device)
        best_acc = max(best_acc, acc)
        print(f"[Epoch {epoch}] test_acc={acc:.2f}% | best={best_acc:.2f}%")

        # Early stop once we exceed 93% to save time
        if best_acc >= 93.0:
            print("Target accuracy reached. Stopping early to meet time constraint.")
            break

    # Save final model
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_acc': best_acc,
        'cfg': cfg.__dict__,
    }, "./checkpoints/resnet18_cifar10.pt")
    print(f"Done. Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()