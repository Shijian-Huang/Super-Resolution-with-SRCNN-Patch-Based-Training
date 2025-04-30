import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DIV2KPatchDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=96, scale=4, augment=True, repeat=100):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_files = sorted(os.listdir(hr_dir))
        self.lr_files = sorted(os.listdir(lr_dir))
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment
        self.repeat = repeat
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_files) * self.repeat

    def __getitem__(self, idx):
        real_idx = idx % len(self.hr_files)
        hr_path = os.path.join(self.hr_dir, self.hr_files[real_idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[real_idx])

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")
        lr = lr.resize(hr.size, Image.BICUBIC)

        hr_w, hr_h = hr.size
        ps = self.patch_size

        if hr_w < ps or hr_h < ps:
            raise ValueError(f"Image too small: {hr_path} ({hr_w}x{hr_h})")

        x = random.randint(0, hr_w - ps)
        y = random.randint(0, hr_h - ps)

        hr_patch = hr.crop((x, y, x + ps, y + ps))
        lr_patch = lr.crop((x, y, x + ps, y + ps))

        if self.augment and random.random() < 0.5:
            hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
            lr_patch = lr_patch.transpose(Image.FLIP_LEFT_RIGHT)

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_files = sorted(os.listdir(hr_dir))
        self.lr_files = sorted(os.listdir(lr_dir))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])

        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")
        lr_img = lr_img.resize(hr_img.size, Image.BICUBIC)

        return self.to_tensor(lr_img), self.to_tensor(hr_img)
