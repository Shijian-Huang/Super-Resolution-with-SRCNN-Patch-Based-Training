import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, target_size=(512, 512)):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_files = sorted(os.listdir(hr_dir))
        self.lr_files = sorted(os.listdir(lr_dir))
        self.to_tensor = transforms.ToTensor()
        self.lr_transform = ResizeWithPadding(target_size)
        self.hr_transform = ResizeWithPadding(target_size)

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])

        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")

        lr_img, _ = self.lr_transform(lr_img)
        hr_img, _ = self.hr_transform(hr_img)

        return self.to_tensor(lr_img), self.to_tensor(hr_img)


class ResizeWithPadding:
    """ Resize while keeping aspect ratio and pad to target size """
    def __init__(self, target_size=(512, 512), fill_color=(0, 0, 0)):
        self.target_size = target_size
        self.fill_color = fill_color

    def __call__(self, img):
        w, h = img.size
        ratio = min(self.target_size[0] / w, self.target_size[1] / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        img_resized = img.resize((new_w, new_h), Image.BICUBIC)
        paste_x = (self.target_size[0] - new_w) // 2
        paste_y = (self.target_size[1] - new_h) // 2
        crop_box = (paste_x, paste_y, paste_x + new_w, paste_y + new_h)

        new_img = Image.new("RGB", self.target_size, self.fill_color)
        new_img.paste(img_resized, (paste_x, paste_y))

        return new_img, crop_box