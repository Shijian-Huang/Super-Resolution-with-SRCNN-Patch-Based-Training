import os
import argparse
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from model_srcnn import SRCNN
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torchvision.transforms as transforms
import csv

def evaluate_dir(lr_dir, hr_dir, model_path, device, save_dir=None):
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    to_tensor = transforms.ToTensor()
    results = []
    filenames = sorted(os.listdir(lr_dir))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for filename in tqdm(filenames, desc="Evaluating"):
        lr_path = os.path.join(lr_dir, filename)
        hr_path = os.path.join(hr_dir, filename.replace("x4", ""))

        if not os.path.exists(hr_path):
            continue

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = lr_img.resize(hr_img.size, Image.BICUBIC)

        lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)
        hr_tensor = to_tensor(hr_img).unsqueeze(0).to(device)

        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        sr_np = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        hr_np = hr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        psnr_val = psnr(hr_np, sr_np, data_range=1.0)
        ssim_val = ssim(hr_np, sr_np, data_range=1.0, channel_axis=-1)
        results.append((filename, psnr_val, ssim_val))

        if save_dir:
            name, _ = os.path.splitext(filename)
            sr_img = Image.fromarray((sr_np * 255).astype(np.uint8))
            sr_img.save(os.path.join(save_dir, f"{name}_SR_PSNR{psnr_val:.2f}_SSIM{ssim_val:.4f}.png"))

    if save_dir:
        with open(os.path.join(save_dir, "evaluation_results.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Filename", "PSNR", "SSIM"])
            writer.writerows(results)
            avg_psnr = np.mean([r[1] for r in results])
            avg_ssim = np.mean([r[2] for r in results])
            writer.writerow(["Average", f"{avg_psnr:.2f}", f"{avg_ssim:.4f}"])

    print(f"\n📊 Avg PSNR: {avg_psnr:.2f} dB | Avg SSIM: {avg_ssim:.4f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_dir", required=True)
    parser.add_argument("--hr_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--save_dir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_dir(args.lr_dir, args.hr_dir, args.model, torch.device(args.device), args.save_dir)
