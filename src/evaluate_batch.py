import os
import argparse
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from model_srcnn import SRCNN
from dataset import ResizeWithPadding
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms


def evaluate_dir(lr_dir, hr_dir, model_path, device, save_dir=None):
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    lr_transform = ResizeWithPadding((512, 512))
    hr_transform = ResizeWithPadding((512, 512))
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

        # Load and resize images with padding (preserve aspect ratio)
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        lr_img, _ = lr_transform(lr_img)
        hr_img, _ = hr_transform(hr_img)

        lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)
        hr_tensor = to_tensor(hr_img).unsqueeze(0).to(device)

        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        sr_np = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        sr_np = np.clip(sr_np, 0, 1)
        sr_img = Image.fromarray((sr_np * 255).astype(np.uint8))

        if save_dir:
            name, _ = os.path.splitext(filename)
            save_path = os.path.join(save_dir, f"{name}_SR.png")
            sr_img.save(save_path)

        # Prepare HR image for metric comparison
        hr_np = np.array(hr_img).astype(np.float32) / 255.0

        # Compute metrics
        psnr_val = psnr(hr_np, sr_np, data_range=1.0)
        ssim_val = ssim(hr_np, sr_np, data_range=1.0, channel_axis=-1)
        results.append((filename, psnr_val, ssim_val))

    print("\nEvaluation Complete:\n")
    avg_psnr = np.mean([r[1] for r in results])
    avg_ssim = np.mean([r[2] for r in results])
    for f, p, s in results:
        print(f"{f}: PSNR={p:.2f}, SSIM={s:.4f}")
    print(f"\n📊 Average PSNR: {avg_psnr:.2f} dB | Average SSIM: {avg_ssim:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_dir", required=True)
    parser.add_argument("--hr_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--save_dir")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_dir(args.lr_dir, args.hr_dir, args.model, torch.device(args.device), args.save_dir)