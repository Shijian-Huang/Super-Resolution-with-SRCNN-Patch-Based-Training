import os
import torch
import cv2
import numpy as np

from model import SRCNN
from dataset import DIV2KDataset

from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def evaluate_single_image(lr_path, hr_path, model_path="checkpoints/srcnn.pth"):
    """
    Evaluate a single low-resolution (LR) and high-resolution (HR) image pair using a trained SRCNN model.
    
    Parameters:
        lr_path (str): Path to the low-resolution image.
        hr_path (str): Path to the high-resolution ground truth image.
        model_path (str): Path to the saved SRCNN model weights (default: "checkpoints/srcnn.pth").
    """
    # Select device: Use Apple MPS (Metal Performance Shaders) if available, otherwise use CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load the trained SRCNN model
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load saved model weights
    model.eval()

    # Read the LR & HR images using OpenCV (default format is BGR)
    lr_img_bgr = cv2.imread(lr_path)
    hr_img_bgr = cv2.imread(hr_path)

    # Convert BGR images to RGB (since PyTorch expects RGB format)
    lr_img_rgb = cv2.cvtColor(lr_img_bgr, cv2.COLOR_BGR2RGB)
    hr_img_rgb = cv2.cvtColor(hr_img_bgr, cv2.COLOR_BGR2RGB)

    # Convert images to PyTorch tensors
    lr_tensor = torch.from_numpy(lr_img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)/ 255.
    hr_tensor = torch.from_numpy(hr_img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)/ 255.

    # Perform inference with the SRCNN model
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
    
    # Convert SR image tensor back to a NumPy array
    sr_img = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr_img = np.clip(sr_img, 0, 1)  # Ensure pixel values are in the valid range [0,1]

    # Convert HR image tensor back to a NumPy array for evaluation
    hr_img = hr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Compute Peak Signal-to-Noise Ratio (PSNR) & Structural Similarity Index (SSIM)
    psnr_val = psnr(hr_img, sr_img, data_range=1.0)
    ssim_val = ssim(hr_img, sr_img, data_range=1.0, multichannel = True)

    print(f"PSNR: {psnr_val:.2f}, SSIM = {ssim_val:.4f}")
    # (Optional) Save the generated SR image to the local directory

    sr_img_bgr = cv2.cvtColor((sr_img*255).astype(np.unit8), cv2.COLOR_RGB2BGR)
    cv2.imwrite("sr_result.png", sr_img_bgr)  # Save the image
    print("SR image saved to sr_result.png")

    if __name__ == "__main__":
        # Test evaluation on a specific image pair
        evaluate_single_image(
            lr_path="../data/DIV2K/LR_bicubic/DIV2K_valid_LR_bicubic/0801x4.png",  # Path to low-resolution image
            hr_path="../data/DIV2K/HR/DIV2K_valid_HR/0801.png"  # Path to high-resolution ground truth image
        )