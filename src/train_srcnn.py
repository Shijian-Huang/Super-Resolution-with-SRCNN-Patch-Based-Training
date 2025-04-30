import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from dataset_patch import DIV2KPatchDataset, DIV2KDataset
from model_srcnn import SRCNN

def train_sr():
    print("===== Starting SRCNN Training =====")
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print("Using device:", device)

    train_hr = os.path.join(SCRIPT_DIR, "../data/DIV2K/HR/train_HR")
    train_lr = os.path.join(SCRIPT_DIR, "../data/DIV2K/LR_bicubic/train_LR_bicubic/X4")
    val_hr   = os.path.join(SCRIPT_DIR, "../data/DIV2K/HR/valid_HR")
    val_lr   = os.path.join(SCRIPT_DIR, "../data/DIV2K/LR_bicubic/valid_LR_bicubic/X4")

    train_dataset = DIV2KPatchDataset(train_hr, train_lr, patch_size=96, repeat=100)
    val_dataset   = DIV2KDataset(val_hr, val_lr)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(SCRIPT_DIR, f"../runs/srcnn_patch_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    os.makedirs("checkpoints", exist_ok=True)

    best_loss = float("inf")
    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for lr_tensor, hr_tensor in train_loader:
            lr_tensor, hr_tensor = lr_tensor.to(device), hr_tensor.to(device)
            sr = model(lr_tensor)
            loss = criterion(sr, hr_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        print(f"[Epoch {epoch+1:03d}/100] Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_tensor, hr_tensor in val_loader:
                lr_tensor, hr_tensor = lr_tensor.to(device), hr_tensor.to(device)
                sr = model(lr_tensor)
                val_loss += criterion(sr, hr_tensor).item()
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        print(f"            >> Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model_path = f"checkpoints/best_srcnn_patch_{timestamp}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"✅ Saved best model to: {model_path}")
        scheduler.step(avg_val_loss)

    writer.close()
    print("✅ Training complete.")

if __name__ == "__main__":
    train_sr()
