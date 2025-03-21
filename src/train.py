import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from dataset import DIV2KDataset
from model_srcnn import SRCNN

def train_sr():
    print("===== Starting SRCNN Training =====")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths
    train_hr_dir = os.path.join(SCRIPT_DIR, "../data/DIV2K/HR/train_HR")
    train_lr_dir = os.path.join(SCRIPT_DIR, "../data/DIV2K/LR_bicubic/train_LR_bicubic/X4")
    val_hr_dir   = os.path.join(SCRIPT_DIR, "../data/DIV2K/HR/valid_HR")
    val_lr_dir   = os.path.join(SCRIPT_DIR, "../data/DIV2K/LR_bicubic/valid_LR_bicubic/X4")

    # Dataset & DataLoader
    train_dataset = DIV2KDataset(train_hr_dir, train_lr_dir)
    val_dataset   = DIV2KDataset(val_hr_dir, val_lr_dir)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=os.path.join(SCRIPT_DIR, f"../runs/srcnn_{timestamp}"))
    os.makedirs("checkpoints", exist_ok=True)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for lr_tensor, hr_tensor in train_loader:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

            sr_tensor = model(lr_tensor)
            loss = criterion(sr_tensor, hr_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr_tensor, hr_tensor in val_loader:
                lr_tensor = lr_tensor.to(device)
                hr_tensor = hr_tensor.to(device)
                sr_tensor = model(lr_tensor)
                val_loss += criterion(sr_tensor, hr_tensor).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        # Save model
        model_path = f"checkpoints/srcnn_epoch{epoch+1}_{timestamp}.pth"
        torch.save(model.state_dict(), model_path)

    writer.close()
    print("✅ Training complete. Model saved.")

if __name__ == "__main__":
    train_sr()