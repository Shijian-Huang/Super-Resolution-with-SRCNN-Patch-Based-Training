# Super-Resolution with SRCNN (Patch-Based Training)

This repository implements a Super-Resolution Convolutional Neural Network (SRCNN) using patch-based training on the DIV2K dataset. It supports TensorBoard logging, PSNR/SSIM evaluation, and image saving for visual comparison.

---

## 📁 Project Structure

```
project-root/
├── Makefile
├── src/
│   ├── train_srcnn.py
│   ├── evaluate_batch.py
│   ├── model_srcnn.py
│   └── dataset_patch.py
├── data/
│   └── DIV2K/
│       ├── HR/
│       │   ├── train_HR/
│       │   └── valid_HR/
│       └── LR_bicubic/
│           └── valid_LR_bicubic/X4/
├── checkpoints/
├── output/
└── runs/
```

---

## 🚀 Usage

### 🔧 Requirements

- Python 3.7+
- PyTorch
- torchvision
- scikit-image
- TensorBoard
- tqdm

Install with:

```bash
pip install -r requirements.txt
```

### 🏗 Training

Trains the SRCNN model on random image patches:

```bash
make train
```

- Uses 96×96 HR patches
- Performs bicubic interpolation on LR inputs
- Applies horizontal flip augmentation

TensorBoard logs are saved under `runs/`.

### 📊 Evaluation

Evaluates the model using full-resolution validation images:

```bash
make evaluate
```

- Outputs SR images to `output/`
- Saves metrics (`PSNR`, `SSIM`) to `evaluation_results.csv`
- Filenames include metrics: `0801_SR_PSNR24.15_SSIM0.8124.png`

### 🧹 Clean Logs and Outputs

```bash
make clean
```

---

## 🧠 Notes

- Patch training speeds up convergence and reduces memory usage
- Full-image validation reflects realistic performance
- Learning rate is dynamically reduced if validation loss plateaus