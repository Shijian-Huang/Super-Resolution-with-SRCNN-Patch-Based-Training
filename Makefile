# Makefile for SRCNN Patch Training and Evaluation

PYTHON = python
SRC_DIR = srcnn_patch_revised
DATA_DIR = data/DIV2K
CHECKPOINT = checkpoints/best_srcnn_patch.pth
DEVICE = mps  # or cuda / cpu

# Train using patch-based dataset
train:
	$(PYTHON) $(SRC_DIR)/train.py

# Evaluate using full-image dataset
evaluate:
	$(PYTHON) $(SRC_DIR)/evaluate_batch.py \
		--lr_dir $(DATA_DIR)/LR_bicubic/valid_LR_bicubic/X4 \
		--hr_dir $(DATA_DIR)/HR/valid_HR \
		--model $(CHECKPOINT) \
		--save_dir output \
		--device $(DEVICE)

# Clean up checkpoints and logs
clean:
	rm -rf checkpoints/*
	rm -rf runs/*
	rm -rf output/*

# Run both train and evaluate
all: train evaluate
