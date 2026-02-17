# divNet: Diverging 3D CNN for Alzheimer's Disease Classification

Implementation of the diverging 3D CNN (divNet / L4) from:

> Khagi, B., & Kwon, G. R. (2020). 3D CNN Design for the Classification of Alzheimer's Disease Using Brain MRI and PET. *IEEE Access*, 8, 217830–217847.

## Architecture

Diverging kernel sizes (3 → 5 → 7 → 9) across four convolutional blocks:

| Block | Kernel | Stride | Pool | Output Size |
|-------|--------|--------|------|-------------|
| 1 | 3×3×3 | 1 | 2×2×2 | 96³ |
| 2 | 5×5×5 | 2 | 2×2×2 | 24³ |
| 3 | 7×7×7 | 2 | 2×2×2 | 6³ |
| 4 | 9×9×9 | 1 | 2×2×2 | 3³ |

Classifier: Flatten(1728) → FC(512) → FC(100) → FC(3)

- Input: `[B, 1, 192, 192, 192]` preprocessed MRI volumes
- Output: `[B, 3]` logits for CN / MCI / AD

## Data Structure

```
data/
└── 3D_tensors/
    ├── CN/
    │   ├── sub-ADNI{ID}_ses-{session}_*.pt
    │   └── ...
    ├── MCI/
    │   └── ...
    └── AD/
        └── ...
```

Each `.pt` file is a float32 tensor of shape `[1, 192, 192, 192]`.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train

```bash
python3 divnet_train.py --config divnet_config.yaml
```

### Test only (with saved checkpoint)

```bash
python3 divnet_train.py --config divnet_config.yaml --test
python3 divnet_train.py --config divnet_config.yaml --test --resume checkpoints/divnet/best_balanced_acc.pth
```

### GPU selection

```bash
python3 divnet_train.py --config divnet_config.yaml --gpu 0
```

## Configuration

All hyperparameters are in `divnet_config.yaml`:

| Section | Parameter | Default |
|---------|-----------|---------|
| data | batch_size | 2 |
| data | train / val / test ratio | 0.70 / 0.15 / 0.15 |
| model | num_filters | 64 |
| model | dropout1 / dropout2 | 0.5 / 0.3 |
| training | optimizer | SGD |
| training | lr | 0.001 |
| training | lr_milestones | [30, 60, 80] |
| training | early_stopping_patience | 20 |

## Key Features

- **Patient-level splitting**: Splits by subject ID to prevent data leakage across train/val/test sets
- **Class-balanced training**: Weighted random sampler + weighted cross-entropy loss
- **Data augmentation**: Gaussian noise and random intensity shift (training only)
- **Multi-metric checkpointing**: Saves best models by accuracy, balanced accuracy, macro AUC, and lowest loss
- **Early stopping**: Based on validation loss with configurable patience

## Files

| File | Description |
|------|-------------|
| `divnet_model.py` | DivNet model architecture |
| `divnet_dataset.py` | Dataset, data loading, patient-level splitting |
| `divnet_train.py` | Training and evaluation script |
| `divnet_config.yaml` | Hyperparameter configuration |
