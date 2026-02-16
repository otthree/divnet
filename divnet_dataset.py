"""
Dataset and data loading utilities for divNet training.

Loads preprocessed .pt MRI volumes from:
    {data_root}/3D_tensors/{CN,MCI,AD}/*.pt

Each .pt file: float32 tensor of shape [1, 192, 192, 192]
Labels: CN=0, MCI=1, AD=2
"""

import os
import random
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


CLASS_MAP = {"CN": 0, "MCI": 1, "AD": 2}


def collect_file_paths(data_root):
    """Scan data_root/3D_tensors/{CN,MCI,AD}/ and return (paths, labels)."""
    tensor_dir = os.path.join(data_root, "3D_tensors")
    paths = []
    labels = []

    for class_name, label in CLASS_MAP.items():
        class_dir = os.path.join(tensor_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: directory not found: {class_dir}")
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.endswith(".pt"):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label)

    print(f"Found {len(paths)} total samples: {dict(Counter(labels))}")
    return paths, labels


def extract_patient_id(filepath):
    """
    Extract patient ID from filename for patient-level splitting.
    Assumes format: sub-ADNI{PATIENT_ID}_ses-{session}_*.pt
    Falls back to full filename if pattern does not match.
    """
    fname = os.path.basename(filepath)
    # Try to extract subject-level ID (everything before _ses-)
    if "_ses-" in fname:
        return fname.split("_ses-")[0]
    # Fallback: use the full filename stem as unique ID
    return os.path.splitext(fname)[0]


def patient_stratified_split(paths, labels, train_ratio, val_ratio, test_ratio, seed):
    """
    Split data by patient ID (stratified) to prevent data leakage.
    Returns (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels).
    """
    # Group files by patient
    patient_to_indices = {}
    for idx, path in enumerate(paths):
        pid = extract_patient_id(path)
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
        patient_to_indices[pid].append(idx)

    # Assign each patient a single label (majority label across their scans)
    patient_ids = list(patient_to_indices.keys())
    patient_labels = []
    for pid in patient_ids:
        pid_labels = [labels[i] for i in patient_to_indices[pid]]
        patient_labels.append(max(set(pid_labels), key=pid_labels.count))

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_pids, valtest_pids, train_plabels, valtest_plabels = train_test_split(
        patient_ids, patient_labels,
        test_size=val_test_ratio,
        stratify=patient_labels,
        random_state=seed,
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    val_pids, test_pids, _, _ = train_test_split(
        valtest_pids, valtest_plabels,
        test_size=relative_test_ratio,
        stratify=valtest_plabels,
        random_state=seed,
    )

    # Convert patient IDs back to file indices
    def pids_to_data(pids):
        indices = []
        for pid in pids:
            indices.extend(patient_to_indices[pid])
        return [paths[i] for i in indices], [labels[i] for i in indices]

    train_data = pids_to_data(train_pids)
    val_data = pids_to_data(val_pids)
    test_data = pids_to_data(test_pids)

    print(f"Split (patients): train={len(train_pids)}, val={len(val_pids)}, test={len(test_pids)}")
    print(f"Split (scans):    train={len(train_data[0])}, val={len(val_data[0])}, test={len(test_data[0])}")

    return train_data, val_data, test_data


class ADNIDataset(Dataset):
    """
    Dataset for loading preprocessed 3D MRI .pt files.

    Args:
        paths: list of .pt file paths
        labels: list of integer labels (CN=0, MCI=1, AD=2)
        augment: whether to apply training augmentation
        noise_std: std of Gaussian noise augmentation
        intensity_shift: max intensity shift magnitude
    """

    def __init__(self, paths, labels, augment=False, noise_std=0.01, intensity_shift=0.1):
        self.paths = paths
        self.labels = labels
        self.augment = augment
        self.noise_std = noise_std
        self.intensity_shift = intensity_shift

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        volume = torch.load(self.paths[idx], map_location="cpu", weights_only=True)

        # Ensure shape [1, 192, 192, 192]
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)

        # Per-volume min-max normalization to [0, 1]
        vmin = volume.min()
        vmax = volume.max()
        if vmax - vmin > 1e-8:
            volume = (volume - vmin) / (vmax - vmin)
        else:
            volume = torch.zeros_like(volume)

        # Training augmentation
        if self.augment:
            # Random Gaussian noise
            noise = torch.randn_like(volume) * self.noise_std
            volume = volume + noise

            # Random intensity shift
            shift = random.uniform(-self.intensity_shift, self.intensity_shift)
            volume = volume + shift

            # Clamp back to valid range
            volume = volume.clamp(0.0, 1.0)

        label = self.labels[idx]
        return volume, label


def compute_class_weights(labels):
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    counts = Counter(labels)
    total = len(labels)
    num_classes = len(counts)
    weights = torch.zeros(num_classes)
    for cls, count in counts.items():
        weights[cls] = total / (num_classes * count)
    return weights


def make_weighted_sampler(labels):
    """Create WeightedRandomSampler for class-balanced training."""
    counts = Counter(labels)
    class_weight = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weight[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def build_dataloaders(cfg):
    """
    Build train/val/test DataLoaders from config.

    Args:
        cfg: parsed YAML config dict

    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    data_cfg = cfg["data"]

    # Collect all file paths
    paths, labels = collect_file_paths(data_cfg["data_root"])

    if len(paths) == 0:
        raise RuntimeError(
            f"No .pt files found in {data_cfg['data_root']}/3D_tensors/{{CN,MCI,AD}}/. "
            "Check data_root in config."
        )

    # Patient-level stratified split
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
        patient_stratified_split(
            paths, labels,
            train_ratio=data_cfg["train_ratio"],
            val_ratio=data_cfg["val_ratio"],
            test_ratio=data_cfg["test_ratio"],
            seed=data_cfg["seed"],
        )

    # Datasets
    train_dataset = ADNIDataset(train_paths, train_labels, augment=True)
    val_dataset = ADNIDataset(val_paths, val_labels, augment=False)
    test_dataset = ADNIDataset(test_paths, test_labels, augment=False)

    # Weighted sampler for training
    train_sampler = make_weighted_sampler(train_labels)

    # Class weights for loss
    class_weights = compute_class_weights(train_labels)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        sampler=train_sampler,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["val_batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg["val_batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights
