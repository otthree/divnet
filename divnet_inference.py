"""
5-Fold inference script for CR (Cognitive Reserve) candidate identification.

Reproduces the same 5-fold splits used during training, loads each fold's
best checkpoint, and identifies subjects whose predictions are discordant
with their true labels.

CR Groups:
    Group 1: True CN(0) -> Predicted MCI(1)
    Group 2: True MCI(1) -> Predicted AD(2)
    Group 3: True CN(0) -> Predicted AD(2)

Usage:
    python3 divnet_inference.py --config divnet_config.yaml
    python3 divnet_inference.py --config divnet_config.yaml --gpu 0
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from divnet_dataset import (
    collect_file_paths,
    patient_stratified_kfold,
    extract_patient_id,
    ADNIDataset,
    CLASS_MAP,
)
from divnet_model import DivNet
from divnet_train import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="DivNet 5-fold inference for CR candidate identification"
    )
    parser.add_argument(
        "--config", type=str, default="divnet_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    return parser.parse_args()


def load_config(config_path):
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


@torch.no_grad()
def run_inference(model, loader, device):
    """Run inference and return per-sample predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []

    for volumes, labels in loader:
        volumes = volumes.to(device, non_blocking=True)
        outputs = model(volumes)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    save_cfg = cfg["save"]
    cv_cfg = cfg.get("cross_validation", {})
    k_folds = cv_cfg.get("k_folds", 5)
    class_names = list(CLASS_MAP.keys())
    num_classes = model_cfg["num_classes"]

    # Collect all file paths and reproduce the same 5-fold splits
    print("Loading data...")
    paths, labels = collect_file_paths(data_cfg["data_root"])
    if len(paths) == 0:
        raise RuntimeError(
            f"No .pt files found in {data_cfg['data_root']}/3d-tensors/{{CN,MCI,AD}}/. "
            "Check data_root in config."
        )

    folds_data = patient_stratified_kfold(
        paths, labels, n_folds=k_folds, seed=data_cfg["seed"]
    )

    # CR group collectors (union across folds)
    # Each group stores tuples of (patient_id, image_id)
    cr_group1_all = []  # True CN -> Pred MCI
    cr_group2_all = []  # True MCI -> Pred AD
    cr_group3_all = []  # True CN -> Pred AD

    for fold_idx in range(k_folds):
        print(f"\n{'#'*60}")
        print(f"  FOLD {fold_idx} INFERENCE")
        print(f"{'#'*60}")

        _, _, val_paths, val_labels = folds_data[fold_idx]

        # Build val DataLoader (no augmentation)
        val_dataset = ADNIDataset(val_paths, val_labels, augment=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=data_cfg["val_batch_size"],
            shuffle=False,
            num_workers=data_cfg["num_workers"],
            pin_memory=True,
        )

        # Load checkpoint
        ckpt_path = os.path.join(
            save_cfg["checkpoint_dir"], f"fold_{fold_idx}", "best_balanced_acc.pth"
        )
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found: {ckpt_path}, skipping fold.")
            continue

        model = DivNet(
            num_filters=model_cfg["num_filters"],
            num_classes=num_classes,
            dropout1=model_cfg["dropout1"],
            dropout2=model_cfg["dropout2"],
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {ckpt_path} (epoch {ckpt.get('epoch', '?')})")

        # Inference
        preds, true_labels = run_inference(model, val_loader, device)

        # Confusion matrix
        cm = confusion_matrix(true_labels, preds, labels=list(range(num_classes)))
        print(f"\n  Confusion Matrix (Fold {fold_idx}):")
        header = "          " + "  ".join(f"{n:>5}" for n in class_names)
        print(f"  {header}")
        for i, name in enumerate(class_names):
            row = "  ".join(f"{cm[i, j]:5d}" for j in range(num_classes))
            print(f"  {name:>8}  {row}")

        # Save confusion matrix PNG
        fig_dir = os.path.join(save_cfg["checkpoint_dir"], "figures")
        plot_confusion_matrix(
            cm, class_names,
            phase=f"Fold {fold_idx} Inference",
            save_dir=fig_dir,
        )

        # Extract CR candidates for this fold
        fold_g1 = []  # CN -> MCI
        fold_g2 = []  # MCI -> AD
        fold_g3 = []  # CN -> AD

        for i in range(len(val_paths)):
            true_lbl = true_labels[i]
            pred_lbl = preds[i]
            fpath = val_paths[i]
            patient_id = extract_patient_id(fpath)
            image_id = os.path.splitext(os.path.basename(fpath))[0]

            if true_lbl == 0 and pred_lbl == 1:
                fold_g1.append((patient_id, image_id))
            elif true_lbl == 1 and pred_lbl == 2:
                fold_g2.append((patient_id, image_id))
            elif true_lbl == 0 and pred_lbl == 2:
                fold_g3.append((patient_id, image_id))

        # Print fold-level CR candidates
        print(f"\n  CR Candidates (Fold {fold_idx}):")
        print(f"    Group 1 (CN -> MCI): {len(fold_g1)} subjects")
        for pid, iid in fold_g1:
            print(f"      patient={pid}  image={iid}")

        print(f"    Group 2 (MCI -> AD): {len(fold_g2)} subjects")
        for pid, iid in fold_g2:
            print(f"      patient={pid}  image={iid}")

        print(f"    Group 3 (CN -> AD): {len(fold_g3)} subjects")
        for pid, iid in fold_g3:
            print(f"      patient={pid}  image={iid}")

        cr_group1_all.extend(fold_g1)
        cr_group2_all.extend(fold_g2)
        cr_group3_all.extend(fold_g3)

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Overall summary (union across folds)
    print(f"\n{'='*60}")
    print(f"  CR Candidate Summary (All Folds)")
    print(f"{'='*60}")

    # Deduplicate by patient ID (keep unique patient IDs per group)
    def unique_patients(group):
        seen = set()
        unique = []
        for pid, iid in group:
            if pid not in seen:
                seen.add(pid)
                unique.append(pid)
        return unique

    g1_patients = unique_patients(cr_group1_all)
    g2_patients = unique_patients(cr_group2_all)
    g3_patients = unique_patients(cr_group3_all)

    print(f"\n  Group 1 (True CN -> Pred MCI): {len(g1_patients)} unique patients")
    for pid in g1_patients:
        print(f"    {pid}")

    print(f"\n  Group 2 (True MCI -> Pred AD): {len(g2_patients)} unique patients")
    for pid in g2_patients:
        print(f"    {pid}")

    print(f"\n  Group 3 (True CN -> Pred AD): {len(g3_patients)} unique patients")
    for pid in g3_patients:
        print(f"    {pid}")

    # Also print all image-level IDs
    print(f"\n  --- Image-level detail ---")
    print(f"\n  Group 1 (CN -> MCI): {len(cr_group1_all)} images")
    for pid, iid in cr_group1_all:
        print(f"    patient={pid}  image={iid}")

    print(f"\n  Group 2 (MCI -> AD): {len(cr_group2_all)} images")
    for pid, iid in cr_group2_all:
        print(f"    patient={pid}  image={iid}")

    print(f"\n  Group 3 (CN -> AD): {len(cr_group3_all)} images")
    for pid, iid in cr_group3_all:
        print(f"    patient={pid}  image={iid}")

    print(f"\n{'='*60}")
    print(f"  Total unique CR candidate patients: "
          f"{len(set(g1_patients + g2_patients + g3_patients))}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
