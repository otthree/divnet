"""
Training script for divNet: Diverging 3D CNN for AD classification.

Usage:
    python divnet_train.py --config divnet_config.yaml
    python divnet_train.py --config divnet_config.yaml --test    # test-only mode
    python divnet_train.py --config divnet_config.yaml --kfold   # 5-fold CV mode
"""

import argparse
import datetime
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import wandb

from divnet_model import DivNet
from divnet_dataset import (
    build_dataloaders,
    build_dataloaders_kfold,
    build_exclude_set,
    collect_file_paths,
    patient_stratified_kfold,
    CLASS_MAP,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train divNet for AD classification")
    parser.add_argument("--config", type=str, default="divnet_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--test", action="store_true",
                        help="Run test evaluation only (requires checkpoint)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training or run test")
    parser.add_argument("--kfold", action="store_true",
                        help="Run k-fold cross-validation mode")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_checkpoint(state, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (volumes, labels) in enumerate(loader):
        volumes = volumes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(volumes)
        loss = criterion(outputs, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        running_loss += loss.item() * volumes.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes=3):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for volumes, labels in loader:
        volumes = volumes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(volumes)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * volumes.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    total = len(all_labels)
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * np.sum(all_preds == all_labels) / total
    balanced_acc = 100.0 * balanced_accuracy_score(all_labels, all_preds)

    # F1 scores
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # Per-class and overall AUC
    auc_results = compute_auc(all_labels, all_probs, num_classes)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    metrics = {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "balanced_accuracy": balanced_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
    }
    for c in range(num_classes):
        metrics[f"f1_class_{c}"] = f1_per_class[c] if c < len(f1_per_class) else 0.0
    metrics.update(auc_results)

    return metrics


def compute_auc(labels, probs, num_classes):
    """Compute per-class AUC, micro AUC, and macro AUC."""
    results = {}

    # Check if all classes are present
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        results["micro_auc"] = 0.0
        results["macro_auc"] = 0.0
        for c in range(num_classes):
            results[f"auc_class_{c}"] = 0.0
        return results

    # One-vs-rest binarized labels
    labels_onehot = np.zeros((len(labels), num_classes))
    for i, lbl in enumerate(labels):
        labels_onehot[i, lbl] = 1

    # Per-class AUC
    per_class_auc = []
    for c in range(num_classes):
        if c in unique_labels and np.sum(labels_onehot[:, c]) > 0:
            try:
                auc_c = roc_auc_score(labels_onehot[:, c], probs[:, c])
            except ValueError:
                auc_c = 0.0
        else:
            auc_c = 0.0
        results[f"auc_class_{c}"] = auc_c
        per_class_auc.append(auc_c)

    # Macro AUC (average of per-class)
    valid_aucs = [a for a in per_class_auc if a > 0]
    results["macro_auc"] = np.mean(valid_aucs) if valid_aucs else 0.0

    # Micro AUC
    try:
        results["micro_auc"] = roc_auc_score(
            labels_onehot, probs, average="micro"
        )
    except ValueError:
        results["micro_auc"] = 0.0

    return results


def compute_per_class_metrics(cm, class_names):
    """Compute sensitivity and specificity per class from confusion matrix."""
    num_classes = cm.shape[0]
    metrics = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics[class_names[i]] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": (tp + tn) / np.sum(cm) if np.sum(cm) > 0 else 0.0,
        }
    return metrics


def plot_confusion_matrix(cm, class_names, phase="Validation", save_dir="figures"):
    """Plot and save a confusion matrix as a heatmap."""
    os.makedirs(save_dir, exist_ok=True)

    # Flip rows so y-axis goes CN(bottom) -> MCI -> AD(top)
    cm_flipped = cm[::-1, :]
    class_names_y = list(reversed(class_names))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_flipped, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names_y,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{phase} Confusion Matrix",
    )

    # Text annotations
    thresh = cm_flipped.max() / 2.0
    for i in range(cm_flipped.shape[0]):
        for j in range(cm_flipped.shape[1]):
            ax.text(
                j, i, format(cm_flipped[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_flipped[i, j] > thresh else "black",
                fontsize=14,
            )

    fig.tight_layout()
    filename = f"{phase.lower().replace(' ', '_')}_confusion_matrix.png"
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved to {save_path}")


def print_metrics(metrics, class_names, phase="Validation"):
    """Print formatted metrics."""
    print(f"\n{'='*60}")
    print(f"  {phase} Results")
    print(f"{'='*60}")
    print(f"  Loss:              {metrics['loss']:.4f}")
    print(f"  Accuracy:          {metrics['accuracy']:.2f}%")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.2f}%")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
    print(f"  Micro AUC:         {metrics['micro_auc']:.4f}")
    print(f"  Macro AUC:         {metrics['macro_auc']:.4f}")

    for i, name in enumerate(class_names):
        print(f"  AUC ({name}):         {metrics[f'auc_class_{i}']:.4f}")
        print(f"  F1  ({name}):         {metrics[f'f1_class_{i}']:.4f}")

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix:")
    header = "          " + "  ".join(f"{n:>5}" for n in class_names)
    print(f"  {header}")
    for i, name in enumerate(class_names):
        row = "  ".join(f"{cm[i, j]:5d}" for j in range(len(class_names)))
        print(f"  {name:>8}  {row}")

    # Per-class sensitivity/specificity
    per_class = compute_per_class_metrics(cm, class_names)
    print(f"\n  Per-class metrics:")
    for name in class_names:
        m = per_class[name]
        print(f"    {name}: Sensitivity={m['sensitivity']:.4f}  "
              f"Specificity={m['specificity']:.4f}")
    print(f"{'='*60}\n")


def train(cfg, device):
    """Main training loop."""
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    save_cfg = cfg["save"]

    class_names = list(CLASS_MAP.keys())

    # Build dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(cfg)
    print(f"Class weights: {class_weights.tolist()}")

    # Model
    model = DivNet(
        num_filters=model_cfg["num_filters"],
        num_classes=model_cfg["num_classes"],
        dropout1=model_cfg["dropout1"],
        dropout2=model_cfg["dropout2"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Optimizer
    if train_cfg.get("optimizer", "SGD").upper() == "ADAM":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_cfg["lr"],
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=train_cfg["weight_decay"],
        )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=train_cfg["lr_milestones"],
        gamma=train_cfg["lr_gamma"],
    )

    # Tracking best metrics
    best = {
        "accuracy": 0.0,
        "balanced_accuracy": 0.0,
        "macro_auc": 0.0,
        "loss": float("inf"),
    }
    patience_counter = 0

    # wandb init
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", True):
        wandb.login()
        wandb.init(
            project=wandb_cfg.get("project", "DivNet AD Classification"),
            name=f"DivNet {datetime.datetime.now()}",
            config=cfg,
        )

    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print(f"Device: {device}")
    print(f"Batch size: {cfg['data']['batch_size']}")
    print(f"LR: {train_cfg['lr']}, Milestones: {train_cfg['lr_milestones']}\n")

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=train_cfg["grad_clip"],
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device,
                               num_classes=model_cfg["num_classes"])

        scheduler.step()
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch:3d}/{train_cfg['epochs']}] "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
              f"Val Loss: {val_metrics['loss']:.4f}  "
              f"Val Acc: {val_metrics['accuracy']:.2f}%  "
              f"Val BAcc: {val_metrics['balanced_accuracy']:.2f}%  "
              f"Val mAUC: {val_metrics['macro_auc']:.4f}  "
              f"LR: {current_lr:.6f}  "
              f"Time: {elapsed:.1f}s")

        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_balanced_acc": val_metrics["balanced_accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_f1_weighted": val_metrics["f1_weighted"],
                "val_macro_auc": val_metrics["macro_auc"],
                "lr": current_lr,
            })

        # Checkpoint saving
        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics": val_metrics,
        }

        checkpoint_dir = save_cfg["checkpoint_dir"]

        # Best accuracy
        if val_metrics["accuracy"] > best["accuracy"]:
            best["accuracy"] = val_metrics["accuracy"]
            save_checkpoint(
                checkpoint_state,
                os.path.join(checkpoint_dir, "best_accuracy.pth"),
            )
            print(f"  -> New best accuracy: {best['accuracy']:.2f}%")

        # Best balanced accuracy
        if val_metrics["balanced_accuracy"] > best["balanced_accuracy"]:
            best["balanced_accuracy"] = val_metrics["balanced_accuracy"]
            save_checkpoint(
                checkpoint_state,
                os.path.join(checkpoint_dir, "best_balanced_acc.pth"),
            )
            print(f"  -> New best balanced accuracy: {best['balanced_accuracy']:.2f}%")

        # Best macro AUC
        if val_metrics["macro_auc"] > best["macro_auc"]:
            best["macro_auc"] = val_metrics["macro_auc"]
            save_checkpoint(
                checkpoint_state,
                os.path.join(checkpoint_dir, "best_macro_auc.pth"),
            )
            print(f"  -> New best macro AUC: {best['macro_auc']:.4f}")

        # Lowest loss
        if val_metrics["loss"] < best["loss"]:
            best["loss"] = val_metrics["loss"]
            patience_counter = 0
            save_checkpoint(
                checkpoint_state,
                os.path.join(checkpoint_dir, "lowest_loss.pth"),
            )
            print(f"  -> New lowest loss: {best['loss']:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= train_cfg["early_stopping_patience"]:
            print(f"\nEarly stopping triggered after {epoch} epochs "
                  f"(patience={train_cfg['early_stopping_patience']})")
            break

    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best Accuracy:          {best['accuracy']:.2f}%")
    print(f"Best Balanced Accuracy: {best['balanced_accuracy']:.2f}%")
    print(f"Best Macro AUC:         {best['macro_auc']:.4f}")
    print(f"Lowest Val Loss:        {best['loss']:.4f}")

    # Evaluate on test set with best balanced accuracy checkpoint
    best_ckpt_path = os.path.join(checkpoint_dir, "best_balanced_acc.pth")
    if os.path.exists(best_ckpt_path):
        print("\nEvaluating best model on test set...")
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = validate(model, test_loader, criterion, device,
                                num_classes=model_cfg["num_classes"])
        print_metrics(test_metrics, class_names, phase="Test")
        save_dir = os.path.join(checkpoint_dir, "figures")
        plot_confusion_matrix(test_metrics["confusion_matrix"], class_names,
                              phase="Test", save_dir=save_dir)

        if wandb.run is not None:
            wandb.log({
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["accuracy"],
                "test_balanced_acc": test_metrics["balanced_accuracy"],
                "test_f1_macro": test_metrics["f1_macro"],
                "test_f1_weighted": test_metrics["f1_weighted"],
                "test_macro_auc": test_metrics["macro_auc"],
            })

    if wandb.run is not None:
        wandb.finish()

    return model


def train_kfold(cfg, device):
    """K-fold cross-validation training loop."""
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    save_cfg = cfg["save"]
    cv_cfg = cfg.get("cross_validation", {})
    k_folds = cv_cfg.get("k_folds", 5)

    class_names = list(CLASS_MAP.keys())
    num_classes = model_cfg["num_classes"]

    # Collect all file paths and create folds
    print("Loading data...")
    data_cfg = cfg["data"]

    exclude_indices = None
    scan_csv = data_cfg.get("scan_csv")
    exclude_csv = data_cfg.get("exclude_csv")
    if scan_csv and exclude_csv:
        exclude_indices = build_exclude_set(scan_csv, exclude_csv)

    paths, labels = collect_file_paths(data_cfg["data_root"], exclude_indices=exclude_indices)
    if len(paths) == 0:
        raise RuntimeError(
            f"No .pt files found in {data_cfg['data_root']}/3d-tensors/{{CN,MCI,AD}}/. "
            "Check data_root in config."
        )

    folds_data = patient_stratified_kfold(paths, labels, n_folds=k_folds, seed=data_cfg["seed"])

    # Collect per-fold best metrics
    all_fold_metrics = []
    wandb_cfg = cfg.get("wandb", {})

    for fold_idx in range(k_folds):
        print(f"\n{'#'*60}")
        print(f"  FOLD {fold_idx + 1} / {k_folds}")
        print(f"{'#'*60}\n")

        # wandb init per fold
        if wandb_cfg.get("enabled", True):
            wandb.login()
            wandb.init(
                project=wandb_cfg.get("project", "DivNet AD Classification"),
                name=f"DivNet fold-{fold_idx}",
                group="kfold_cv",
                config={**cfg, "fold": fold_idx},
                reinit=True,
            )

        train_loader, val_loader, class_weights = build_dataloaders_kfold(cfg, fold_idx, folds_data)
        print(f"Class weights: {class_weights.tolist()}")

        # Fresh model
        model = DivNet(
            num_filters=model_cfg["num_filters"],
            num_classes=num_classes,
            dropout1=model_cfg["dropout1"],
            dropout2=model_cfg["dropout2"],
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        if train_cfg.get("optimizer", "SGD").upper() == "ADAM":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_cfg["lr"],
                weight_decay=train_cfg["weight_decay"],
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=train_cfg["lr"],
                momentum=train_cfg.get("momentum", 0.9),
                weight_decay=train_cfg["weight_decay"],
            )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=train_cfg["lr_milestones"],
            gamma=train_cfg["lr_gamma"],
        )

        best = {
            "balanced_accuracy": 0.0,
            "loss": float("inf"),
        }
        best_metrics = None
        patience_counter = 0

        fold_ckpt_dir = os.path.join(save_cfg["checkpoint_dir"], f"fold_{fold_idx}")

        for epoch in range(1, train_cfg["epochs"] + 1):
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                grad_clip=train_cfg["grad_clip"],
            )
            val_metrics = validate(model, val_loader, criterion, device, num_classes=num_classes)

            scheduler.step()
            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"Fold {fold_idx} Epoch [{epoch:3d}/{train_cfg['epochs']}] "
                  f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
                  f"Val Loss: {val_metrics['loss']:.4f}  "
                  f"Val BAcc: {val_metrics['balanced_accuracy']:.2f}%  "
                  f"Val mAUC: {val_metrics['macro_auc']:.4f}  "
                  f"LR: {current_lr:.6f}  Time: {elapsed:.1f}s")

            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_balanced_acc": val_metrics["balanced_accuracy"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "val_f1_weighted": val_metrics["f1_weighted"],
                    "val_macro_auc": val_metrics["macro_auc"],
                    "lr": current_lr,
                })

            # Track best by balanced accuracy
            if val_metrics["balanced_accuracy"] > best["balanced_accuracy"]:
                best["balanced_accuracy"] = val_metrics["balanced_accuracy"]
                best_metrics = val_metrics.copy()
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "fold": fold_idx,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    os.path.join(fold_ckpt_dir, "best_balanced_acc.pth"),
                )
                print(f"  -> New best balanced accuracy: {best['balanced_accuracy']:.2f}%")

            # Early stopping on val loss
            if val_metrics["loss"] < best["loss"]:
                best["loss"] = val_metrics["loss"]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= train_cfg["early_stopping_patience"]:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(patience={train_cfg['early_stopping_patience']})")
                break

        # Load best checkpoint and re-evaluate for this fold
        best_ckpt = os.path.join(fold_ckpt_dir, "best_balanced_acc.pth")
        if os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            best_metrics = validate(model, val_loader, criterion, device, num_classes=num_classes)

        print_metrics(best_metrics, class_names, phase=f"Fold {fold_idx} Best Val")
        fold_fig_dir = os.path.join(save_cfg["checkpoint_dir"], "figures")
        plot_confusion_matrix(best_metrics["confusion_matrix"], class_names,
                              phase=f"Fold {fold_idx}", save_dir=fold_fig_dir)

        # Collect metrics for summary
        cm = best_metrics["confusion_matrix"]
        per_class = compute_per_class_metrics(cm, class_names)
        fold_record = {
            "accuracy": best_metrics["accuracy"],
            "balanced_accuracy": best_metrics["balanced_accuracy"],
            "f1_macro": best_metrics["f1_macro"],
            "f1_weighted": best_metrics["f1_weighted"],
            "macro_auc": best_metrics["macro_auc"],
        }
        for i, name in enumerate(class_names):
            fold_record[f"auc_{name}"] = best_metrics[f"auc_class_{i}"]
            fold_record[f"sensitivity_{name}"] = per_class[name]["sensitivity"]
            fold_record[f"specificity_{name}"] = per_class[name]["specificity"]
        all_fold_metrics.append(fold_record)

        if wandb.run is not None:
            wandb.finish()

    # ---- Summary across all folds ----
    print(f"\n{'='*60}")
    print(f"  {k_folds}-Fold Cross-Validation Summary")
    print(f"{'='*60}")

    metric_keys = list(all_fold_metrics[0].keys())
    for key in metric_keys:
        values = [m[key] for m in all_fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {key:30s}: {mean_val:.4f} +/- {std_val:.4f}")

    print(f"{'='*60}\n")


def test_only(cfg, checkpoint_path, device):
    """Run evaluation on test set using a saved checkpoint."""
    model_cfg = cfg["model"]
    class_names = list(CLASS_MAP.keys())

    # Build dataloaders
    print("Loading data...")
    _, val_loader, test_loader, class_weights = build_dataloaders(cfg)

    # Model
    model = DivNet(
        num_filters=model_cfg["num_filters"],
        num_classes=model_cfg["num_classes"],
        dropout1=model_cfg["dropout1"],
        dropout2=model_cfg["dropout2"],
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint from epoch {checkpoint.get('epoch', '?')}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    save_dir = os.path.join(cfg["save"]["checkpoint_dir"], "figures")

    # Evaluate on validation set
    val_metrics = validate(model, val_loader, criterion, device,
                           num_classes=model_cfg["num_classes"])
    print_metrics(val_metrics, class_names, phase="Validation")
    plot_confusion_matrix(val_metrics["confusion_matrix"], class_names,
                          phase="Validation", save_dir=save_dir)

    # Evaluate on test set
    test_metrics = validate(model, test_loader, criterion, device,
                            num_classes=model_cfg["num_classes"])
    print_metrics(test_metrics, class_names, phase="Test")
    plot_confusion_matrix(test_metrics["confusion_matrix"], class_names,
                          phase="Test", save_dir=save_dir)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if args.test:
        # Test-only mode
        checkpoint_path = args.resume
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                cfg["save"]["checkpoint_dir"], "best_balanced_acc.pth"
            )
        test_only(cfg, checkpoint_path, device)
    elif args.kfold:
        # K-fold cross-validation mode
        train_kfold(cfg, device)
    else:
        # Training mode
        train(cfg, device)


if __name__ == "__main__":
    main()
