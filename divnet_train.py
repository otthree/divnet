"""
Training script for divNet: Diverging 3D CNN for AD classification.

Usage:
    python divnet_train.py --config divnet_config.yaml
    python divnet_train.py --config divnet_config.yaml --test  # test-only mode
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

from divnet_model import DivNet
from divnet_dataset import build_dataloaders, CLASS_MAP


def parse_args():
    parser = argparse.ArgumentParser(description="Train divNet for AD classification")
    parser.add_argument("--config", type=str, default="divnet_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--test", action="store_true",
                        help="Run test evaluation only (requires checkpoint)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training or run test")
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

    # Per-class and overall AUC
    auc_results = compute_auc(all_labels, all_probs, num_classes)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    metrics = {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm,
    }
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


def print_metrics(metrics, class_names, phase="Validation"):
    """Print formatted metrics."""
    print(f"\n{'='*60}")
    print(f"  {phase} Results")
    print(f"{'='*60}")
    print(f"  Loss:              {metrics['loss']:.4f}")
    print(f"  Accuracy:          {metrics['accuracy']:.2f}%")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.2f}%")
    print(f"  Micro AUC:         {metrics['micro_auc']:.4f}")
    print(f"  Macro AUC:         {metrics['macro_auc']:.4f}")

    for i, name in enumerate(class_names):
        print(f"  AUC ({name}):         {metrics[f'auc_class_{i}']:.4f}")

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
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["lr"],
        momentum=train_cfg["momentum"],
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
        checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = validate(model, test_loader, criterion, device,
                                num_classes=model_cfg["num_classes"])
        print_metrics(test_metrics, class_names, phase="Test")

    return model


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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint from epoch {checkpoint.get('epoch', '?')}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # Evaluate on validation set
    val_metrics = validate(model, val_loader, criterion, device,
                           num_classes=model_cfg["num_classes"])
    print_metrics(val_metrics, class_names, phase="Validation")

    # Evaluate on test set
    test_metrics = validate(model, test_loader, criterion, device,
                            num_classes=model_cfg["num_classes"])
    print_metrics(test_metrics, class_names, phase="Test")


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
    else:
        # Training mode
        train(cfg, device)


if __name__ == "__main__":
    main()
