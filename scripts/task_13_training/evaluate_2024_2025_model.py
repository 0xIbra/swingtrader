#!/usr/bin/env python3
"""
Evaluate 2024-2025 model on test set.
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir / '../task_11_pytorch_dataset'))
sys.path.insert(0, str(script_dir / '../task_12_model'))

import torch
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from tcn_model import DualHeadTCN
from pytorch_dataset import SequenceDataset
from torch.utils.data import DataLoader


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute per-class and macro metrics."""
    acc = accuracy_score(y_true, y_pred)

    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_flat = f1_per_class[0] if len(f1_per_class) > 0 else 0.0
    f1_long = f1_per_class[1] if len(f1_per_class) > 1 else 0.0
    f1_short = f1_per_class[2] if len(f1_per_class) > 2 else 0.0

    # Macro F1
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class precision & recall
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_flat': f1_flat,
        'f1_long': f1_long,
        'f1_short': f1_short,
        'precision_flat': precision_per_class[0] if len(precision_per_class) > 0 else 0.0,
        'precision_long': precision_per_class[1] if len(precision_per_class) > 1 else 0.0,
        'precision_short': precision_per_class[2] if len(precision_per_class) > 2 else 0.0,
        'recall_flat': recall_per_class[0] if len(recall_per_class) > 0 else 0.0,
        'recall_long': recall_per_class[1] if len(recall_per_class) > 1 else 0.0,
        'recall_short': recall_per_class[2] if len(recall_per_class) > 2 else 0.0,
        'confusion_matrix': cm
    }


def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions and metrics."""
    model.eval()

    all_preds = []
    all_targets = []
    all_reg_preds = []
    all_reg_targets = []

    with torch.no_grad():
        for X, y_class, y_reg in dataloader:
            X = X.to(device)
            y_class = y_class.to(device)
            y_reg = y_reg.to(device)

            class_logits, reg_out = model(X)
            preds = torch.argmax(class_logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_class.cpu().numpy())
            all_reg_preds.append(reg_out.cpu().numpy())
            all_reg_targets.append(y_reg.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    reg_pred = np.concatenate(all_reg_preds)
    reg_true = np.concatenate(all_reg_targets)

    # Classification metrics
    metrics = compute_classification_metrics(y_true, y_pred)

    # Regression metrics
    reg_mae = np.mean(np.abs(reg_pred - reg_true))
    reg_rmse = np.sqrt(np.mean((reg_pred - reg_true) ** 2))

    metrics['reg_mae'] = reg_mae
    metrics['reg_rmse'] = reg_rmse

    return metrics, y_true, y_pred


def main():
    print("="*80)
    print("2024-2025 MODEL EVALUATION")
    print("="*80)

    # Paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # scripts/task_13_training -> scripts -> project_root
    data_dir = project_root / 'data'
    checkpoint_dir = project_root / 'checkpoints_2024_2025'

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"\n🖥️  Using device: {device}")

    # Load datasets
    print(f"\n📦 Loading 2024-2025 datasets...")
    train_dataset = SequenceDataset(str(data_dir / 'train_normalized_2024_2025.pkl'))
    val_dataset = SequenceDataset(str(data_dir / 'val_normalized_2024_2025.pkl'))
    test_dataset = SequenceDataset(str(data_dir / 'test_normalized_2024_2025.pkl'))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"  ✓ Train: {len(train_dataset):,} samples (Jan 2024 - Feb 2025)")
    print(f"  ✓ Val:   {len(val_dataset):,} samples (Feb 2025 - Jul 2025)")
    print(f"  ✓ Test:  {len(test_dataset):,} samples (Jul 2025 - Nov 2025)")

    # Load model
    print(f"\n🏗️  Loading trained model...")
    model = DualHeadTCN(
        n_features=55,
        n_channels=[64, 64, 128, 128, 256],
        kernel_size=3,
        dropout=0.2,
        n_classes=3,
        n_reg_outputs=4,
        pooling='last'
    )

    checkpoint_path = checkpoint_dir / 'best_model.pt'
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"  ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    val_f1 = checkpoint.get('val_f1', 0.0)
    if isinstance(val_f1, (int, float)):
        print(f"  ✓ Val F1 (macro): {val_f1:.4f}")

    # Evaluate on all splits
    print(f"\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    for split_name, loader in [('TRAIN', train_loader), ('VAL', val_loader), ('TEST', test_loader)]:
        print(f"\n📊 {split_name} SET:")
        print("-" * 80)

        metrics, y_true, y_pred = evaluate_model(model, loader, device)

        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  F1 (macro):   {metrics['f1_macro']:.4f}")
        print(f"  F1 (FLAT):    {metrics['f1_flat']:.4f}  (Precision: {metrics['precision_flat']:.4f}, Recall: {metrics['recall_flat']:.4f})")
        print(f"  F1 (LONG):    {metrics['f1_long']:.4f}  (Precision: {metrics['precision_long']:.4f}, Recall: {metrics['recall_long']:.4f})")
        print(f"  F1 (SHORT):   {metrics['f1_short']:.4f}  (Precision: {metrics['precision_short']:.4f}, Recall: {metrics['recall_short']:.4f})")
        print(f"  Reg MAE:      {metrics['reg_mae']:.4f}")
        print(f"  Reg RMSE:     {metrics['reg_rmse']:.4f}")

        print(f"\n  Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"                Predicted")
        print(f"         FLAT   LONG  SHORT")
        print(f"  FLAT   {cm[0,0]:4d}   {cm[0,1]:4d}   {cm[0,2]:4d}")
        print(f"  LONG   {cm[1,0]:4d}   {cm[1,1]:4d}   {cm[1,2]:4d}")
        print(f"  SHORT  {cm[2,0]:4d}   {cm[2,1]:4d}   {cm[2,2]:4d}")

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

