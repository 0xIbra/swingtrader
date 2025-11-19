#!/usr/bin/env python3
"""
Evaluate trained TCN model on test set.
"""

import torch
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from tcn_model import DualHeadTCN
from pytorch_dataset import SequenceDataset, load_all_dataloaders


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_flat': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
        'f1_long': f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
        'f1_short': f1_per_class[2] if len(f1_per_class) > 2 else 0.0,
        'precision_flat': precision_per_class[0] if len(precision_per_class) > 0 else 0.0,
        'precision_long': precision_per_class[1] if len(precision_per_class) > 1 else 0.0,
        'precision_short': precision_per_class[2] if len(precision_per_class) > 2 else 0.0,
        'recall_flat': recall_per_class[0] if len(recall_per_class) > 0 else 0.0,
        'recall_long': recall_per_class[1] if len(recall_per_class) > 1 else 0.0,
        'recall_short': recall_per_class[2] if len(recall_per_class) > 2 else 0.0,
        'confusion_matrix': cm.tolist()
    }

    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    return {
        'reg_mae': mae,
        'reg_mse': mse,
        'reg_rmse': rmse
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate model on a dataset."""
    model.eval()

    all_y_true_class = []
    all_y_pred_class = []
    all_y_true_reg = []
    all_y_pred_reg = []

    for batch in dataloader:
        X, y_class, y_reg = batch
        X = X.to(device)
        y_class = y_class.to(device)
        y_reg = y_reg.to(device)

        # Forward pass
        logits, reg_out = model(X)

        # Get predictions
        y_pred = torch.argmax(logits, dim=1)

        # Collect predictions
        all_y_true_class.append(y_class.cpu().numpy())
        all_y_pred_class.append(y_pred.cpu().numpy())
        all_y_true_reg.append(y_reg.cpu().numpy())
        all_y_pred_reg.append(reg_out.cpu().numpy())

    # Concatenate all batches
    y_true_class = np.concatenate(all_y_true_class)
    y_pred_class = np.concatenate(all_y_pred_class)
    y_true_reg = np.concatenate(all_y_true_reg)
    y_pred_reg = np.concatenate(all_y_pred_reg)

    # Compute metrics
    class_metrics = compute_classification_metrics(y_true_class, y_pred_class)
    reg_metrics = compute_regression_metrics(y_true_reg, y_pred_reg)

    return {**class_metrics, **reg_metrics}


def main():
    print("="*80)
    print("MODEL EVALUATION ON TEST SET")
    print("="*80)

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / 'data'
    checkpoint_dir = project_root / 'checkpoints'

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"\n🖥️  Using device: {device}")

    # Load datasets
    print(f"\n📦 Loading datasets from: {data_dir}")
    loaders = load_all_dataloaders(str(data_dir), batch_size=64, num_workers=0)

    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    print(f"  ✓ Train: {len(train_loader.dataset):,} samples")
    print(f"  ✓ Val:   {len(val_loader.dataset):,} samples")
    print(f"  ✓ Test:  {len(test_loader.dataset):,} samples")

    # Load model
    print(f"\n🏗️  Loading trained model...")
    model = DualHeadTCN(
        n_features=55,
        n_channels=[64, 64, 128, 128, 256],  # 5 layers
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
    else:
        print(f"  ✓ Val F1 (macro): {val_f1}")

    # Evaluate on all splits
    print("\n" + "="*80)
    print("EVALUATING ON ALL SPLITS")
    print("="*80)

    print("\n📊 Evaluating on TRAINING set...")
    train_metrics = evaluate_model(model, train_loader, device)

    print("📊 Evaluating on VALIDATION set...")
    val_metrics = evaluate_model(model, val_loader, device)

    print("📊 Evaluating on TEST set...")
    test_metrics = evaluate_model(model, test_loader, device)

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    def print_metrics(split_name, metrics):
        print(f"\n{split_name} SET:")
        print(f"  Accuracy:       {metrics['accuracy']:.4f}")
        print(f"  F1 (macro):     {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted):  {metrics['f1_weighted']:.4f}")
        print(f"\n  Per-Class F1:")
        print(f"    FLAT:  {metrics['f1_flat']:.4f}")
        print(f"    LONG:  {metrics['f1_long']:.4f}")
        print(f"    SHORT: {metrics['f1_short']:.4f}")
        print(f"\n  Per-Class Precision:")
        print(f"    FLAT:  {metrics['precision_flat']:.4f}")
        print(f"    LONG:  {metrics['precision_long']:.4f}")
        print(f"    SHORT: {metrics['precision_short']:.4f}")
        print(f"\n  Per-Class Recall:")
        print(f"    FLAT:  {metrics['recall_flat']:.4f}")
        print(f"    LONG:  {metrics['recall_long']:.4f}")
        print(f"    SHORT: {metrics['recall_short']:.4f}")
        print(f"\n  Regression MAE: {metrics['reg_mae']:.4f}")
        print(f"\n  Confusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"           Pred FLAT  Pred LONG  Pred SHORT")
        print(f"  FLAT     {cm[0,0]:6d}     {cm[0,1]:6d}      {cm[0,2]:6d}")
        print(f"  LONG     {cm[1,0]:6d}     {cm[1,1]:6d}      {cm[1,2]:6d}")
        print(f"  SHORT    {cm[2,0]:6d}     {cm[2,1]:6d}      {cm[2,2]:6d}")

    print_metrics("TRAIN", train_metrics)
    print_metrics("VALIDATION", val_metrics)
    print_metrics("TEST", test_metrics)

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)

    print(f"\n{'Metric':<20} {'Train':<12} {'Val':<12} {'Test':<12} {'Val-Train':<12} {'Test-Train':<12}")
    print("-" * 80)

    metrics_to_compare = [
        ('Accuracy', 'accuracy'),
        ('F1 (macro)', 'f1_macro'),
        ('F1 (FLAT)', 'f1_flat'),
        ('F1 (LONG)', 'f1_long'),
        ('F1 (SHORT)', 'f1_short'),
        ('Reg MAE', 'reg_mae')
    ]

    for name, key in metrics_to_compare:
        train_val = train_metrics[key]
        val_val = val_metrics[key]
        test_val = test_metrics[key]
        val_gap = val_val - train_val
        test_gap = test_val - train_val

        print(f"{name:<20} {train_val:>11.4f} {val_val:>11.4f} {test_val:>11.4f} {val_gap:>11.4f} {test_gap:>11.4f}")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Analyze test vs val
    test_f1 = test_metrics['f1_macro']
    val_f1 = val_metrics['f1_macro']
    train_f1 = train_metrics['f1_macro']

    if test_f1 > val_f1:
        diff = test_f1 - val_f1
        print(f"\n✅ TEST F1 ({test_f1:.4f}) > VAL F1 ({val_f1:.4f}) by {diff:.4f}")
        print("   → Model performs better on recent data (Nov 2024-2025)")
        print("   → Validation period (Nov 2023-Nov 2024) was anomalously difficult")
    else:
        diff = val_f1 - test_f1
        print(f"\n⚠️  VAL F1 ({val_f1:.4f}) > TEST F1 ({test_f1:.4f}) by {diff:.4f}")
        print("   → Model struggles with most recent data")

    # Overfitting check
    train_test_gap = train_f1 - test_f1
    if train_test_gap > 0.15:
        print(f"\n⚠️  Large train-test gap: {train_test_gap:.4f}")
        print("   → Model is overfitting - consider:")
        print("     • Reduce model complexity (n_layers: 5 → 3)")
        print("     • Increase dropout (0.2 → 0.4)")
        print("     • Add more training data")
    elif train_test_gap > 0.05:
        print(f"\n📊 Moderate train-test gap: {train_test_gap:.4f}")
        print("   → Some overfitting, but within acceptable range")
    else:
        print(f"\n✅ Small train-test gap: {train_test_gap:.4f}")
        print("   → Good generalization")

    # Class-specific analysis
    print("\n" + "="*80)
    print("CLASS-SPECIFIC PERFORMANCE (Test Set)")
    print("="*80)

    test_f1_flat = test_metrics['f1_flat']
    test_f1_long = test_metrics['f1_long']
    test_f1_short = test_metrics['f1_short']

    best_class = max([('FLAT', test_f1_flat), ('LONG', test_f1_long), ('SHORT', test_f1_short)], key=lambda x: x[1])
    worst_class = min([('FLAT', test_f1_flat), ('LONG', test_f1_long), ('SHORT', test_f1_short)], key=lambda x: x[1])

    print(f"\nBest:  {best_class[0]} (F1 = {best_class[1]:.4f})")
    print(f"Worst: {worst_class[0]} (F1 = {worst_class[1]:.4f})")

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()

