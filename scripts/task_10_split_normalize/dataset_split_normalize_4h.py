#!/usr/bin/env python3
"""
TASK 10: Dataset Split + Normalization
Prepare train/val/test splits with z-score normalization.
"""

from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict


def split_dataset_chronological(
    X: np.ndarray,
    y_class: np.ndarray,
    y_reg: np.ndarray,
    timestamps: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> Dict:
    """
    Split dataset chronologically into train/val/test.

    Args:
        X: Feature sequences [n_samples, seq_len, n_features]
        y_class: Classification labels [n_samples]
        y_reg: Regression targets [n_samples, 4]
        timestamps: Timestamps for each sample [n_samples]
        train_ratio: Training set ratio (default 0.6)
        val_ratio: Validation set ratio (default 0.2)
        test_ratio: Test set ratio (default 0.2)

    Returns:
        Dictionary with train/val/test splits
    """
    print("\n" + "="*80)
    print(f"Chronological Dataset Split")
    print(f"  Train: {train_ratio*100:.0f}%")
    print(f"  Val:   {val_ratio*100:.0f}%")
    print(f"  Test:  {test_ratio*100:.0f}%")
    print("="*80)

    n_samples = len(X)

    # Calculate split indices
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    print(f"\nTotal samples: {n_samples:,}")
    print(f"\nSplit indices:")
    print(f"  Train: 0 to {train_end:,} ({train_end:,} samples)")
    print(f"  Val:   {train_end:,} to {val_end:,} ({val_end - train_end:,} samples)")
    print(f"  Test:  {val_end:,} to {n_samples:,} ({n_samples - val_end:,} samples)")

    # Split the data
    X_train = X[:train_end]
    X_val = X[train_end:val_end]
    X_test = X[val_end:]

    y_class_train = y_class[:train_end]
    y_class_val = y_class[train_end:val_end]
    y_class_test = y_class[val_end:]

    y_reg_train = y_reg[:train_end]
    y_reg_val = y_reg[train_end:val_end]
    y_reg_test = y_reg[val_end:]

    timestamps_train = timestamps[:train_end]
    timestamps_val = timestamps[train_end:val_end]
    timestamps_test = timestamps[val_end:]

    # Display date ranges
    print(f"\n📅 Date Ranges:")
    print(f"  Train: {pd.to_datetime(timestamps_train[0])} to {pd.to_datetime(timestamps_train[-1])}")
    print(f"  Val:   {pd.to_datetime(timestamps_val[0])} to {pd.to_datetime(timestamps_val[-1])}")
    print(f"  Test:  {pd.to_datetime(timestamps_test[0])} to {pd.to_datetime(timestamps_test[-1])}")

    # Check class distribution in each split
    print(f"\n📊 Class Distribution by Split:")

    for split_name, y_split in [('Train', y_class_train), ('Val', y_class_val), ('Test', y_class_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        print(f"\n  {split_name}:")
        class_names = {0: 'FLAT', 1: 'LONG', 2: 'SHORT'}
        for class_id, count in zip(unique, counts):
            pct = (count / len(y_split)) * 100
            print(f"    {class_id} ({class_names[class_id]:5s}): {count:6,} ({pct:5.2f}%)")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_class_train': y_class_train,
        'y_class_val': y_class_val,
        'y_class_test': y_class_test,
        'y_reg_train': y_reg_train,
        'y_reg_val': y_reg_val,
        'y_reg_test': y_reg_test,
        'timestamps_train': timestamps_train,
        'timestamps_val': timestamps_val,
        'timestamps_test': timestamps_test
    }


def compute_normalization_params(X_train: np.ndarray) -> Dict:
    """
    Compute z-score normalization parameters from training data.

    Args:
        X_train: Training features [n_samples, seq_len, n_features]

    Returns:
        Dictionary with mean and std for each feature
    """
    print("\n" + "="*80)
    print("Computing Normalization Parameters (Z-Score)")
    print("="*80)

    print(f"\nTraining data shape: {X_train.shape}")

    # Reshape to [n_samples * seq_len, n_features] to compute statistics
    X_train_flat = X_train.reshape(-1, X_train.shape[2])

    print(f"Flattened shape: {X_train_flat.shape}")

    # Compute mean and std for each feature
    mean = np.mean(X_train_flat, axis=0)
    std = np.std(X_train_flat, axis=0)

    # Prevent division by zero (for constant features)
    std = np.where(std < 1e-8, 1.0, std)

    # Check for constant features
    constant_features = np.where(std == 1.0)[0]
    if len(constant_features) > 0:
        print(f"\n⚠️  Warning: {len(constant_features)} constant/near-constant features detected")
        print(f"   These will not be normalized (std set to 1.0)")

    print(f"\n✓ Computed normalization parameters for {len(mean)} features")

    # Display statistics for a few features
    print(f"\n📊 Sample Normalization Parameters (first 10 features):")
    for i in range(min(10, len(mean))):
        print(f"  Feature {i:2d}: mean={mean[i]:10.4f}, std={std[i]:10.4f}")

    return {
        'mean': mean,
        'std': std
    }


def apply_normalization(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Apply z-score normalization to features.

    Args:
        X: Features [n_samples, seq_len, n_features]
        mean: Mean for each feature [n_features]
        std: Std for each feature [n_features]

    Returns:
        Normalized features
    """
    # Z-score normalization: (X - mean) / std
    # Broadcasting handles the shape automatically
    X_normalized = (X - mean) / std

    return X_normalized


def normalize_splits(
    splits: Dict,
    norm_params: Dict
) -> Dict:
    """
    Apply normalization to all splits.

    Args:
        splits: Dictionary with train/val/test splits
        norm_params: Normalization parameters (mean, std)

    Returns:
        Dictionary with normalized splits
    """
    print("\n" + "="*80)
    print("Applying Normalization to All Splits")
    print("="*80)

    mean = norm_params['mean']
    std = norm_params['std']

    print(f"\nNormalizing training set...")
    X_train_norm = apply_normalization(splits['X_train'], mean, std)
    print(f"  ✓ Train: {X_train_norm.shape}")

    print(f"\nNormalizing validation set...")
    X_val_norm = apply_normalization(splits['X_val'], mean, std)
    print(f"  ✓ Val: {X_val_norm.shape}")

    print(f"\nNormalizing test set...")
    X_test_norm = apply_normalization(splits['X_test'], mean, std)
    print(f"  ✓ Test: {X_test_norm.shape}")

    # Verify normalization (train should have mean~0, std~1)
    print(f"\n📊 Normalization Verification (Train Set):")
    X_train_flat = X_train_norm.reshape(-1, X_train_norm.shape[2])
    train_mean = np.mean(X_train_flat, axis=0)
    train_std = np.std(X_train_flat, axis=0)

    print(f"  Mean (should be ~0):    min={train_mean.min():.6f}, max={train_mean.max():.6f}")
    print(f"  Std (should be ~1):     min={train_std.min():.6f}, max={train_std.max():.6f}")

    # Check for NaN or Inf
    if np.isnan(X_train_norm).any():
        print(f"\n  ⚠️  Warning: NaN values detected in normalized train data")
    if np.isnan(X_val_norm).any():
        print(f"\n  ⚠️  Warning: NaN values detected in normalized val data")
    if np.isnan(X_test_norm).any():
        print(f"\n  ⚠️  Warning: NaN values detected in normalized test data")

    if not (np.isnan(X_train_norm).any() or np.isnan(X_val_norm).any() or np.isnan(X_test_norm).any()):
        print(f"\n  ✓ No NaN values detected")

    return {
        'X_train': X_train_norm,
        'X_val': X_val_norm,
        'X_test': X_test_norm,
        'y_class_train': splits['y_class_train'],
        'y_class_val': splits['y_class_val'],
        'y_class_test': splits['y_class_test'],
        'y_reg_train': splits['y_reg_train'],
        'y_reg_val': splits['y_reg_val'],
        'y_reg_test': splits['y_reg_test'],
        'timestamps_train': splits['timestamps_train'],
        'timestamps_val': splits['timestamps_val'],
        'timestamps_test': splits['timestamps_test']
    }


def save_normalized_dataset(
    splits: Dict,
    norm_params: Dict,
    metadata: Dict,
    output_dir: str
) -> None:
    """
    Save normalized dataset and normalization parameters.

    Args:
        splits: Normalized train/val/test splits
        norm_params: Normalization parameters
        metadata: Original dataset metadata
        output_dir: Output directory
    """
    print("\n" + "="*80)
    print("Saving Normalized Dataset")
    print("="*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save train split
    train_file = output_path / 'train_normalized.pkl'
    print(f"\n📁 Saving train split: {train_file}")
    with open(train_file, 'wb') as f:
        pickle.dump({
            'X': splits['X_train'],
            'y_class': splits['y_class_train'],
            'y_reg': splits['y_reg_train'],
            'timestamps': splits['timestamps_train'],
            'feature_names': metadata['feature_names'],
            'n_samples': len(splits['X_train']),
            'seq_len': metadata['seq_len'],
            'n_features': metadata['n_features']
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    train_size_mb = train_file.stat().st_size / (1024 ** 2)
    print(f"  ✓ Train saved ({train_size_mb:.2f} MB)")

    # Save val split
    val_file = output_path / 'val_normalized.pkl'
    print(f"\n📁 Saving val split: {val_file}")
    with open(val_file, 'wb') as f:
        pickle.dump({
            'X': splits['X_val'],
            'y_class': splits['y_class_val'],
            'y_reg': splits['y_reg_val'],
            'timestamps': splits['timestamps_val'],
            'feature_names': metadata['feature_names'],
            'n_samples': len(splits['X_val']),
            'seq_len': metadata['seq_len'],
            'n_features': metadata['n_features']
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    val_size_mb = val_file.stat().st_size / (1024 ** 2)
    print(f"  ✓ Val saved ({val_size_mb:.2f} MB)")

    # Save test split
    test_file = output_path / 'test_normalized.pkl'
    print(f"\n📁 Saving test split: {test_file}")
    with open(test_file, 'wb') as f:
        pickle.dump({
            'X': splits['X_test'],
            'y_class': splits['y_class_test'],
            'y_reg': splits['y_reg_test'],
            'timestamps': splits['timestamps_test'],
            'feature_names': metadata['feature_names'],
            'n_samples': len(splits['X_test']),
            'seq_len': metadata['seq_len'],
            'n_features': metadata['n_features']
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    test_size_mb = test_file.stat().st_size / (1024 ** 2)
    print(f"  ✓ Test saved ({test_size_mb:.2f} MB)")

    # Save normalization parameters
    norm_file = output_path / 'normalization_params.pkl'
    print(f"\n📁 Saving normalization parameters: {norm_file}")
    with open(norm_file, 'wb') as f:
        pickle.dump({
            'mean': norm_params['mean'],
            'std': norm_params['std'],
            'feature_names': metadata['feature_names']
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    norm_size_kb = norm_file.stat().st_size / 1024
    print(f"  ✓ Normalization params saved ({norm_size_kb:.2f} KB)")

    # Save metadata
    metadata_file = output_path / 'dataset_metadata.txt'
    print(f"\n📁 Saving metadata: {metadata_file}")

    with open(metadata_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Normalized Dataset Metadata\n")
        f.write("="*80 + "\n\n")
        f.write(f"Created: {pd.Timestamp.now()}\n\n")

        f.write("Files:\n")
        f.write(f"  - train_normalized.pkl ({train_size_mb:.2f} MB)\n")
        f.write(f"  - val_normalized.pkl ({val_size_mb:.2f} MB)\n")
        f.write(f"  - test_normalized.pkl ({test_size_mb:.2f} MB)\n")
        f.write(f"  - normalization_params.pkl ({norm_size_kb:.2f} KB)\n\n")

        f.write("Dataset Configuration:\n")
        f.write(f"  - Sequence length: {metadata['seq_len']} bars\n")
        f.write(f"  - Number of features: {metadata['n_features']}\n")
        f.write(f"  - Total samples: {metadata['n_samples']:,}\n\n")

        f.write("Split Sizes:\n")
        f.write(f"  - Train: {len(splits['X_train']):,} samples\n")
        f.write(f"  - Val:   {len(splits['X_val']):,} samples\n")
        f.write(f"  - Test:  {len(splits['X_test']):,} samples\n\n")

        f.write("Date Ranges:\n")
        f.write(f"  - Train: {pd.to_datetime(splits['timestamps_train'][0])} to {pd.to_datetime(splits['timestamps_train'][-1])}\n")
        f.write(f"  - Val:   {pd.to_datetime(splits['timestamps_val'][0])} to {pd.to_datetime(splits['timestamps_val'][-1])}\n")
        f.write(f"  - Test:  {pd.to_datetime(splits['timestamps_test'][0])} to {pd.to_datetime(splits['timestamps_test'][-1])}\n\n")

        f.write("Normalization:\n")
        f.write(f"  - Method: Z-score (mean=0, std=1)\n")
        f.write(f"  - Fitted on: Training set only\n")
        f.write(f"  - Applied to: All splits\n\n")

        f.write(f"Features ({len(metadata['feature_names'])}):\n")
        for i, fname in enumerate(metadata['feature_names'], 1):
            f.write(f"  {i:3d}. {fname}\n")

    print(f"  ✓ Metadata saved")

    total_size_mb = train_size_mb + val_size_mb + test_size_mb
    print(f"\n✓ Total dataset size: {total_size_mb:.2f} MB")


def main():
    """Main pipeline execution"""
    
    print("="*80)
    print("TASK 10: Dataset Split + Normalization")
    print("="*80)
    
    # Get project root directory (cross-platform)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Load sequence dataset
    input_file = project_root / 'data' / 'sequences_eurusd_4h_42.pkl'
    print(f"\nLoading sequence dataset: {input_file}")

    with open(input_file, 'rb') as f:
        dataset = pickle.load(f)

    print(f"✓ Loaded dataset")
    print(f"  - Samples: {dataset['n_samples']:,}")
    print(f"  - Shape: {dataset['X'].shape}")
    print(f"  - Features: {dataset['n_features']}")

    # Step 1: Split dataset chronologically
    splits = split_dataset_chronological(
        X=dataset['X'],
        y_class=dataset['y_class'],
        y_reg=dataset['y_reg'],
        timestamps=dataset['timestamps'],
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )

    # Step 2: Compute normalization parameters on training set
    norm_params = compute_normalization_params(splits['X_train'])

    # Step 3: Apply normalization to all splits
    normalized_splits = normalize_splits(splits, norm_params)

    # Step 4: Save normalized dataset
    output_dir = project_root / 'data'
    metadata = {
        'feature_names': dataset['feature_names'],
        'seq_len': dataset['seq_len'],
        'n_features': dataset['n_features'],
        'n_samples': dataset['n_samples']
    }

    save_normalized_dataset(normalized_splits, norm_params, metadata, output_dir)

    print("\n" + "="*80)
    print("✅ TASK 10 COMPLETE")
    print("="*80)

    print("\n📦 Output Files:")
    print(f"  - train_normalized.pkl ({len(normalized_splits['X_train']):,} samples)")
    print(f"  - val_normalized.pkl ({len(normalized_splits['X_val']):,} samples)")
    print(f"  - test_normalized.pkl ({len(normalized_splits['X_test']):,} samples)")
    print(f"  - normalization_params.pkl")
    print(f"  - dataset_metadata.txt")

    print("\n📊 Dataset Summary:")
    print(f"  - Total sequences: {dataset['n_samples']:,}")
    print(f"  - Train: {len(normalized_splits['X_train']):,} (60%)")
    print(f"  - Val:   {len(normalized_splits['X_val']):,} (20%)")
    print(f"  - Test:  {len(normalized_splits['X_test']):,} (20%)")

    print("\n✅ Dataset ready for PyTorch DataLoader (TASK 11)!")


if __name__ == "__main__":
    main()

