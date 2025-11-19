#!/usr/bin/env python3
"""
TASK 11: PyTorch Dataset + DataLoader
Implement Dataset class for TCN training.
"""

from __future__ import annotations

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Dict, Optional


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for TCN training on EURUSD sequences.

    Returns:
        X_t: [seq_len, n_features] - Feature sequence
        y_class_t: int - Direction label (0=FLAT, 1=LONG, 2=SHORT)
        y_reg_t: [4] - Regression targets (mfe_l, mae_l, mfe_s, mae_s)
    """

    def __init__(self, data_path: str):
        """
        Initialize dataset from pickle file.

        Args:
            data_path: Path to normalized pickle file (train/val/test)
        """
        print(f"\n📦 Loading dataset from: {data_path}")

        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # Extract data
        self.X = data['X']  # [n_samples, seq_len, n_features]
        self.y_class = data['y_class']  # [n_samples]
        self.y_reg = data['y_reg']  # [n_samples, 4]
        self.timestamps = data['timestamps']  # [n_samples]
        self.feature_names = data['feature_names']

        # Metadata
        self.n_samples = data['n_samples']
        self.seq_len = data['seq_len']
        self.n_features = data['n_features']

        print(f"  ✓ Loaded {self.n_samples:,} samples")
        print(f"  ✓ Shape: [{self.seq_len}, {self.n_features}]")

        # Verify data integrity
        assert len(self.X) == self.n_samples
        assert len(self.y_class) == self.n_samples
        assert len(self.y_reg) == self.n_samples
        assert self.X.shape == (self.n_samples, self.seq_len, self.n_features)
        assert self.y_reg.shape == (self.n_samples, 4)

        # Convert to PyTorch tensors for faster access
        print(f"  Converting to PyTorch tensors...")
        self.X = torch.from_numpy(self.X).float()  # [n_samples, seq_len, n_features]
        self.y_class = torch.from_numpy(self.y_class).long()  # [n_samples]
        self.y_reg = torch.from_numpy(self.y_reg).float()  # [n_samples, 4]

        print(f"  ✓ Tensors ready")
        print(f"    X dtype: {self.X.dtype}, shape: {self.X.shape}")
        print(f"    y_class dtype: {self.y_class.dtype}, shape: {self.y_class.shape}")
        print(f"    y_reg dtype: {self.y_reg.dtype}, shape: {self.y_reg.shape}")

    def __len__(self) -> int:
        """Return total number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            X_t: [seq_len, n_features] - Feature sequence
            y_class_t: int - Direction label
            y_reg_t: [4] - Regression targets
        """
        return self.X[idx], self.y_class[idx], self.y_reg[idx]

    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution statistics."""
        unique, counts = torch.unique(self.y_class, return_counts=True)
        return {int(cls): int(cnt) for cls, cnt in zip(unique, counts)}

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for balanced training (inverse frequency).

        Returns:
            weights: [n_classes] - Weight for each class
        """
        class_dist = self.get_class_distribution()
        n_classes = len(class_dist)

        # Compute weights as inverse frequency
        total = self.n_samples
        weights = torch.zeros(n_classes)

        for cls, count in class_dist.items():
            weights[cls] = total / (n_classes * count)

        return weights

    def summary(self) -> str:
        """Return formatted summary of dataset."""
        class_dist = self.get_class_distribution()
        class_names = {0: 'FLAT', 1: 'LONG', 2: 'SHORT'}

        summary = []
        summary.append(f"\n{'='*60}")
        summary.append(f"Dataset Summary")
        summary.append(f"{'='*60}")
        summary.append(f"Samples: {self.n_samples:,}")
        summary.append(f"Sequence Length: {self.seq_len} bars")
        summary.append(f"Features: {self.n_features}")
        summary.append(f"\nClass Distribution:")
        for cls, count in sorted(class_dist.items()):
            pct = (count / self.n_samples) * 100
            summary.append(f"  {cls} ({class_names[cls]:5s}): {count:6,} ({pct:5.2f}%)")
        summary.append(f"\nFeature Statistics:")
        summary.append(f"  Mean: {self.X.mean():.6f}")
        summary.append(f"  Std:  {self.X.std():.6f}")
        summary.append(f"  Min:  {self.X.min():.6f}")
        summary.append(f"  Max:  {self.X.max():.6f}")
        summary.append(f"{'='*60}\n")

        return '\n'.join(summary)


def create_dataloader(
    dataset: SequenceDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create PyTorch DataLoader from dataset.

    Args:
        dataset: SequenceDataset instance
        batch_size: Batch size (default 64 as per specs)
        shuffle: Whether to shuffle data (True for train, False for val/test)
        num_workers: Number of worker processes (default 4 as per specs)
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def load_all_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Load train/val/test datasets and create dataloaders.

    Args:
        data_dir: Directory containing normalized pickle files
        batch_size: Batch size for all loaders
        num_workers: Number of worker processes

    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    print("="*80)
    print("Loading All Datasets and Creating DataLoaders")
    print("="*80)

    data_path = Path(data_dir)

    # Load datasets
    train_dataset = SequenceDataset(str(data_path / 'train_normalized.pkl'))
    val_dataset = SequenceDataset(str(data_path / 'val_normalized.pkl'))
    test_dataset = SequenceDataset(str(data_path / 'test_normalized.pkl'))

    # Create dataloaders
    print("\n📦 Creating DataLoaders...")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for test
        num_workers=num_workers
    )

    print(f"\n✓ DataLoaders created:")
    print(f"  Train: {len(train_loader):,} batches ({len(train_dataset):,} samples)")
    print(f"  Val:   {len(val_loader):,} batches ({len(val_dataset):,} samples)")
    print(f"  Test:  {len(test_loader):,} batches ({len(test_dataset):,} samples)")

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


def demo_dataloader():
    """Demo function to test dataloader functionality."""
    print("="*80)
    print("TASK 11: PyTorch Dataset + DataLoader Demo")
    print("="*80)

    # Load all dataloaders
    data_dir = '/Users/ibra/code/swingtrader/data'
    loaders = load_all_dataloaders(data_dir, batch_size=64, num_workers=4)

    # Print dataset summaries
    print(loaders['train_dataset'].summary())
    print(loaders['val_dataset'].summary())
    print(loaders['test_dataset'].summary())

    # Test batch loading
    print("="*80)
    print("Testing Batch Loading")
    print("="*80)

    train_loader = loaders['train']

    # Get first batch
    batch_iter = iter(train_loader)
    X_batch, y_class_batch, y_reg_batch = next(batch_iter)

    print(f"\n📦 First Batch:")
    print(f"  X shape: {X_batch.shape} (expected: [batch_size, seq_len, n_features])")
    print(f"  y_class shape: {y_class_batch.shape} (expected: [batch_size])")
    print(f"  y_reg shape: {y_reg_batch.shape} (expected: [batch_size, 4])")

    print(f"\n📊 Batch Statistics:")
    print(f"  X dtype: {X_batch.dtype}")
    print(f"  y_class dtype: {y_class_batch.dtype}")
    print(f"  y_reg dtype: {y_reg_batch.dtype}")

    print(f"\n  X mean: {X_batch.mean():.6f}")
    print(f"  X std:  {X_batch.std():.6f}")
    print(f"  X min:  {X_batch.min():.6f}")
    print(f"  X max:  {X_batch.max():.6f}")

    print(f"\n  y_class unique: {torch.unique(y_class_batch).tolist()}")
    print(f"  y_class counts: {torch.bincount(y_class_batch).tolist()}")

    print(f"\n  y_reg mean: {y_reg_batch.mean(dim=0).tolist()}")
    print(f"  y_reg std:  {y_reg_batch.std(dim=0).tolist()}")

    # Test iteration speed
    print(f"\n⚡ Testing Iteration Speed...")
    import time

    n_batches = 100
    start_time = time.time()

    for i, (X, y_class, y_reg) in enumerate(train_loader):
        if i >= n_batches:
            break

    elapsed = time.time() - start_time
    batches_per_sec = n_batches / elapsed
    samples_per_sec = n_batches * X.shape[0] / elapsed

    print(f"  Processed {n_batches} batches in {elapsed:.2f}s")
    print(f"  Speed: {batches_per_sec:.1f} batches/sec, {samples_per_sec:.1f} samples/sec")

    # Compute class weights
    print(f"\n⚖️  Class Weights (for balanced loss):")
    train_dataset = loaders['train_dataset']
    class_weights = train_dataset.get_class_weights()
    class_names = {0: 'FLAT', 1: 'LONG', 2: 'SHORT'}
    for i, weight in enumerate(class_weights):
        print(f"  {i} ({class_names[i]:5s}): {weight:.4f}")

    print("\n" + "="*80)
    print("✅ TASK 11 COMPLETE")
    print("="*80)
    print("\n📦 Ready for TASK 12 (TCN Model Implementation)!")
    print("\nUsage Example:")
    print("```python")
    print("from pytorch_dataset import load_all_dataloaders")
    print("")
    print("# Load all dataloaders")
    print("loaders = load_all_dataloaders('/path/to/data', batch_size=64)")
    print("")
    print("# Training loop")
    print("for epoch in range(num_epochs):")
    print("    for X, y_class, y_reg in loaders['train']:")
    print("        # X: [batch_size, 168, 55]")
    print("        # y_class: [batch_size]")
    print("        # y_reg: [batch_size, 4]")
    print("        # ... training code ...")
    print("```")


if __name__ == "__main__":
    demo_dataloader()

