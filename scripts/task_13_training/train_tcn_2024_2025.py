#!/usr/bin/env python3
"""
Train TCN model on 2024-2025 data ONLY.
Focus on recent market patterns.
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir / '../task_11_pytorch_dataset'))
sys.path.insert(0, str(script_dir / '../task_12_model'))

from train_tcn import train_model, DualHeadTCN, DualHeadLoss
from pytorch_dataset import SequenceDataset
from torch.utils.data import DataLoader
import torch


def main():
    print("="*80)
    print("TASK 13: Training Loop (2024-2025 DATA ONLY)")
    print("="*80)
    print("\n📋 Training on RECENT data:")
    print("  ✅ Train: Jan 2024 - Feb 2025  (13 months)")
    print("  ✅ Val:   Feb 2025 - Jul 2025  (5 months)")
    print("  ✅ Test:  Jul 2025 - Nov 2025  (4 months)")
    print("\n💡 All data from SAME market regime (no regime shift!)")

    # Device detection
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"\n🖥️  Using device: {device}")

    # Config based on device
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 VRAM: {gpu_memory_gb:.1f} GB")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        if gpu_memory_gb >= 20:
            batch_size = 256  # Smaller dataset
            num_workers = 8
            print("📊 Using HIGH-END GPU config (batch_size=256)")
        elif gpu_memory_gb >= 10:
            batch_size = 128
            num_workers = 6
            print("📊 Using MID-RANGE GPU config (batch_size=128)")
        else:
            batch_size = 64
            num_workers = 4
            print("📊 Using STANDARD GPU config (batch_size=64)")
    else:
        batch_size = 64
        num_workers = 4
        print("📊 Using CPU/MPS config (batch_size=64)")

    # Configuration
    CONFIG = {
        'data_dir': str(Path(__file__).parent.parent / 'data'),
        'checkpoint_dir': str(Path(__file__).parent.parent / 'checkpoints_2024_2025'),
        'log_dir': str(Path(__file__).parent.parent / 'logs_2024_2025'),
        'batch_size': batch_size,
        'num_workers': num_workers,
        'n_features': 55,
        'dropout': 0.2,
        'lambda_reg': 0.05,
        'use_class_weights': False,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 50,
        'early_stopping_patience': 10,
        'grad_clip': 1.0,
        'device': device
    }

    # Create output directories
    Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)
    Path(CONFIG['log_dir']).mkdir(exist_ok=True)

    # Load 2024-2025 datasets
    print(f"\n📦 Loading 2024-2025 datasets...")

    train_dataset = SequenceDataset(f"{CONFIG['data_dir']}/train_normalized_2024_2025.pkl")
    val_dataset = SequenceDataset(f"{CONFIG['data_dir']}/val_normalized_2024_2025.pkl")
    test_dataset = SequenceDataset(f"{CONFIG['data_dir']}/test_normalized_2024_2025.pkl")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device == 'cuda')
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device == 'cuda')
    )

    print(f"  ✓ Train: {len(train_dataset):,} samples")
    print(f"  ✓ Val:   {len(val_dataset):,} samples")
    print(f"  ✓ Test:  {len(test_dataset):,} samples")

    # Create model
    print(f"\n🏗️  Creating TCN model...")
    model = DualHeadTCN(
        n_features=CONFIG['n_features'],
        n_channels=[64, 64, 128, 128, 256],
        kernel_size=3,
        dropout=CONFIG['dropout'],
        n_classes=3,
        n_reg_outputs=4,
        pooling='last'
    )
    model = model.to(device)

    print(f"  ✓ Model created")
    print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss
    criterion = DualHeadLoss(
        lambda_reg=CONFIG['lambda_reg'],
        class_weights=None
    )

    # Train
    print(f"\n🚀 Starting training on 2024-2025 data...")
    print(f"  Expected: Better generalization (same market regime!)")

    metrics_tracker = train_model(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay'],
        grad_clip=CONFIG['grad_clip'],
        early_stopping_patience=CONFIG['early_stopping_patience'],
        checkpoint_dir=CONFIG['checkpoint_dir'],
        log_dir=CONFIG['log_dir'],
        device=device
    )

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE ON 2024-2025 DATA")
    print("="*80)
    print(f"\n📁 Outputs:")
    print(f"  Checkpoints: {CONFIG['checkpoint_dir']}")
    print(f"  Logs: {CONFIG['log_dir']}")
    print(f"\n💡 Next: Evaluate to see if LONG class performs better!")


if __name__ == '__main__':
    main()

