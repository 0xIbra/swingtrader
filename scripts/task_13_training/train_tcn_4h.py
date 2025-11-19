#!/usr/bin/env python3
"""
TASK 13: Training Loop (4H Timeframe)
Train TCN model on 4H EURUSD data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / '../task_11_pytorch_dataset'))
sys.path.insert(0, str(script_dir / '../task_12_model'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import logging
from datetime import datetime

# Import our modules
from tcn_model import create_tcn_model, DualHeadTCN, DualHeadLoss
from pytorch_dataset import SequenceDataset, create_dataloader


def setup_logging(log_file: str):
    """Setup logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_epoch(
    model: DualHeadTCN,
    criterion: DualHeadLoss,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: str,
    epoch: int
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_ce_loss = 0
    total_reg_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for X, y_class, y_reg in pbar:
        X = X.to(device)
        y_class = y_class.to(device)
        y_reg = y_reg.to(device)

        # Forward pass
        direction_logits, excursion_preds = model(X)

        # Compute loss
        loss, ce_loss, reg_loss = criterion(direction_logits, excursion_preds, y_class, y_reg)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_reg_loss += reg_loss.item()

        # Get predictions
        preds = direction_logits.argmax(dim=1).cpu().numpy()
        labels = y_class.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'reg': f'{reg_loss.item():.4f}'
        })

    # Compute metrics
    avg_loss = total_loss / len(train_loader)
    avg_ce_loss = total_ce_loss / len(train_loader)
    avg_reg_loss = total_reg_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return {
        'loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'reg_loss': avg_reg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels
    }


def validate_epoch(
    model: DualHeadTCN,
    criterion: DualHeadLoss,
    val_loader: DataLoader,
    device: str,
    epoch: int
) -> dict:
    """Validate for one epoch."""
    model.eval()

    total_loss = 0
    total_ce_loss = 0
    total_reg_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

        for X, y_class, y_reg in pbar:
            X = X.to(device)
            y_class = y_class.to(device)
            y_reg = y_reg.to(device)

            # Forward pass
            direction_logits, excursion_preds = model(X)

            # Compute loss
            loss, ce_loss, reg_loss = criterion(direction_logits, excursion_preds, y_class, y_reg)

            # Accumulate metrics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_reg_loss += reg_loss.item()

            # Get predictions
            preds = direction_logits.argmax(dim=1).cpu().numpy()
            labels = y_class.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })

    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    avg_ce_loss = total_ce_loss / len(val_loader)
    avg_reg_loss = total_reg_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return {
        'loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'reg_loss': avg_reg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels
    }


def main():
    """Main training loop."""
    print("="*80)
    print("TASK 13: TCN Training Loop (4H Timeframe)")
    print("="*80)

    # Setup
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # task_13_training -> scripts -> project_root
    data_dir = project_root / 'data'

    # Create checkpoint and log directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = project_root / f'checkpoints_4h_{timestamp}'
    checkpoint_dir.mkdir(exist_ok=True)

    log_file = project_root / f'logs_4h_training_{timestamp}.log'
    logger = setup_logging(str(log_file))

    logger.info("="*80)
    logger.info("TCN Training on 4H EURUSD Data")
    logger.info("="*80)

    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load datasets
    logger.info("\nLoading datasets...")
    train_dataset = SequenceDataset(str(data_dir / 'train_normalized.pkl'))
    val_dataset = SequenceDataset(str(data_dir / 'val_normalized.pkl'))

    # Create dataloaders
    batch_size = 128
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Get class weights
    class_weights = train_dataset.get_class_weights()
    logger.info(f"Class weights: {class_weights}")

    # Create model
    logger.info("\nCreating model...")
    model, criterion = create_tcn_model(
        n_features=55,
        n_layers=5,
        dropout=0.2,
        pooling='last',
        lambda_reg=0.05,
        class_weights=class_weights,
        device=device
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    logger.info("\nOptimizer: AdamW(lr=1e-4, weight_decay=0.01)")
    logger.info("Scheduler: CosineAnnealingLR(T_max=50)")

    # Training loop
    num_epochs = 50
    best_val_f1 = 0
    patience = 10
    patience_counter = 0

    logger.info(f"\nStarting training for {num_epochs} epochs...")
    logger.info(f"Early stopping patience: {patience}")

    for epoch in range(1, num_epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info(f"{'='*80}")

        # Train
        train_metrics = train_epoch(model, criterion, optimizer, train_loader, device, epoch)

        # Validate
        val_metrics = validate_epoch(model, criterion, val_loader, device, epoch)

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log metrics
        logger.info(f"\nEpoch {epoch} Summary:")
        logger.info(f"  LR: {current_lr:.2e}")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, CE: {train_metrics['ce_loss']:.4f}, "
                   f"Reg: {train_metrics['reg_loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, CE: {val_metrics['ce_loss']:.4f}, "
                   f"Reg: {val_metrics['reg_loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0

            checkpoint_path = checkpoint_dir / f'best_model_4h.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': val_metrics['f1'],
                'val_loss': val_metrics['loss']
            }, checkpoint_path)

            logger.info(f"  ✓ New best model saved! Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{patience}")

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"\nEarly stopping triggered after {epoch} epochs")
            break

        # Classification report every 5 epochs
        if epoch % 5 == 0:
            logger.info(f"\nValidation Classification Report (Epoch {epoch}):")
            logger.info("\n" + classification_report(
                val_metrics['labels'],
                val_metrics['predictions'],
                target_names=['FLAT', 'LONG', 'SHORT'],
                digits=4
            ))

    logger.info("\n" + "="*80)
    logger.info("✅ TRAINING COMPLETE (4H)")
    logger.info("="*80)
    logger.info(f"Best Val F1: {best_val_f1:.4f}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Log file: {log_file}")

    print(f"\n✅ Training complete! Best Val F1: {best_val_f1:.4f}")
    print(f"📁 Checkpoints saved to: {checkpoint_dir}")
    print(f"📋 Training log: {log_file}")


if __name__ == "__main__":
    main()

