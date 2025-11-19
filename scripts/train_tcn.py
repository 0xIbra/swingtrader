#!/usr/bin/env python3
"""
TASK 13: Training Loop
Implement training + validation engine with metrics tracking and checkpointing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import time
from datetime import datetime

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tcn_model import create_tcn_model, DualHeadTCN, DualHeadLoss
from pytorch_dataset import load_all_dataloaders


class MetricsTracker:
    """Track and log training metrics."""
    
    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train': [],
            'val': []
        }
        
        self.best_val_f1 = 0.0
        self.best_val_mae = float('inf')
        self.best_epoch = 0
    
    def update(self, epoch: int, metrics: Dict, split: str = 'train'):
        """
        Update metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
            split: 'train' or 'val'
        """
        metrics['epoch'] = epoch
        self.history[split].append(metrics)
    
    def is_best_model(self, val_f1: float, val_mae: float) -> bool:
        """
        Check if current model is the best so far.
        
        Args:
            val_f1: Validation F1 score (macro)
            val_mae: Validation MAE (regression)
        
        Returns:
            True if this is the best model
        """
        # Primary metric: F1 score (higher is better)
        # Secondary metric: MAE (lower is better)
        is_best = False
        
        if val_f1 > self.best_val_f1:
            is_best = True
            self.best_val_f1 = val_f1
            self.best_val_mae = val_mae
        elif val_f1 == self.best_val_f1 and val_mae < self.best_val_mae:
            is_best = True
            self.best_val_mae = val_mae
        
        return is_best
    
    def save_logs(self):
        """Save training history to JSON file."""
        log_file = self.log_dir / 'training_history.json'
        
        with open(log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"  ✓ Logs saved to {log_file}")
    
    def save_csv(self):
        """Save training history to CSV files."""
        # Train metrics
        if self.history['train']:
            train_df = pd.DataFrame(self.history['train'])
            train_csv = self.log_dir / 'train_metrics.csv'
            train_df.to_csv(train_csv, index=False)
            print(f"  ✓ Train metrics saved to {train_csv}")
        
        # Val metrics
        if self.history['val']:
            val_df = pd.DataFrame(self.history['val'])
            val_csv = self.log_dir / 'val_metrics.csv'
            val_df.to_csv(val_csv, index=False)
            print(f"  ✓ Val metrics saved to {val_csv}")
    
    def print_epoch_summary(self, epoch: int, num_epochs: int, train_metrics: Dict, val_metrics: Dict):
        """Print formatted epoch summary."""
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        print(f"\n📊 Training Metrics:")
        print(f"  Total Loss:  {train_metrics['total_loss']:.4f}")
        print(f"  CE Loss:     {train_metrics['ce_loss']:.4f}")
        print(f"  Reg Loss:    {train_metrics['reg_loss']:.4f}")
        print(f"  Accuracy:    {train_metrics['accuracy']:.4f}")
        print(f"  F1 (macro):  {train_metrics['f1_macro']:.4f}")
        print(f"  F1 (LONG):   {train_metrics['f1_long']:.4f}")
        print(f"  F1 (SHORT):  {train_metrics['f1_short']:.4f}")
        print(f"  Reg MAE:     {train_metrics['reg_mae']:.4f}")
        
        print(f"\n📊 Validation Metrics:")
        print(f"  Total Loss:  {val_metrics['total_loss']:.4f}")
        print(f"  CE Loss:     {val_metrics['ce_loss']:.4f}")
        print(f"  Reg Loss:    {val_metrics['reg_loss']:.4f}")
        print(f"  Accuracy:    {val_metrics['accuracy']:.4f}")
        print(f"  F1 (macro):  {val_metrics['f1_macro']:.4f}")
        print(f"  F1 (LONG):   {val_metrics['f1_long']:.4f}")
        print(f"  F1 (SHORT):  {val_metrics['f1_short']:.4f}")
        print(f"  Reg MAE:     {val_metrics['reg_mae']:.4f}")
        
        print(f"\n⏱️  Time: {train_metrics['epoch_time']:.2f}s")


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like F1, 'min' for metrics like loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check for improvement
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels [n_samples]
        y_pred: Predicted labels [n_samples]
    
    Returns:
        Dictionary of metrics
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics (0=FLAT, 1=LONG, 2=SHORT)
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


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values [n_samples, 4]
        y_pred: Predicted values [n_samples, 4]
    
    Returns:
        Dictionary of metrics
    """
    # Overall metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Per-output metrics (mfe_l, mae_l, mfe_s, mae_s)
    mae_per_output = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(4)]
    
    metrics = {
        'reg_mae': mae,
        'reg_mse': mse,
        'reg_rmse': rmse,
        'mae_mfe_l': mae_per_output[0],
        'mae_mae_l': mae_per_output[1],
        'mae_mfe_s': mae_per_output[2],
        'mae_mae_s': mae_per_output[3]
    }
    
    return metrics


def train_epoch(
    model: DualHeadTCN,
    train_loader: DataLoader,
    criterion: DualHeadLoss,
    optimizer: optim.Optimizer,
    device: str,
    grad_clip: Optional[float] = None
) -> Dict:
    """
    Train for one epoch.
    
    Args:
        model: TCN model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        grad_clip: Gradient clipping value (optional)
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    ce_loss_sum = 0.0
    reg_loss_sum = 0.0
    
    all_y_true = []
    all_y_pred = []
    all_y_reg_true = []
    all_y_reg_pred = []
    
    for X, y_class, y_reg in train_loader:
        # Move to device
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
        
        # Gradient clipping (optional)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        ce_loss_sum += ce_loss.item()
        reg_loss_sum += reg_loss.item()
        
        # Store predictions for metrics
        y_pred = direction_logits.argmax(dim=1)
        all_y_true.append(y_class.cpu().numpy())
        all_y_pred.append(y_pred.cpu().numpy())
        all_y_reg_true.append(y_reg.cpu().numpy())
        all_y_reg_pred.append(excursion_preds.detach().cpu().numpy())
    
    # Compute epoch metrics
    num_batches = len(train_loader)
    
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    all_y_reg_true = np.concatenate(all_y_reg_true)
    all_y_reg_pred = np.concatenate(all_y_reg_pred)
    
    # Classification metrics
    class_metrics = compute_classification_metrics(all_y_true, all_y_pred)
    
    # Regression metrics
    reg_metrics = compute_regression_metrics(all_y_reg_true, all_y_reg_pred)
    
    # Combine metrics
    metrics = {
        'total_loss': total_loss / num_batches,
        'ce_loss': ce_loss_sum / num_batches,
        'reg_loss': reg_loss_sum / num_batches,
        **class_metrics,
        **reg_metrics
    }
    
    return metrics


@torch.no_grad()
def validate_epoch(
    model: DualHeadTCN,
    val_loader: DataLoader,
    criterion: DualHeadLoss,
    device: str
) -> Dict:
    """
    Validate for one epoch.
    
    Args:
        model: TCN model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    ce_loss_sum = 0.0
    reg_loss_sum = 0.0
    
    all_y_true = []
    all_y_pred = []
    all_y_reg_true = []
    all_y_reg_pred = []
    
    for X, y_class, y_reg in val_loader:
        # Move to device
        X = X.to(device)
        y_class = y_class.to(device)
        y_reg = y_reg.to(device)
        
        # Forward pass
        direction_logits, excursion_preds = model(X)
        
        # Compute loss
        loss, ce_loss, reg_loss = criterion(direction_logits, excursion_preds, y_class, y_reg)
        
        # Accumulate losses
        total_loss += loss.item()
        ce_loss_sum += ce_loss.item()
        reg_loss_sum += reg_loss.item()
        
        # Store predictions for metrics
        y_pred = direction_logits.argmax(dim=1)
        all_y_true.append(y_class.cpu().numpy())
        all_y_pred.append(y_pred.cpu().numpy())
        all_y_reg_true.append(y_reg.cpu().numpy())
        all_y_reg_pred.append(excursion_preds.cpu().numpy())
    
    # Compute epoch metrics
    num_batches = len(val_loader)
    
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    all_y_reg_true = np.concatenate(all_y_reg_true)
    all_y_reg_pred = np.concatenate(all_y_reg_pred)
    
    # Classification metrics
    class_metrics = compute_classification_metrics(all_y_true, all_y_pred)
    
    # Regression metrics
    reg_metrics = compute_regression_metrics(all_y_reg_true, all_y_reg_pred)
    
    # Combine metrics
    metrics = {
        'total_loss': total_loss / num_batches,
        'ce_loss': ce_loss_sum / num_batches,
        'reg_loss': reg_loss_sum / num_batches,
        **class_metrics,
        **reg_metrics
    }
    
    return metrics


def save_checkpoint(
    model: DualHeadTCN,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: Dict,
    checkpoint_dir: str,
    filename: str = 'checkpoint.pt'
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    checkpoint_path = Path(checkpoint_dir) / filename
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"  ✓ Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    model: DualHeadTCN,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    checkpoint_path: str,
    device: str
) -> Tuple[int, Dict]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into (optional)
        checkpoint_path: Path to checkpoint file
        device: Device to load to
    
    Returns:
        epoch: Epoch number from checkpoint
        metrics: Metrics from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    
    print(f"✓ Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {epoch}")
    print(f"  Val F1: {metrics.get('f1_macro', 'N/A')}")
    print(f"  Val MAE: {metrics.get('reg_mae', 'N/A')}")
    
    return epoch, metrics


def train_model(
    model: DualHeadTCN,
    criterion: DualHeadLoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    grad_clip: Optional[float] = 1.0,
    early_stopping_patience: int = 10,
    checkpoint_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    device: str = 'cpu',
    resume_from: Optional[str] = None
):
    """
    Main training loop.
    
    Args:
        model: TCN model
        criterion: Loss function
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for AdamW (default 1e-4 as per specs)
        weight_decay: L2 regularization
        grad_clip: Gradient clipping value
        early_stopping_patience: Patience for early stopping
        checkpoint_dir: Directory to save checkpoints (None = auto-detect)
        log_dir: Directory to save logs (None = auto-detect)
        device: Device to train on
        resume_from: Path to checkpoint to resume from
    """
    # Set default paths relative to project root if not provided
    if checkpoint_dir is None or log_dir is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        if checkpoint_dir is None:
            checkpoint_dir = str(project_root / 'checkpoints')
        if log_dir is None:
            log_dir = str(project_root / 'logs')
    
    print("\n" + "="*80)
    print("Training Configuration")
    print("="*80)
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Gradient clipping: {grad_clip}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("="*80 + "\n")
    
    # Optimizer (AdamW as per specs)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (cosine annealing as per specs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Metrics tracker
    metrics_tracker = MetricsTracker(log_dir)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=1e-4,
        mode='max'  # Maximize F1 score
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if resume_from is not None:
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, resume_from, device)
        start_epoch += 1
    
    # Training loop
    print("🚀 Starting training...\n")
    
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Add timing and LR to metrics
        epoch_time = time.time() - epoch_start_time
        train_metrics['epoch_time'] = epoch_time
        train_metrics['learning_rate'] = current_lr
        val_metrics['epoch_time'] = epoch_time
        val_metrics['learning_rate'] = current_lr
        
        # Update metrics tracker
        metrics_tracker.update(epoch, train_metrics, 'train')
        metrics_tracker.update(epoch, val_metrics, 'val')
        
        # Print summary
        metrics_tracker.print_epoch_summary(epoch, num_epochs, train_metrics, val_metrics)
        
        # Save best model
        if metrics_tracker.is_best_model(val_metrics['f1_macro'], val_metrics['reg_mae']):
            print(f"\n🌟 New best model! (F1: {val_metrics['f1_macro']:.4f}, MAE: {val_metrics['reg_mae']:.4f})")
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                checkpoint_dir, 'best_model.pt'
            )
            metrics_tracker.best_epoch = epoch
        
        # Save regular checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'
            )
        
        # Save latest checkpoint (for resuming)
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics,
            checkpoint_dir, 'latest_checkpoint.pt'
        )
        
        # Early stopping check
        if early_stopping(val_metrics['f1_macro']):
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            print(f"   No improvement for {early_stopping_patience} epochs")
            break
    
    # Save training logs
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\n💾 Saving training logs...")
    metrics_tracker.save_logs()
    metrics_tracker.save_csv()
    
    print(f"\n🏆 Best Model:")
    print(f"  Epoch: {metrics_tracker.best_epoch}")
    print(f"  Val F1 (macro): {metrics_tracker.best_val_f1:.4f}")
    print(f"  Val MAE: {metrics_tracker.best_val_mae:.4f}")
    
    print(f"\n📁 Outputs:")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Logs: {log_dir}")
    
    return metrics_tracker


def main():
    """Main training pipeline."""
    print("="*80)
    print("TASK 13: Training Loop")
    print("="*80)
    
    # Get project root directory (cross-platform)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Configuration
    CONFIG = {
        'data_dir': str(project_root / 'data'),
        'checkpoint_dir': str(project_root / 'checkpoints'),
        'log_dir': str(project_root / 'logs'),
        'batch_size': 64,
        'num_workers': 4,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'grad_clip': 1.0,
        'early_stopping_patience': 10,
        'n_features': 55,
        'n_layers': 5,
        'dropout': 0.2,
        'lambda_reg': 1.0,
        'use_class_weights': True
    }
    
    # Device selection
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"\n🖥️  Using device: {device}")
    
    # Load data
    print(f"\n📦 Loading datasets...")
    loaders = load_all_dataloaders(
        CONFIG['data_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    train_dataset = loaders['train_dataset']
    
    # Get class weights
    class_weights = None
    if CONFIG['use_class_weights']:
        class_weights = train_dataset.get_class_weights()
        print(f"\n⚖️  Class weights: {class_weights}")
    
    # Create model
    print(f"\n🏗️  Creating TCN model...")
    model, criterion = create_tcn_model(
        n_features=CONFIG['n_features'],
        n_layers=CONFIG['n_layers'],
        dropout=CONFIG['dropout'],
        lambda_reg=CONFIG['lambda_reg'],
        class_weights=class_weights,
        device=device
    )
    
    # Train model
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
        device=device,
        resume_from=None  # Set to checkpoint path to resume
    )
    
    print("\n" + "="*80)
    print("✅ TASK 13 COMPLETE")
    print("="*80)
    print("\nNext: TASK 14 (Walk-Forward Evaluation) or test the trained model!")


if __name__ == "__main__":
    main()

