#!/usr/bin/env python3
"""
TASK 12: TCN Model Implementation
Build Temporal Convolutional Network with dual heads for direction classification and MFE/MAE regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalBlock(nn.Module):
    """
    Temporal Convolutional Block with residual connection.

    Architecture:
        Input → Conv1D → LayerNorm → ReLU → Dropout →
                Conv1D → LayerNorm → ReLU → Dropout →
                + Residual → Output
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        """
        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor for temporal convolution
            dropout: Dropout rate
        """
        super().__init__()

        # Calculate padding to maintain sequence length
        # padding = (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation

        # First convolution
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation
        )
        self.ln1 = nn.LayerNorm(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation
        )
        self.ln2 = nn.LayerNorm(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection (1x1 conv if input/output channels differ)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # Final activation
        self.relu = nn.ReLU()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, n_inputs, seq_len]

        Returns:
            Output tensor [batch_size, n_outputs, seq_len]
        """
        # First convolution block
        out = self.conv1(x)

        # Remove extra padding to maintain sequence length
        if self.padding > 0:
            out = out[:, :, :-self.padding]

        # Transpose for LayerNorm: [batch, channels, seq] -> [batch, seq, channels]
        out = out.transpose(1, 2)
        out = self.ln1(out)
        out = out.transpose(1, 2)  # Back to [batch, channels, seq]

        out = self.relu1(out)
        out = self.dropout1(out)

        # Second convolution block
        out = self.conv2(out)

        # Remove extra padding
        if self.padding > 0:
            out = out[:, :, :-self.padding]

        # Transpose for LayerNorm
        out = out.transpose(1, 2)
        out = self.ln2(out)
        out = out.transpose(1, 2)

        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)

        # Add residual and apply activation
        out = self.relu(out + res)

        return out


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network Encoder.

    Stack of TemporalBlocks with increasing dilation rates.
    """

    def __init__(
        self,
        n_inputs: int,
        n_channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        """
        Args:
            n_inputs: Number of input features
            n_channels: List of output channels for each layer (determines depth)
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        num_levels = len(n_channels)

        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8, 16, ...
            in_channels = n_inputs if i == 0 else n_channels[i - 1]
            out_channels = n_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )

        self.network = nn.Sequential(*layers)
        self.output_channels = n_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, n_features]

        Returns:
            Output tensor [batch_size, seq_len, output_channels]
        """
        # TCN expects [batch, channels, seq_len] format
        x = x.transpose(1, 2)  # [batch, n_features, seq_len]

        # Pass through TCN layers
        out = self.network(x)  # [batch, output_channels, seq_len]

        # Transpose back to [batch, seq_len, output_channels]
        out = out.transpose(1, 2)

        return out


class DualHeadTCN(nn.Module):
    """
    Complete TCN model with dual heads for classification and regression.

    Architecture:
        Input → TCN Encoder → [Direction Head (classification), Excursion Head (regression)]
    """

    def __init__(
        self,
        n_features: int,
        n_channels: list[int] = [64, 64, 128, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.2,
        n_classes: int = 3,
        n_reg_outputs: int = 4,
        pooling: str = 'last'  # 'last', 'mean', or 'max'
    ):
        """
        Args:
            n_features: Number of input features (55 for our dataset)
            n_channels: List of channels for TCN layers (5-7 layers as per specs)
            kernel_size: Convolution kernel size (default 3)
            dropout: Dropout rate
            n_classes: Number of classification classes (3: FLAT, LONG, SHORT)
            n_reg_outputs: Number of regression outputs (4: mfe_l, mae_l, mfe_s, mae_s)
            pooling: How to aggregate sequence output ('last', 'mean', 'max')
        """
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.n_reg_outputs = n_reg_outputs
        self.pooling = pooling

        # TCN Encoder
        self.encoder = TCNEncoder(
            n_inputs=n_features,
            n_channels=n_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )

        hidden_dim = self.encoder.output_channels

        # Direction Classification Head
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

        # MFE/MAE Regression Head
        self.excursion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_reg_outputs)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights."""
        for module in [self.direction_head, self.excursion_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, n_features]

        Returns:
            direction_logits: [batch_size, n_classes] - Raw logits for direction
            excursion_preds: [batch_size, n_reg_outputs] - MFE/MAE predictions
        """
        # Pass through TCN encoder
        encoded = self.encoder(x)  # [batch_size, seq_len, hidden_dim]

        # Aggregate sequence into single vector
        if self.pooling == 'last':
            # Use last timestep
            pooled = encoded[:, -1, :]  # [batch_size, hidden_dim]
        elif self.pooling == 'mean':
            # Average pooling
            pooled = encoded.mean(dim=1)  # [batch_size, hidden_dim]
        elif self.pooling == 'max':
            # Max pooling
            pooled = encoded.max(dim=1)[0]  # [batch_size, hidden_dim]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        # Direction classification
        direction_logits = self.direction_head(pooled)  # [batch_size, n_classes]

        # Excursion regression
        excursion_preds = self.excursion_head(pooled)  # [batch_size, n_reg_outputs]

        return direction_logits, excursion_preds

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with probabilities.

        Args:
            x: Input tensor [batch_size, seq_len, n_features]

        Returns:
            direction_probs: [batch_size, n_classes] - Class probabilities
            direction_preds: [batch_size] - Predicted class indices
            excursion_preds: [batch_size, n_reg_outputs] - MFE/MAE predictions
        """
        direction_logits, excursion_preds = self.forward(x)

        # Convert logits to probabilities
        direction_probs = F.softmax(direction_logits, dim=1)

        # Get predicted class
        direction_preds = direction_logits.argmax(dim=1)

        return direction_probs, direction_preds, excursion_preds


class DualHeadLoss(nn.Module):
    """
    Combined loss for dual-head TCN model.

    Loss = CE(direction) + λ * MSE(mfe/mae regression)
    """

    def __init__(
        self,
        lambda_reg: float = 1.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            lambda_reg: Weight for regression loss (λ in the formula)
            class_weights: Optional class weights for balanced CE loss
        """
        super().__init__()

        self.lambda_reg = lambda_reg

        # Classification loss (Cross-Entropy)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Regression loss (Mean Squared Error)
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        direction_logits: torch.Tensor,
        excursion_preds: torch.Tensor,
        y_class: torch.Tensor,
        y_reg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            direction_logits: [batch_size, n_classes] - Raw logits
            excursion_preds: [batch_size, 4] - Predicted MFE/MAE
            y_class: [batch_size] - True direction labels
            y_reg: [batch_size, 4] - True MFE/MAE values

        Returns:
            total_loss: Combined loss
            ce_loss: Classification loss component
            reg_loss: Regression loss component
        """
        # Classification loss
        ce_loss = self.ce_loss(direction_logits, y_class)

        # Regression loss
        reg_loss = self.mse_loss(excursion_preds, y_reg)

        # Combined loss
        total_loss = ce_loss + self.lambda_reg * reg_loss

        return total_loss, ce_loss, reg_loss


def create_tcn_model(
    n_features: int = 55,
    n_layers: int = 5,
    hidden_channels: int = 128,
    dropout: float = 0.2,
    pooling: str = 'last',
    lambda_reg: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
    device: str = 'cpu'
) -> Tuple[DualHeadTCN, DualHeadLoss]:
    """
    Factory function to create TCN model and loss function.

    Args:
        n_features: Number of input features
        n_layers: Number of TCN layers (5-7 as per specs)
        hidden_channels: Number of channels per layer
        dropout: Dropout rate
        pooling: Sequence pooling method
        lambda_reg: Regression loss weight
        class_weights: Optional class weights for balanced loss
        device: Device to place model on

    Returns:
        model: DualHeadTCN model
        criterion: DualHeadLoss function
    """
    # Create channel list with varying sizes
    # Start smaller, grow in middle layers, stay stable at end
    if n_layers == 5:
        n_channels = [64, 64, 128, 128, 256]
    elif n_layers == 6:
        n_channels = [64, 64, 128, 128, 256, 256]
    elif n_layers == 7:
        n_channels = [64, 64, 128, 128, 256, 256, 512]
    else:
        # Default: use hidden_channels for all layers
        n_channels = [hidden_channels] * n_layers

    # Create model
    model = DualHeadTCN(
        n_features=n_features,
        n_channels=n_channels,
        kernel_size=3,
        dropout=dropout,
        n_classes=3,
        n_reg_outputs=4,
        pooling=pooling
    )

    # Move to device
    model = model.to(device)

    # Create loss function
    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = DualHeadLoss(
        lambda_reg=lambda_reg,
        class_weights=class_weights
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*80}")
    print(f"TCN Model Created")
    print(f"{'='*80}")
    print(f"Architecture:")
    print(f"  Input features: {n_features}")
    print(f"  TCN layers: {n_layers}")
    print(f"  Channel progression: {n_channels}")
    print(f"  Kernel size: 3")
    print(f"  Dilation rates: {[2**i for i in range(n_layers)]}")
    print(f"  Pooling: {pooling}")
    print(f"  Dropout: {dropout}")
    print(f"\nOutput Heads:")
    print(f"  Direction: 3 classes (FLAT, LONG, SHORT)")
    print(f"  Excursion: 4 values (mfe_l, mae_l, mfe_s, mae_s)")
    print(f"\nLoss Configuration:")
    print(f"  Lambda (regression weight): {lambda_reg}")
    print(f"  Class weights: {'Enabled' if class_weights is not None else 'Disabled'}")
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Device: {device}")
    print(f"{'='*80}\n")

    return model, criterion


def demo_model():
    """Demo function to test model architecture."""
    print("="*80)
    print("TASK 12: TCN Model Implementation Demo")
    print("="*80)

    # Check for GPU
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Using device: {device}")

    # Load a sample batch from training data
    print(f"\n📦 Loading sample data...")
    from pytorch_dataset import SequenceDataset

    dataset = SequenceDataset('/Users/ibra/code/swingtrader/data/train_normalized.pkl')

    # Get class weights for balanced training
    class_weights = dataset.get_class_weights()
    print(f"\n⚖️  Class weights: {class_weights}")

    # Create model (5 layers as minimum per specs)
    model, criterion = create_tcn_model(
        n_features=55,
        n_layers=5,
        hidden_channels=128,
        dropout=0.2,
        pooling='last',
        lambda_reg=1.0,
        class_weights=class_weights,
        device=device
    )

    # Test forward pass with a small batch
    print(f"📊 Testing forward pass...")
    batch_size = 8
    X_batch = dataset.X[:batch_size].to(device)  # [8, 168, 55]
    y_class_batch = dataset.y_class[:batch_size].to(device)  # [8]
    y_reg_batch = dataset.y_reg[:batch_size].to(device)  # [8, 4]

    print(f"  Input shape: {X_batch.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        direction_logits, excursion_preds = model(X_batch)

        print(f"\n✓ Forward pass successful!")
        print(f"  Direction logits shape: {direction_logits.shape} (expected: [8, 3])")
        print(f"  Excursion preds shape: {excursion_preds.shape} (expected: [8, 4])")

        # Test prediction function
        direction_probs, direction_preds, _ = model.predict(X_batch)

        print(f"\n📊 Predictions:")
        print(f"  Direction probs shape: {direction_probs.shape}")
        print(f"  Direction preds shape: {direction_preds.shape}")
        print(f"  Predicted classes: {direction_preds.tolist()}")
        print(f"  True classes: {y_class_batch.tolist()}")

        # Test loss computation
        total_loss, ce_loss, reg_loss = criterion(
            direction_logits,
            excursion_preds,
            y_class_batch,
            y_reg_batch
        )

        print(f"\n📉 Loss (untrained model):")
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  CE loss: {ce_loss.item():.4f}")
        print(f"  Reg loss: {reg_loss.item():.4f}")

    # Test single sample inference
    print(f"\n📊 Testing single sample inference...")
    X_single = dataset.X[0:1].to(device)  # [1, 168, 55]

    with torch.no_grad():
        direction_probs, direction_pred, excursion_pred = model.predict(X_single)

        class_names = {0: 'FLAT', 1: 'LONG', 2: 'SHORT'}
        pred_class = direction_pred.item()

        print(f"  Input shape: {X_single.shape}")
        print(f"  Predicted direction: {pred_class} ({class_names[pred_class]})")
        print(f"  Direction probabilities:")
        for i, prob in enumerate(direction_probs[0]):
            print(f"    {i} ({class_names[i]:5s}): {prob.item():.4f}")
        print(f"  Predicted excursions (mfe_l, mae_l, mfe_s, mae_s):")
        print(f"    {excursion_pred[0].tolist()}")

    print("\n" + "="*80)
    print("✅ TASK 12 COMPLETE")
    print("="*80)

    print("\n📦 Model is ready for training!")
    print("\nUsage Example:")
    print("```python")
    print("from tcn_model import create_tcn_model")
    print("from pytorch_dataset import load_all_dataloaders")
    print("")
    print("# Load data")
    print("loaders = load_all_dataloaders('/path/to/data')")
    print("")
    print("# Create model")
    print("model, criterion = create_tcn_model(")
    print("    n_features=55,")
    print("    n_layers=5,")
    print("    class_weights=loaders['train_dataset'].get_class_weights()")
    print(")")
    print("")
    print("# Training loop")
    print("for X, y_class, y_reg in loaders['train']:")
    print("    direction_logits, excursion_preds = model(X)")
    print("    total_loss, ce_loss, reg_loss = criterion(")
    print("        direction_logits, excursion_preds, y_class, y_reg")
    print("    )")
    print("    total_loss.backward()")
    print("```")

    print("\nNext: TASK 13 (Training Loop) 🚀")


if __name__ == "__main__":
    demo_model()

