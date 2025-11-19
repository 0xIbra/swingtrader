#!/usr/bin/env python3
"""
TASK 9: Sequence Window Builder
Construct TCN-ready sliding windows for time series modeling.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path


def select_feature_columns(df: pd.DataFrame) -> list:
    """
    Select relevant feature columns for modeling.

    Exclude:
    - timestamp (not a feature)
    - raw OHLCV (we use derived features instead)
    - raw MFE/MAE values (we use normalized versions)
    - reward columns (derived from labels)
    - direction (this is the target)

    Args:
        df: DataFrame with all features

    Returns:
        List of feature column names
    """
    # Start with all columns
    all_cols = df.columns.tolist()

    # Define columns to exclude
    exclude_cols = [
        'timestamp',           # Not a feature
        'open', 'high', 'low', 'close', 'volume',  # Use derived features instead
        'MFE_long', 'MAE_long', 'MFE_short', 'MAE_short',  # Use normalized versions
        'reward_long', 'reward_short',  # Derived from labels
        'direction',           # This is the target
        'mfe_l', 'mae_l', 'mfe_s', 'mae_s'  # These are regression targets, not features
    ]

    # Select features
    feature_cols = [col for col in all_cols if col not in exclude_cols]

    return feature_cols


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    seq_len: int = 168,
    skip_missing: bool = True
) -> dict:
    """
    Build sliding window sequences for TCN model.

    For each timestamp t with labels:
    - X = features[t-167:t] (shape: [168, feature_dim])
    - y_class = direction[t]
    - y_reg = [mfe_l, mae_l, mfe_s, mae_s][t]

    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature columns to use
        seq_len: Sequence length (default 168 = 1 week)
        skip_missing: Whether to skip windows with missing_flag=1 (default True)

    Returns:
        Dictionary with X, y_class, y_reg arrays and metadata
    """
    print("\n" + "="*80)
    print(f"Building Sequences (seq_len={seq_len})")
    print("="*80)

    print(f"\nFeatures selected: {len(feature_cols)}")
    print(f"Sequence length: {seq_len} bars ({seq_len} hours)")
    print(f"Skip missing data: {skip_missing}")

    # Lists to store sequences
    X_list = []
    y_class_list = []
    y_reg_list = []
    timestamps_list = []

    # Get labeled indices (where direction and MFE/MAE exist)
    labeled_mask = (
        df['direction'].notna() &
        df['mfe_l'].notna() &
        df['mae_l'].notna() &
        df['mfe_s'].notna() &
        df['mae_s'].notna()
    )
    labeled_indices = df[labeled_mask].index.tolist()

    print(f"\nTotal bars with labels: {len(labeled_indices):,}")
    print(f"First valid sequence index: {seq_len - 1}")

    # Filter to indices where we have enough history
    valid_indices = [idx for idx in labeled_indices if idx >= seq_len - 1]

    print(f"Bars with sufficient history ({seq_len} bars): {len(valid_indices):,}")

    # Build sequences
    skipped_missing = 0
    skipped_nan = 0

    for idx in tqdm(valid_indices, desc="Building sequences"):
        # Define window [t-167 : t] (inclusive, so 168 bars total)
        window_start = idx - seq_len + 1
        window_end = idx + 1

        # Extract window
        window = df.iloc[window_start:window_end]

        # Check for missing data flag
        if skip_missing and 'missing_flag' in df.columns:
            if window['missing_flag'].sum() > 0:
                skipped_missing += 1
                continue

        # Extract features
        X = window[feature_cols].values  # Shape: [seq_len, n_features]

        # Check for NaN in features
        if np.isnan(X).any():
            skipped_nan += 1
            continue

        # Extract labels at time t (the last bar)
        y_class = df.loc[idx, 'direction']
        y_reg = np.array([
            df.loc[idx, 'mfe_l'],
            df.loc[idx, 'mae_l'],
            df.loc[idx, 'mfe_s'],
            df.loc[idx, 'mae_s']
        ])

        # Store
        X_list.append(X)
        y_class_list.append(y_class)
        y_reg_list.append(y_reg)
        timestamps_list.append(df.loc[idx, 'timestamp'])

    # Convert to numpy arrays
    X = np.array(X_list, dtype=np.float32)  # [n_samples, seq_len, n_features]
    y_class = np.array(y_class_list, dtype=np.int64)  # [n_samples]
    y_reg = np.array(y_reg_list, dtype=np.float32)  # [n_samples, 4]
    timestamps = np.array(timestamps_list)

    print(f"\n✓ Sequences built: {len(X):,}")
    print(f"  - Skipped (missing data): {skipped_missing:,}")
    print(f"  - Skipped (NaN in features): {skipped_nan:,}")
    print(f"  - Total valid sequences: {len(X):,}")

    print(f"\n📊 Array Shapes:")
    print(f"  - X (features):      {X.shape} = [n_samples, seq_len, n_features]")
    print(f"  - y_class (labels):  {y_class.shape} = [n_samples]")
    print(f"  - y_reg (MFE/MAE):   {y_reg.shape} = [n_samples, 4]")

    print(f"\n📊 Class Distribution in Sequences:")
    unique, counts = np.unique(y_class, return_counts=True)
    class_names = {0: 'FLAT', 1: 'LONG', 2: 'SHORT'}
    for class_id, count in zip(unique, counts):
        pct = (count / len(y_class)) * 100
        print(f"  {class_id} ({class_names[class_id]:5s}): {count:6,} ({pct:5.2f}%)")

    print(f"\n📊 Memory Usage:")
    x_size_mb = X.nbytes / (1024 ** 2)
    y_class_size_mb = y_class.nbytes / (1024 ** 2)
    y_reg_size_mb = y_reg.nbytes / (1024 ** 2)
    total_mb = x_size_mb + y_class_size_mb + y_reg_size_mb

    print(f"  - X:       {x_size_mb:8.2f} MB")
    print(f"  - y_class: {y_class_size_mb:8.2f} MB")
    print(f"  - y_reg:   {y_reg_size_mb:8.2f} MB")
    print(f"  - Total:   {total_mb:8.2f} MB")

    # Create dataset dictionary
    dataset = {
        'X': X,
        'y_class': y_class,
        'y_reg': y_reg,
        'timestamps': timestamps,
        'feature_names': feature_cols,
        'seq_len': seq_len,
        'n_features': len(feature_cols),
        'n_samples': len(X)
    }

    return dataset


def analyze_sequences(dataset: dict) -> None:
    """
    Analyze sequence dataset statistics.

    Args:
        dataset: Dictionary containing X, y_class, y_reg
    """
    print("\n" + "="*80)
    print("Sequence Dataset Analysis")
    print("="*80)

    X = dataset['X']
    y_class = dataset['y_class']
    y_reg = dataset['y_reg']

    # Feature statistics
    print("\n📊 Feature Statistics (across all sequences):")

    # Reshape X to [n_samples * seq_len, n_features] for statistics
    X_flat = X.reshape(-1, X.shape[2])

    feature_names = dataset['feature_names']

    print(f"\n  Top features by variance:")
    feature_vars = np.var(X_flat, axis=0)
    top_var_indices = np.argsort(feature_vars)[-10:][::-1]

    for i, idx in enumerate(top_var_indices, 1):
        print(f"    {i:2d}. {feature_names[idx]:30s} var={feature_vars[idx]:.6f}")

    # Check for constant features
    constant_features = []
    for i, fname in enumerate(feature_names):
        if feature_vars[i] < 1e-10:
            constant_features.append(fname)

    if constant_features:
        print(f"\n  ⚠️  Warning: {len(constant_features)} nearly constant features detected:")
        for fname in constant_features[:5]:
            print(f"    - {fname}")
        if len(constant_features) > 5:
            print(f"    ... and {len(constant_features) - 5} more")

    # Regression target statistics
    print("\n📊 Regression Target Statistics (MFE/MAE):")
    target_names = ['mfe_l', 'mae_l', 'mfe_s', 'mae_s']
    for i, name in enumerate(target_names):
        values = y_reg[:, i]
        print(f"  {name:10s}: mean={values.mean():7.3f}, std={values.std():7.3f}, "
              f"min={values.min():7.3f}, max={values.max():7.3f}")

    # Temporal coverage
    print("\n📊 Temporal Coverage:")
    timestamps = pd.to_datetime(dataset['timestamps'])
    print(f"  - First sequence: {timestamps.min()}")
    print(f"  - Last sequence:  {timestamps.max()}")
    print(f"  - Date range:     {(timestamps.max() - timestamps.min()).days} days")


def save_dataset(dataset: dict, output_dir: str, filename: str) -> str:
    """
    Save dataset to disk in pickle format.

    Args:
        dataset: Dictionary containing sequences
        output_dir: Output directory path
        filename: Output filename

    Returns:
        Full path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename

    print(f"\n📁 Saving dataset to: {filepath}")

    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = filepath.stat().st_size / (1024 ** 2)
    print(f"✓ Dataset saved ({file_size_mb:.2f} MB)")

    return str(filepath)


def main():
    """Main pipeline execution"""

    print("="*80)
    print("TASK 9: Sequence Window Builder")
    print("="*80)

    # Get project root directory (cross-platform)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Load data with direction labels
    input_file = project_root / 'data' / 'EURUSD_1H_2020_2025_with_direction.csv'
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    print(f"✓ Loaded {len(df):,} price bars")
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"✓ Total columns: {len(df.columns)}")

    # Step 1: Select features
    print("\n" + "="*80)
    print("Step 1: Feature Selection")
    print("="*80)

    feature_cols = select_feature_columns(df)

    print(f"\n✓ Selected {len(feature_cols)} features:")

    # Group features by category for display
    feature_categories = {
        'Session': [c for c in feature_cols if 'session' in c or 'hour_' in c or 'dow_' in c],
        'Economic Events': [c for c in feature_cols if 'event' in c or ('hours_' in c and 'event' not in c)],
        'Sentiment': [c for c in feature_cols if 'sent_' in c],
        'Macro Regime': [c for c in feature_cols if any(m in c for m in ['spx_', 'vix_', 'yield10_', 'gold_', 'oil_'])],
        'Price Technical': [c for c in feature_cols if any(t in c for t in ['ret_', 'atr_', 'vol_', 'ema_', 'rsi_', 'body_', 'wick_', 'dist_', 'bars_since'])],
        'Data Quality': [c for c in feature_cols if c == 'missing_flag']
    }

    for category, cols in feature_categories.items():
        if cols:
            print(f"\n  {category} ({len(cols)}):")
            for col in cols[:5]:  # Show first 5
                print(f"    - {col}")
            if len(cols) > 5:
                print(f"    ... and {len(cols) - 5} more")

    # Step 2: Build sequences
    print("\n" + "="*80)
    print("Step 2: Building Sequences")
    print("="*80)

    SEQ_LEN = 168  # 1 week of hourly data
    dataset = build_sequences(df, feature_cols, seq_len=SEQ_LEN, skip_missing=False)

    # Step 3: Analyze sequences
    analyze_sequences(dataset)

    # Step 4: Save dataset
    print("\n" + "="*80)
    print("Step 4: Saving Dataset")
    print("="*80)

    output_dir = project_root / 'data'
    filename = 'sequences_eurusd_1h_168.pkl'

    saved_path = save_dataset(dataset, str(output_dir), filename)

    # Save metadata as readable text
    metadata_file = str(output_dir / 'sequences_metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Sequence Dataset Metadata\n")
        f.write("="*80 + "\n\n")
        f.write(f"File: {filename}\n")
        f.write(f"Created: {pd.Timestamp.now()}\n\n")
        f.write(f"Sequence Configuration:\n")
        f.write(f"  - Sequence length: {dataset['seq_len']} bars (168 hours = 1 week)\n")
        f.write(f"  - Number of features: {dataset['n_features']}\n")
        f.write(f"  - Number of samples: {dataset['n_samples']:,}\n")
        f.write(f"  - Date range: {pd.to_datetime(dataset['timestamps']).min()} to {pd.to_datetime(dataset['timestamps']).max()}\n\n")
        f.write(f"Data Shapes:\n")
        f.write(f"  - X (features): {dataset['X'].shape}\n")
        f.write(f"  - y_class: {dataset['y_class'].shape}\n")
        f.write(f"  - y_reg: {dataset['y_reg'].shape}\n\n")
        f.write(f"Features ({len(dataset['feature_names'])}):\n")
        for i, fname in enumerate(dataset['feature_names'], 1):
            f.write(f"  {i:3d}. {fname}\n")

    print(f"✓ Metadata saved to: {metadata_file}")

    print("\n" + "="*80)
    print("✅ TASK 9 COMPLETE")
    print("="*80)
    print(f"✓ Sequence dataset created: {saved_path}")
    print(f"✓ Total sequences: {dataset['n_samples']:,}")
    print(f"✓ Sequence shape: [n_samples={dataset['n_samples']}, seq_len={dataset['seq_len']}, n_features={dataset['n_features']}]")
    print(f"\n✅ Dataset ready for TASK 10 (Split + Normalization)!")

    # Quick load test
    print("\n📊 Quick Load Test:")
    with open(saved_path, 'rb') as f:
        loaded = pickle.load(f)
    print(f"✓ Dataset loads successfully")
    print(f"✓ Keys: {list(loaded.keys())}")
    print(f"✓ X shape: {loaded['X'].shape}")


if __name__ == "__main__":
    main()

