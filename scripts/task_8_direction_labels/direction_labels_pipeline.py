#!/usr/bin/env python3
"""
TASK 8: Direction Label Generator
Compute LONG / SHORT / FLAT class labels based on MFE/MAE.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


def compute_direction_labels(
    df: pd.DataFrame,
    cost_factor: float = 0.1,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Compute direction class labels (FLAT=0, LONG=1, SHORT=2).

    Given cost factor in ATR units (spread + commission):
    - reward_long = max(mfe_l - cost, 0)
    - reward_short = max(mfe_s - cost, 0)

    Label rule:
    - if max(reward_long, reward_short) < threshold: class = 0 (FLAT)
    - elif reward_long > reward_short: class = 1 (LONG)
    - else: class = 2 (SHORT)

    Args:
        df: DataFrame with mfe_l, mfe_s columns
        cost_factor: Trading cost in ATR units (default 0.1 = 10% of ATR)
        threshold: Minimum reward to consider non-FLAT (default 0.5 ATR)

    Returns:
        DataFrame with direction label columns added
    """
    print("\n" + "="*80)
    print(f"Computing Direction Labels")
    print(f"  Cost Factor: {cost_factor} ATR")
    print(f"  Threshold: {threshold} ATR")
    print("="*80)

    df = df.copy()

    # Compute net rewards after cost
    df['reward_long'] = np.maximum(df['mfe_l'] - cost_factor, 0)
    df['reward_short'] = np.maximum(df['mfe_s'] - cost_factor, 0)

    # Initialize direction class
    df['direction'] = 0  # Default to FLAT

    # Get labeled rows (where MFE/MAE exist)
    labeled_mask = df['mfe_l'].notna() & df['mfe_s'].notna()
    labeled_indices = df[labeled_mask].index

    print(f"\nApplying labeling rules to {len(labeled_indices):,} bars...")

    for idx in labeled_indices:
        reward_l = df.loc[idx, 'reward_long']
        reward_s = df.loc[idx, 'reward_short']
        max_reward = max(reward_l, reward_s)

        if max_reward < threshold:
            # Not enough reward - FLAT
            df.loc[idx, 'direction'] = 0
        elif reward_l > reward_s:
            # Long is better - LONG
            df.loc[idx, 'direction'] = 1
        else:
            # Short is better - SHORT
            df.loc[idx, 'direction'] = 2

    return df


def analyze_direction_labels(df: pd.DataFrame) -> None:
    """
    Analyze and report statistics on direction labels.

    Args:
        df: DataFrame with direction labels
    """
    print("\n" + "="*80)
    print("Direction Label Statistics")
    print("="*80)

    labeled_df = df[df['mfe_l'].notna()].copy()

    # Class distribution
    class_counts = labeled_df['direction'].value_counts().sort_index()
    total = len(labeled_df)

    print("\n📊 Class Distribution:")
    class_names = {0: 'FLAT', 1: 'LONG', 2: 'SHORT'}
    for class_id, count in class_counts.items():
        pct = (count / total) * 100
        print(f"  {class_id} ({class_names[class_id]:5s}): {count:6,} ({pct:5.2f}%)")

    # Check for class imbalance
    print("\n📊 Class Balance Analysis:")
    max_class_pct = (class_counts.max() / total) * 100
    min_class_pct = (class_counts.min() / total) * 100
    imbalance_ratio = class_counts.max() / class_counts.min()

    print(f"  - Largest class:  {max_class_pct:.2f}%")
    print(f"  - Smallest class: {min_class_pct:.2f}%")
    print(f"  - Imbalance ratio: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 3:
        print("  ⚠️  Warning: Significant class imbalance detected (>3:1)")
        print("     Consider adjusting cost_factor or threshold parameters")
    else:
        print("  ✓ Class balance is reasonable")

    # Reward statistics by class
    print("\n📊 Reward Statistics by Class:")
    for class_id in [0, 1, 2]:
        class_df = labeled_df[labeled_df['direction'] == class_id]
        if len(class_df) > 0:
            print(f"\n  {class_names[class_id]}:")
            print(f"    reward_long:  mean={class_df['reward_long'].mean():.4f}, "
                  f"std={class_df['reward_long'].std():.4f}")
            print(f"    reward_short: mean={class_df['reward_short'].mean():.4f}, "
                  f"std={class_df['reward_short'].std():.4f}")
            print(f"    mfe_l: mean={class_df['mfe_l'].mean():.4f}, "
                  f"std={class_df['mfe_l'].std():.4f}")
            print(f"    mfe_s: mean={class_df['mfe_s'].mean():.4f}, "
                  f"std={class_df['mfe_s'].std():.4f}")

    # Temporal distribution
    print("\n📊 Temporal Distribution:")
    labeled_df['year'] = pd.to_datetime(labeled_df['timestamp']).dt.year
    yearly_dist = labeled_df.groupby('year')['direction'].value_counts(normalize=True).unstack(fill_value=0)

    print("\n  Yearly class distribution (%):")
    print(yearly_dist.mul(100).round(2).to_string())


def optimize_parameters(
    df: pd.DataFrame,
    cost_factors: list = [0.15, 0.2, 0.25, 0.3],
    thresholds: list = [1.0, 1.5, 2.0, 2.5, 3.0]
) -> Tuple[float, float]:
    """
    Try different parameter combinations and suggest optimal values.

    Goal: Find parameters that create balanced classes while maximizing trading opportunities.

    Args:
        df: DataFrame with MFE/MAE labels
        cost_factors: List of cost factors to try
        thresholds: List of thresholds to try

    Returns:
        Tuple of (best_cost_factor, best_threshold)
    """
    print("\n" + "="*80)
    print("Parameter Optimization")
    print("="*80)

    print("\nTesting parameter combinations...")

    labeled_df = df[df['mfe_l'].notna()].copy()
    results = []

    for cost in cost_factors:
        for thresh in thresholds:
            # Compute labels with these parameters
            reward_long = np.maximum(labeled_df['mfe_l'] - cost, 0)
            reward_short = np.maximum(labeled_df['mfe_s'] - cost, 0)

            direction = np.zeros(len(labeled_df), dtype=int)

            for i, (idx, row) in enumerate(labeled_df.iterrows()):
                rl = reward_long.iloc[i]
                rs = reward_short.iloc[i]
                max_reward = max(rl, rs)

                if max_reward < thresh:
                    direction[i] = 0  # FLAT
                elif rl > rs:
                    direction[i] = 1  # LONG
                else:
                    direction[i] = 2  # SHORT

            # Compute metrics
            unique, counts = np.unique(direction, return_counts=True)
            class_counts = dict(zip(unique, counts))

            flat_pct = class_counts.get(0, 0) / len(labeled_df) * 100
            long_pct = class_counts.get(1, 0) / len(labeled_df) * 100
            short_pct = class_counts.get(2, 0) / len(labeled_df) * 100

            # Calculate imbalance (standard deviation of class percentages)
            class_pcts = [flat_pct, long_pct, short_pct]
            imbalance = np.std(class_pcts)

            # Calculate trading opportunity (non-FLAT percentage)
            trading_opp = 100 - flat_pct

            # Score: prefer FLAT in 20-30% range for better ML training
            # Lower imbalance is better, FLAT closer to 25% is better
            target_flat = 25.0
            flat_penalty = abs(flat_pct - target_flat) * 3  # Heavily penalize deviation from 25%
            imbalance_penalty = imbalance * 2  # Keep LONG/SHORT balanced
            score = 100 - flat_penalty - imbalance_penalty

            results.append({
                'cost': cost,
                'threshold': thresh,
                'flat_pct': flat_pct,
                'long_pct': long_pct,
                'short_pct': short_pct,
                'imbalance': imbalance,
                'trading_opp': trading_opp,
                'score': score
            })

    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)

    print("\n📊 Top 5 Parameter Combinations:")
    print(results_df.head().to_string(index=False))

    # Best parameters
    best = results_df.iloc[0]
    best_cost = best['cost']
    best_thresh = best['threshold']

    print(f"\n✓ Recommended Parameters:")
    print(f"  - Cost Factor: {best_cost} ATR")
    print(f"  - Threshold: {best_thresh} ATR")
    print(f"  - Results: FLAT={best['flat_pct']:.1f}%, LONG={best['long_pct']:.1f}%, SHORT={best['short_pct']:.1f}%")
    print(f"  - Trading Opportunity: {best['trading_opp']:.1f}%")
    print(f"  - Class Imbalance (std): {best['imbalance']:.2f}")

    return best_cost, best_thresh


def main():
    """Main pipeline execution"""

    print("="*80)
    print("TASK 8: Direction Label Generator")
    print("="*80)

    # Get project root directory (cross-platform)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Scripts are in task_XX subdirectories

    # Load data with MFE/MAE labels
    input_file = project_root / 'data' / 'EURUSD_4H_2020_2025_with_labels.csv'
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    print(f"✓ Loaded {len(df):,} price bars")
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"✓ Current feature count: {len(df.columns)}")

    # Verify required columns exist
    required_cols = ['mfe_l', 'mae_l', 'mfe_s', 'mae_s']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"\n❌ Error: Missing required columns: {missing_cols}")
        return

    print(f"✓ All required columns present: {required_cols}")

    # Step 1: Optimize parameters
    best_cost, best_threshold = optimize_parameters(df)

    # Step 2: Compute direction labels with optimal parameters
    print("\n" + "="*80)
    print("Computing Final Direction Labels")
    print("="*80)

    df = compute_direction_labels(df, cost_factor=best_cost, threshold=best_threshold)

    # Step 3: Analyze results
    analyze_direction_labels(df)

    # Save enhanced data
    output_file = project_root / 'data' / 'EURUSD_4H_2020_2025_with_direction.csv'
    df.to_csv(output_file, index=False)

    print("\n" + "="*80)
    print("✅ TASK 8 COMPLETE")
    print("="*80)
    print(f"✓ Output file: {output_file}")
    print(f"✓ Total rows: {len(df):,}")
    print(f"✓ Total features: {len(df.columns)}")

    print(f"\n✓ New label columns added (3):")
    new_cols = ['reward_long', 'reward_short', 'direction']
    for col in new_cols:
        print(f"  - {col}")

    print("\n📊 Sample of direction labels (first 20 labeled rows):")
    sample_cols = ['timestamp', 'close', 'mfe_l', 'mfe_s', 'reward_long', 'reward_short', 'direction']
    sample_df = df[df['mfe_l'].notna()][sample_cols].head(20)

    # Add readable direction
    direction_map = {0: 'FLAT', 1: 'LONG', 2: 'SHORT'}
    sample_df = sample_df.copy()
    sample_df['dir_label'] = sample_df['direction'].map(direction_map)

    print(sample_df[['timestamp', 'close', 'mfe_l', 'mfe_s', 'direction', 'dir_label']].to_string(index=False))

    print("\n✅ Dataset ready for TASK 9 (Sequence Window Builder)!")


if __name__ == "__main__":
    main()

