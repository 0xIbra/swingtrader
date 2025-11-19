#!/usr/bin/env python3
"""
Filter sequences to only 2024-2025 data.
Train on RECENT market conditions only.
"""

import pickle
import numpy as np
from pathlib import Path
from datetime import datetime


def filter_sequences_by_date(
    sequences_file: str,
    start_date: str,
    output_file: str
):
    """
    Filter sequences to only include dates >= start_date.

    Args:
        sequences_file: Path to sequences pickle file
        start_date: ISO format date string (e.g. '2024-01-01T00:00:00+00:00')
        output_file: Path to save filtered sequences
    """
    print("\n" + "="*80)
    print("FILTERING SEQUENCES TO 2024-2025 DATA ONLY")
    print("="*80)

    # Load sequences
    print(f"\n📦 Loading: {sequences_file}")
    with open(sequences_file, 'rb') as f:
        data = pickle.load(f)

    X = data['X']
    y_class = data['y_class']
    y_reg = data['y_reg']
    timestamps = data['timestamps']

    print(f"  ✓ Loaded {len(timestamps):,} sequences")
    print(f"  ✓ Date range: {timestamps[0]} to {timestamps[-1]}")

    # Convert timestamps to datetime for comparison
    if isinstance(timestamps[0], str):
        timestamps_dt = np.array([datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps])
    else:
        timestamps_dt = timestamps

    start = datetime.fromisoformat(start_date)

    # Create mask for sequences to KEEP (>= start_date)
    keep_mask = timestamps_dt >= start

    n_kept = keep_mask.sum()
    n_removed = len(keep_mask) - n_kept
    pct_kept = (n_kept / len(keep_mask)) * 100

    print(f"\n📊 Filtering Results:")
    print(f"  Start date:  {start_date}")
    print(f"  Original:    {len(keep_mask):,} sequences")
    print(f"  Kept:        {n_kept:,} sequences ({pct_kept:.1f}%)")
    print(f"  Removed:     {n_removed:,} sequences")

    # Filter all arrays
    X_filtered = X[keep_mask]
    y_class_filtered = y_class[keep_mask]
    y_reg_filtered = y_reg[keep_mask]
    timestamps_filtered = timestamps[keep_mask]

    # Class distribution
    unique, counts = np.unique(y_class_filtered, return_counts=True)
    print(f"\n📊 Class Distribution (2024-2025):")
    for cls, count in zip(unique, counts):
        pct = count / len(y_class_filtered) * 100
        class_name = ['FLAT', 'LONG', 'SHORT'][cls]
        print(f"  {class_name:5s}: {count:6,} ({pct:5.2f}%)")

    # Save filtered data
    print(f"\n💾 Saving filtered sequences to: {output_file}")
    filtered_data = {
        'X': X_filtered,
        'y_class': y_class_filtered,
        'y_reg': y_reg_filtered,
        'timestamps': timestamps_filtered,
        'feature_names': data['feature_names'],
        'seq_len': data['seq_len'],
        'n_features': data['n_features'],
        'n_samples': len(X_filtered)
    }

    with open(output_file, 'wb') as f:
        pickle.dump(filtered_data, f)

    print(f"  ✓ Saved {len(X_filtered):,} filtered sequences")
    print(f"\n📅 New Date Range: {timestamps_filtered[0]} to {timestamps_filtered[-1]}")

    return filtered_data


def main():
    print("="*80)
    print("TRAIN ON RECENT DATA ONLY (2024-2025)")
    print("="*80)
    print("\n💡 Strategy: Train on patterns that are RELEVANT to current markets")
    print("  ✓ Avoids regime shift problems")
    print("  ✓ Learns from recent price action")
    print("  ✓ Better generalization to near-future")

    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / 'data'

    sequences_file = data_dir / 'sequences_eurusd_1h_168.pkl'
    output_file = data_dir / 'sequences_eurusd_1h_168_2024_2025.pkl'

    # Filter to 2024-01-01 onwards
    start_date = '2024-01-01T00:00:00+00:00'

    print(f"\n🎯 Keeping data from: {start_date} onwards")
    print(f"   Expected: ~12,000 sequences")

    # Filter sequences
    filtered_data = filter_sequences_by_date(
        str(sequences_file),
        start_date,
        str(output_file)
    )

    print("\n" + "="*80)
    print("✅ FILTERING COMPLETE")
    print("="*80)
    print(f"\n📁 Filtered data saved to: {output_file}")
    print(f"\nNext steps:")
    print(f"  1. Split and normalize (60/20/20)")
    print(f"  2. Train TCN on recent patterns")
    print(f"  3. Expect MUCH better test performance!")


if __name__ == '__main__':
    main()

