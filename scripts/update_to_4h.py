#!/usr/bin/env python3
"""
Master script to update all pipeline parameters from 1H to 4H timeframe.

This script updates:
- File paths (EURUSD_1H → EURUSD_4H)
- Sequence length (168 bars → 42 bars)
- MFE/MAE horizon (24 bars → 6 bars)
- Rolling windows (24, 72, 168 → 6, 18, 42)
- EMA periods (24, 72, 168 → 6, 18, 42)
"""

from pathlib import Path
import re

# Project configuration
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'

# File path replacements
FILE_REPLACEMENTS = {
    'EURUSD_1H_2020': 'EURUSD_4H_2020',
    'sequences_eurusd_1h_168': 'sequences_eurusd_4h_42',
}

# Parameter replacements (for specific contexts)
PARAM_REPLACEMENTS = {
    # Sequence length: 168 → 42 (1 week of 4H data)
    'seq_len: int = 168': 'seq_len: int = 42',
    'SEQ_LEN = 168': 'SEQ_LEN = 42',
    'sequences_eurusd_1h_168': 'sequences_eurusd_4h_42',

    # MFE/MAE horizon: 24 bars → 6 bars (24 hours)
    'horizon = 24': 'horizon = 6',

    # Rolling windows for volatility: 24, 72, 168 → 6, 18, 42
    'windows=[24, 72, 168]': 'windows=[6, 18, 42]',

    # EMA periods: 24, 72, 168 → 6, 18, 42
    'periods=[24, 72, 168]': 'periods=[6, 18, 42]',

    # Recent structure window: 168 → 42
    'window=168': 'window=42',
    'window: int = 168': 'window: int = 42',
}

# Comment/docstring replacements
DOC_REPLACEMENTS = {
    '168 bars (168 hours = 1 week)': '42 bars (168 hours = 1 week of 4H data)',
    '168 hours = 1 week': '168 hours = 1 week of 4H data',
    '24 bars (24 hours = 1 day)': '6 bars (24 hours = 1 day of 4H data)',
    '1 week of hourly data': '1 week of 4-hour data',
    'window (default 168 = 1 week)': 'window (default 42 = 1 week)',
    '[24, 72, 168]': '[6, 18, 42]',
    '24, 72, 168': '6, 18, 42',
    '1 day, 3 days, 1 week': '1 day, 3 days, 1 week',
    '24h, 72h, 168h': '24h, 72h, 168h',
    'vol_24h': 'vol_24h',
    'vol_72h': 'vol_72h',
    'vol_168h': 'vol_168h',
    'ema_24': 'ema_24',
    'ema_72': 'ema_72',
    'ema_168': 'ema_168',
}

# Files to update
FILES_TO_UPDATE = [
    SCRIPTS_DIR / 'task_2_sessions' / 'session_feature_generator.py',
    SCRIPTS_DIR / 'task_3_economic_events' / 'economic_calendar_pipeline.py',
    SCRIPTS_DIR / 'task_4_news_sentiment' / 'news_sentiment_pipeline.py',
    SCRIPTS_DIR / 'task_5_macro_indicators' / 'macro_regime_pipeline.py',
    SCRIPTS_DIR / 'task_6_price_features' / 'price_features_pipeline.py',
    SCRIPTS_DIR / 'task_7_mfe_mae_labels' / 'mfe_mae_labels_pipeline.py',
    SCRIPTS_DIR / 'task_8_direction_labels' / 'direction_labels_pipeline.py',
    SCRIPTS_DIR / 'task_9_sequences' / 'sequence_window_builder.py',
    SCRIPTS_DIR / 'task_10_split_normalize' / 'dataset_split_normalize.py',
]


def update_file_content(file_path: Path) -> None:
    """Update a single file with 4H parameters."""

    print(f"\n📝 Updating: {file_path.relative_to(PROJECT_ROOT)}")

    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes = []

    # Apply file path replacements
    for old, new in FILE_REPLACEMENTS.items():
        if old in content:
            content = content.replace(old, new)
            changes.append(f"  - {old} → {new}")

    # Apply parameter replacements
    for old, new in PARAM_REPLACEMENTS.items():
        if old in content:
            content = content.replace(old, new)
            changes.append(f"  - {old} → {new}")

    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"   ✅ Applied {len(changes)} changes:")
        for change in changes[:5]:  # Show first 5 changes
            print(change)
        if len(changes) > 5:
            print(f"   ... and {len(changes) - 5} more")
    else:
        print("   ⏭️  No changes needed")


def main():
    """Main execution."""

    print("=" * 80)
    print("Updating Pipeline Scripts: 1H → 4H Timeframe")
    print("=" * 80)

    print(f"\n📂 Project root: {PROJECT_ROOT}")
    print(f"📂 Scripts directory: {SCRIPTS_DIR}")

    print(f"\n🔧 Will update {len(FILES_TO_UPDATE)} files:")
    for file_path in FILES_TO_UPDATE:
        print(f"   - {file_path.relative_to(PROJECT_ROOT)}")

    print("\n" + "=" * 80)

    # Update each file
    for file_path in FILES_TO_UPDATE:
        if not file_path.exists():
            print(f"\n⚠️  File not found: {file_path.relative_to(PROJECT_ROOT)}")
            continue

        update_file_content(file_path)

    print("\n" + "=" * 80)
    print("✅ Update complete!")
    print("=" * 80)

    print("\n📋 Summary of key changes:")
    print("   - Sequence length: 168 bars → 42 bars")
    print("   - MFE/MAE horizon: 24 bars → 6 bars")
    print("   - Rolling windows: [24, 72, 168] → [6, 18, 42]")
    print("   - File paths: EURUSD_1H → EURUSD_4H")
    print("   - Sequence files: sequences_eurusd_1h_168 → sequences_eurusd_4h_42")


if __name__ == "__main__":
    main()
