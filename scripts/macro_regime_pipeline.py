#!/usr/bin/env python3
"""
TASK 5: Macro Regime Pipeline
Fetch external macro indicators (VIX, SPX, yields, DXY, GOLD, OIL) and compute features.
"""

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# EODHD API Configuration
API_KEY = os.environ.get("EODHD_API_TOKEN")
BASE_URL = 'https://eodhd.com/api/intraday/'

# Macro symbols to fetch (EODHD format: SYMBOL.EXCHANGE)
MACRO_SYMBOLS = {
    'SPX': 'GSPC.INDX',       # SP500 Index
    'VIX': 'VIX.INDX',        # VIX Volatility Index
    'YIELD10': 'TNX.INDX',    # US 10-Year Treasury Yield
    'DXY': 'DXY.INDX',        # Dollar Index
    'GOLD': 'GLD.US',         # SPDR Gold Trust ETF (proxy for gold)
    'OIL': 'USO.US'           # United States Oil Fund (proxy for oil)
}


def fetch_macro_data(symbol: str, start_timestamp: int, end_timestamp: int, api_key: str) -> pd.DataFrame:
    """
    Fetch 1h intraday data for a macro indicator from EODHD API.

    Args:
        symbol: Trading symbol (e.g., '^GSPC', '^VIX')
        start_timestamp: Start Unix timestamp
        end_timestamp: End Unix timestamp
        api_key: EODHD API key

    Returns:
        DataFrame with OHLCV data
    """
    url = f"{BASE_URL}{symbol}"
    params = {
        'api_token': api_key,
        'interval': '1h',
        'from': start_timestamp,
        'to': end_timestamp,
        'fmt': 'json'
    }

    try:
        response = requests.get(url, params=params, timeout=60)

        if response.status_code != 200:
            print(f"  ⚠ Error: API returned status code {response.status_code} for {symbol}")
            return pd.DataFrame()

        data = response.json()

        if not data:
            print(f"  ⚠ No data returned for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Convert datetime to timestamp
        if 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])
            df.drop(columns=['datetime', 'gmtoffset'], errors='ignore', inplace=True)
        elif 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], (int, float)):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Keep only necessary columns
        columns_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in columns_to_keep if col in df.columns]]

        # Sort by timestamp
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"  ✓ {symbol}: {len(df):,} bars retrieved")

        return df

    except Exception as e:
        print(f"  ⚠ Error fetching {symbol}: {e}")
        return pd.DataFrame()


def fetch_all_macro_indicators(start_date: datetime, end_date: datetime, api_key: str) -> dict:
    """
    Fetch all macro indicators with 1h frequency.

    Returns:
        Dictionary mapping indicator names to DataFrames
    """
    print("\nFetching macro indicators from EODHD API...")
    print(f"Date range: {start_date} to {end_date}\n")

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    macro_data = {}

    for name, symbol in tqdm(MACRO_SYMBOLS.items(), desc="Fetching macro indicators"):
        df = fetch_macro_data(symbol, start_timestamp, end_timestamp, api_key)

        if not df.empty:
            macro_data[name] = df

        # Rate limiting
        time.sleep(0.5)

    return macro_data


def compute_macro_features(price_df: pd.DataFrame, macro_data: dict) -> pd.DataFrame:
    """
    Compute macro regime features and align with price data.

    For each 1h bar compute:
    - returns: spx_ret_1h, vix_ret_1h, dxy_ret_1h, gold_ret_1h, oil_ret_1h
    - absolute levels: vix_level, yield10_level, dxy_level, spx_level, gold_level, oil_level
    - vol-adjusted z-scores (rolling 24h window): vix_z, dxy_z, yield10_z, spx_z, gold_z, oil_z
    """
    print("\nComputing macro regime features...")

    price_df = price_df.copy()
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)

    # Store original row count to check for duplicates
    original_rows = len(price_df)
    print(f"Original price data rows: {original_rows:,}")

    # Process each macro indicator
    for name, df in macro_data.items():
        print(f"\nProcessing {name}...")

        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Compute 1h log returns
        df[f'{name.lower()}_ret_1h'] = np.log(df['close'] / df['close'].shift(1))

        # Store absolute level (close price)
        df[f'{name.lower()}_level'] = df['close']

        # Compute rolling 24h statistics for z-score
        window = 24
        df[f'{name.lower()}_mean_24h'] = df['close'].rolling(window=window, min_periods=1).mean()
        df[f'{name.lower()}_std_24h'] = df['close'].rolling(window=window, min_periods=1).std()

        # Compute z-score (vol-adjusted)
        df[f'{name.lower()}_z'] = (df['close'] - df[f'{name.lower()}_mean_24h']) / (df[f'{name.lower()}_std_24h'] + 1e-8)

        # Select features to merge
        feature_cols = [
            'timestamp',
            f'{name.lower()}_ret_1h',
            f'{name.lower()}_level',
            f'{name.lower()}_z'
        ]

        # Remove duplicates from macro data before merging
        duplicates = df[df['timestamp'].duplicated(keep=False)]
        if len(duplicates) > 0:
            print(f"  ⚠ Found {len(duplicates)} duplicate timestamps in {name} data, removing...")
            df = df.drop_duplicates(subset=['timestamp'], keep='first')

        # Merge with price data (left join to keep all price bars)
        before_merge = len(price_df)
        price_df = price_df.merge(
            df[feature_cols],
            on='timestamp',
            how='left'
        )
        after_merge = len(price_df)

        if after_merge != before_merge:
            print(f"  ⚠ Warning: Row count changed from {before_merge:,} to {after_merge:,} after merge!")
            print(f"  ⚠ This should not happen - check for issues in data")

        print(f"  ✓ Added features: {feature_cols[1:]}")

    # Forward-fill missing macro data (when markets are closed)
    macro_feature_cols = [col for col in price_df.columns if any(
        macro.lower() in col for macro in ['spx', 'vix', 'dxy', 'yield10', 'gold', 'oil']
    )]

    print(f"\nForward-filling {len(macro_feature_cols)} macro features...")
    for col in macro_feature_cols:
        price_df[col] = price_df[col].ffill()

    # Fill any remaining NaNs at the beginning with 0 (for returns) or mean (for levels/z-scores)
    for col in macro_feature_cols:
        if '_ret_' in col:
            price_df[col] = price_df[col].fillna(0)
        elif '_z' in col:
            price_df[col] = price_df[col].fillna(0)
        else:
            price_df[col] = price_df[col].fillna(price_df[col].mean())

    print(f"\n✓ Generated {len(macro_feature_cols)} macro features")

    return price_df


def main():
    """Main pipeline execution"""

    print("="*80)
    print("TASK 5: Macro Regime Pipeline")
    print("="*80)

    # Load existing price data with sentiment
    price_file = '/Users/ibra/code/swingtrader/EURUSD_1H_2020_2025_with_sentiment.csv'
    print(f"\nLoading price data from: {price_file}")
    price_df = pd.read_csv(price_file)
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)

    print(f"✓ Loaded {len(price_df):,} price bars")
    print(f"✓ Date range: {price_df['timestamp'].min()} to {price_df['timestamp'].max()}")
    print(f"✓ Current feature count: {len(price_df.columns)}")

    # Determine date range
    start_date = price_df['timestamp'].min().to_pydatetime()
    end_date = price_df['timestamp'].max().to_pydatetime()

    # Step 1: Fetch all macro indicators
    print("\n" + "="*80)
    print("Step 1: Fetching macro indicators")
    print("="*80)
    macro_data = fetch_all_macro_indicators(start_date, end_date, API_KEY)

    if not macro_data:
        print("\n⚠ No macro data fetched. Exiting.")
        return

    print(f"\n✓ Successfully fetched {len(macro_data)} macro indicators:")
    for name, df in macro_data.items():
        print(f"  - {name}: {len(df):,} bars, {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Step 2: Compute macro features
    print("\n" + "="*80)
    print("Step 2: Computing macro regime features")
    print("="*80)
    price_df = compute_macro_features(price_df, macro_data)

    # Step 3: Save enhanced data
    output_file = '/Users/ibra/code/swingtrader/EURUSD_1H_2020_2025_with_macro.csv'
    price_df.to_csv(output_file, index=False)

    print("\n" + "="*80)
    print("✅ TASK 5 COMPLETE")
    print("="*80)
    print(f"✓ Output file: {output_file}")
    print(f"✓ Total rows: {len(price_df):,}")
    print(f"✓ Total features: {len(price_df.columns)}")

    print(f"\n✓ All feature columns:")
    for col in price_df.columns:
        print(f"  - {col}")

    print("\n📊 Sample of macro features (first 5 rows):")
    macro_cols = [col for col in price_df.columns if any(
        macro.lower() in col for macro in ['spx', 'vix', 'dxy', 'yield10', 'gold', 'oil']
    )]
    if macro_cols:
        print(price_df[['timestamp'] + macro_cols].head())

    print("\n📊 Sample of macro features (last 5 rows):")
    if macro_cols:
        print(price_df[['timestamp'] + macro_cols].tail())

    print("\n📊 Macro feature statistics:")
    if macro_cols:
        print(price_df[macro_cols].describe())


if __name__ == "__main__":
    main()

