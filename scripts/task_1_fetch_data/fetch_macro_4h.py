#!/usr/bin/env python3
"""Fetch macro indicator data and resample to 4H.

Fetches 1H data for macro instruments (SPX, VIX, DXY, GOLD, OIL, 10Y) and resamples to 4H.
Using 1H as source since it's more reliable than 1m for these instruments.
"""

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import os
from pathlib import Path

# EODHD API Configuration
API_KEY = os.environ.get("EODHD_API_TOKEN")
INTERVAL = '1h'
BASE_URL = 'https://eodhd.com/api/intraday/'

# Macro instruments configuration (EODHD format: SYMBOL.EXCHANGE)
MACRO_SYMBOLS = {
    'SPX': 'GSPC.INDX',       # S&P 500 Index
    'VIX': 'VIX.INDX',        # Volatility Index
    'DXY': 'DXY.INDX',        # Dollar Index
    'GOLD': 'GLD.US',         # SPDR Gold Trust ETF (proxy for gold)
    'OIL': 'USO.US',          # United States Oil Fund (proxy for oil)
    '10Y': 'TNX.INDX'         # 10-Year Treasury Yield
}

# Output configuration
OUTPUT_DIR = Path('data/raw/macro')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_eodhd_data(symbol, interval, start_timestamp, end_timestamp, api_key):
    """
    Fetch intraday data from EODHD API

    Args:
        symbol: Trading symbol (e.g., '^GSPC')
        interval: Time interval (e.g., '1h')
        start_timestamp: Start Unix timestamp
        end_timestamp: End Unix timestamp
        api_key: EODHD API key

    Returns:
        DataFrame with OHLCV data
    """

    # Construct the API URL
    url = f"{BASE_URL}{symbol}"
    params = {
        'api_token': api_key,
        'interval': interval,
        'from': start_timestamp,
        'to': end_timestamp,
        'fmt': 'json'
    }

    # Make the request
    response = requests.get(url, params=params)

    # Check if request was successful
    if response.status_code != 200:
        print(f"❌ Error: API returned status code {response.status_code}")
        return None

    # Parse JSON data
    try:
        candle_data = response.json()
    except Exception as e:
        print(f"❌ Error parsing JSON: {e}")
        return None

    if not candle_data:
        return None

    # Convert to DataFrame
    df = pd.DataFrame(candle_data)

    # Use datetime column and convert it
    if 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'])
        columns_to_drop = ['datetime', 'gmtoffset']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    elif 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], (int, float)):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Keep only OHLCV + timestamp columns
    columns_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[[col for col in columns_to_keep if col in df.columns]]

    # Sort by timestamp
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def fetch_macro_instrument(name, symbol, start_date, end_date, api_key):
    """
    Fetch 1H data for a macro instrument.

    Args:
        name: Instrument name (e.g., 'SPX')
        symbol: EODHD symbol (e.g., '^GSPC')
        start_date: Start datetime
        end_date: End datetime
        api_key: EODHD API key

    Returns:
        DataFrame with 1H OHLCV data
    """
    print(f"\n📥 Fetching {name} ({symbol})...")

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    df = fetch_eodhd_data(symbol, INTERVAL, start_timestamp, end_timestamp, api_key)

    if df is not None and len(df) > 0:
        print(f"   ✓ Retrieved {len(df):,} bars")
        print(f"   ✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
    else:
        print(f"   ⚠️  No data retrieved")
        return None


def resample_to_4h(df_1h, name):
    """
    Resample 1H data to 4H.

    Args:
        df_1h: DataFrame with 1H OHLCV data
        name: Instrument name for logging

    Returns:
        DataFrame with 4H OHLCV data
    """
    # Set timestamp as index
    df_1h = df_1h.copy()
    df_1h.set_index('timestamp', inplace=True)
    df_1h.index = pd.to_datetime(df_1h.index)

    # Resample to 4H
    df_4h = df_1h.resample('4H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Remove rows with NaN
    df_4h.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

    # Reset index
    df_4h.reset_index(inplace=True)

    print(f"   🔄 Resampled to {len(df_4h):,} 4H bars")

    return df_4h


def main():
    """
    Main function to fetch and resample macro indicators.
    """
    print("=" * 80)
    print("Macro Indicators Fetcher & Resampler (1H → 4H)")
    print("=" * 80)
    print()

    # Date range (same as EURUSD)
    start_date = datetime(2020, 10, 1)
    end_date = datetime.now()
    end_year = end_date.year

    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📂 Output directory: {OUTPUT_DIR.absolute()}")
    print()

    # Fetch each macro instrument
    macro_data_4h = {}

    for name, symbol in MACRO_SYMBOLS.items():
        try:
            # Fetch 1H data
            df_1h = fetch_macro_instrument(name, symbol, start_date, end_date, API_KEY)

            if df_1h is not None and len(df_1h) > 0:
                # Resample to 4H
                df_4h = resample_to_4h(df_1h, name)

                # Save both 1H and 4H
                output_1h = OUTPUT_DIR / f'{name}_1H_2020_{end_year}.csv'
                output_4h = OUTPUT_DIR / f'{name}_4H_2020_{end_year}.csv'

                df_1h.to_csv(output_1h, index=False)
                df_4h.to_csv(output_4h, index=False)

                print(f"   💾 Saved: {output_1h.name}")
                print(f"   💾 Saved: {output_4h.name}")

                macro_data_4h[name] = df_4h
            else:
                print(f"   ❌ Failed to fetch {name}")

            # Rate limiting
            time.sleep(2)

        except Exception as e:
            print(f"   ❌ Error processing {name}: {e}")
            continue

    print()
    print("=" * 80)

    if macro_data_4h:
        print(f"\n✅ SUCCESS!")
        print(f"📊 Fetched {len(macro_data_4h)} / {len(MACRO_SYMBOLS)} instruments")
        print()
        print("Summary:")
        for name, df in macro_data_4h.items():
            print(f"   {name:6s}: {len(df):,} 4H bars | {df['timestamp'].min()} to {df['timestamp'].max()}")
    else:
        print("\n❌ No data was fetched!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
