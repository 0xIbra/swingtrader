#!/usr/bin/env python3
"""Fetch EODHD 1-minute candlestick data for EURUSD.

This script fetches 1-minute intraday data in monthly chunks to avoid API timeouts.
Data is saved in monthly partitions and then combined into a single file.
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
SYMBOL = 'EURUSD.FOREX'
INTERVAL = '1m'
BASE_URL = 'https://eodhd.com/api/intraday/'

# Output configuration
OUTPUT_DIR = Path('data/raw/1m')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_eodhd_data(symbol, interval, start_timestamp, end_timestamp, api_key):
    """
    Fetch intraday data from EODHD API

    Args:
        symbol: Trading symbol (e.g., 'EURUSD.FOREX')
        interval: Time interval (e.g., '1m', '5m', '1h')
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

    # Convert timestamps to readable dates for display
    start_date_display = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    end_date_display = datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')

    print(f"Fetching {interval} data for {symbol}")
    print(f"Date range: {start_date_display} to {end_date_display}")

    # Make the request with progress bar
    with tqdm(total=100, desc="Downloading", unit="%", ncols=80, leave=False) as pbar:
        response = requests.get(url, params=params, stream=True)

        # Get total size if available
        total_size = int(response.headers.get('content-length', 0))

        if total_size > 0:
            # Download with actual progress
            block_size = 8192
            downloaded = 0
            data = b""

            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    data += chunk
                    downloaded += len(chunk)
                    progress = (downloaded / total_size) * 100
                    pbar.update(progress - pbar.n)

            pbar.update(100 - pbar.n)
        else:
            # No content-length header, just show spinner
            data = response.content
            pbar.update(100)

    # Check if request was successful
    if response.status_code != 200:
        print(f"❌ Error: API returned status code {response.status_code}")
        print(f"Response: {response.text[:500]}")
        return None

    # Parse JSON data
    try:
        candle_data = response.json()
    except Exception as e:
        print(f"❌ Error parsing JSON: {e}")
        print(f"Response text: {response.text[:500]}")
        return None

    if not candle_data:
        print("⚠️  No data returned from API")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(candle_data)

    # Use datetime column and convert it
    if 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'])
        # Drop the original datetime column and Unix timestamp if present
        columns_to_drop = ['datetime', 'gmtoffset']
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    elif 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], (int, float)):
        # Convert Unix timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Keep only OHLCV + timestamp columns
    columns_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[[col for col in columns_to_keep if col in df.columns]]

    # Sort by timestamp
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"✓ Retrieved {len(df):,} candlestick entries")
    if len(df) > 0:
        print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def fetch_month_chunk(year, month, symbol, interval, api_key):
    """
    Fetch data for a specific month.

    Args:
        year: Year (e.g., 2020)
        month: Month (1-12)
        symbol: Trading symbol
        interval: Time interval
        api_key: EODHD API key

    Returns:
        DataFrame with OHLCV data for that month
    """
    # Calculate month boundaries
    start_date = datetime(year, month, 1)

    # Calculate last day of month
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(seconds=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    return fetch_eodhd_data(symbol, interval, start_timestamp, end_timestamp, api_key)


def main():
    """
    Main function to fetch 1-minute data in monthly chunks from October 2020 to present.
    """
    print("=" * 80)
    print("EURUSD 1-Minute Data Fetcher (Monthly Chunks)")
    print("=" * 80)
    print()

    # Calculate date range
    start_year = 2020
    start_month = 10  # October (EODHD limitation)
    end_date = datetime.now()
    end_year = end_date.year
    end_month = end_date.month

    # Generate list of (year, month) tuples
    months_to_fetch = []
    current_year = start_year
    current_month = start_month

    while (current_year, current_month) <= (end_year, end_month):
        months_to_fetch.append((current_year, current_month))

        # Move to next month
        if current_month == 12:
            current_year += 1
            current_month = 1
        else:
            current_month += 1

    print(f"📅 Fetching data for {len(months_to_fetch)} months (Oct 2020 - {end_date.strftime('%b %Y')})")
    print(f"📂 Output directory: {OUTPUT_DIR.absolute()}")
    print()

    # Fetch data month by month
    all_dataframes = []
    failed_months = []

    for year, month in tqdm(months_to_fetch, desc="Overall Progress", unit="month"):
        month_str = f"{year}-{month:02d}"
        output_file = OUTPUT_DIR / f"EURUSD_1m_{month_str}.csv"

        # Skip if already exists
        if output_file.exists():
            print(f"⏭️  Skipping {month_str} (already exists)")
            df = pd.read_csv(output_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            all_dataframes.append(df)
            continue

        print(f"\n📥 Fetching {month_str}...")

        try:
            df = fetch_month_chunk(year, month, SYMBOL, INTERVAL, API_KEY)

            if df is not None and len(df) > 0:
                # Save monthly file
                df.to_csv(output_file, index=False)
                print(f"💾 Saved to: {output_file.name}")
                all_dataframes.append(df)
            else:
                print(f"⚠️  No data for {month_str}")
                failed_months.append(month_str)

            # Rate limiting - be nice to the API
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error fetching {month_str}: {e}")
            failed_months.append(month_str)
            continue

    print()
    print("=" * 80)

    # Combine all months into single file
    if all_dataframes:
        print(f"\n🔗 Combining {len(all_dataframes)} monthly files...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df.sort_values('timestamp', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        # Remove duplicates (in case of overlapping data)
        before_dedup = len(combined_df)
        combined_df.drop_duplicates(subset=['timestamp'], inplace=True)
        after_dedup = len(combined_df)
        if before_dedup != after_dedup:
            print(f"🧹 Removed {before_dedup - after_dedup:,} duplicate timestamps")

        # Save combined file
        combined_file = OUTPUT_DIR / f'EURUSD_1m_2020_{end_year}.csv'
        combined_df.to_csv(combined_file, index=False)

        print(f"\n✅ SUCCESS!")
        print(f"📊 Total rows: {len(combined_df):,}")
        print(f"📅 Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        print(f"💾 Combined file: {combined_file.absolute()}")

        # Display sample
        print(f"\n📋 First 5 rows:")
        print(combined_df.head())
        print(f"\n📋 Last 5 rows:")
        print(combined_df.tail())

    else:
        print("❌ No data was fetched!")

    if failed_months:
        print(f"\n⚠️  Failed months: {', '.join(failed_months)}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
