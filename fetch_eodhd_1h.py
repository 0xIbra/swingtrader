#!/usr/bin/env python3
"""Fetch EODHD 1-hour candlestick data for EURUSD."""

from __future__ import annotations


from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import os

# EODHD API Configuration
API_KEY = os.environ.get("EODHD_API_TOKEN")
SYMBOL = 'EURUSD.FOREX'  # EODHD forex format
INTERVAL = '1h'
BASE_URL = 'https://eodhd.com/api/intraday/'

def fetch_eodhd_data(symbol, interval, start_timestamp, end_timestamp, api_key):
    """
    Fetch intraday data from EODHD API

    Args:
        symbol: Trading symbol (e.g., 'EURUSD.FOREX')
        interval: Time interval (e.g., '1h', '5m', '1m')
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
    print(f"URL: {url}")
    print()

    # Make the request with progress bar
    with tqdm(total=100, desc="Downloading data", unit="%", ncols=80) as pbar:
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

    print()

    # Check if request was successful
    if response.status_code != 200:
        print(f"Error: API returned status code {response.status_code}")
        print(f"Response: {response.text}")
        return None

    # Parse JSON data
    try:
        candle_data = response.json()
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(f"Response text: {response.text[:500]}")
        return None

    if not candle_data:
        print("No data returned from API")
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
    print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()

    return df


def main():
    # Calculate date range (from October 2020 to now)
    # Note: EODHD API only provides intraday forex data from October 2020 onwards
    end_date = datetime.now()
    start_date = datetime(2020, 10, 1)  # October 1, 2020 (earliest available)

    # Convert to Unix timestamps
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    # Fetch data
    df = fetch_eodhd_data(
        symbol=SYMBOL,
        interval=INTERVAL,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        api_key=API_KEY
    )

    if df is not None:
        # Save to CSV
        output_file = f'EURUSD_1H_2020_{end_date.year}.csv'
        df.to_csv(output_file, index=False)
        print(f"✓ Data saved to: {output_file}")
        print(f"✓ Total rows: {len(df):,}")
        print(f"✓ Date range in file: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Display sample
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nLast 5 rows:")
        print(df.tail())
        print("\nDataFrame info:")
        print(df.info())
    else:
        print("Failed to fetch data. Please check your API key and parameters.")


if __name__ == "__main__":
    main()

