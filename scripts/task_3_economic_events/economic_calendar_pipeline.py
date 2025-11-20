#!/usr/bin/env python3
"""
TASK 3: Economic Calendar Pipeline
Fetch economic events from EODHD API and generate proximity features.
"""

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import time

# EODHD API Configuration
API_KEY = os.environ.get("EODHD_API_TOKEN")
BASE_URL = 'https://eodhd.com/api/economic-events'

def fetch_economic_events(start_date: datetime, end_date: datetime, api_key: str) -> pd.DataFrame:
    """
    Fetch ALL economic calendar events from EODHD API using offset-based pagination.

    Args:
        start_date: Start date for events
        end_date: End date for events
        api_key: EODHD API key

    Returns:
        DataFrame with economic events
    """
    all_events = []
    url = BASE_URL

    print(f"\nFetching economic events from EODHD API...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print()

    # Format dates for API
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')

    # Pagination parameters
    limit = 1000  # Max per request
    offset = 0
    total_fetched = 0

    print("📥 Fetching with offset-based pagination...")

    with tqdm(desc="Fetching events", unit=" batches") as pbar:
        while True:
            params = {
                'api_token': api_key,
                'from': from_date,
                'to': to_date,
                'limit': limit,
                'offset': offset,
                'fmt': 'json'
            }

            try:
                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    batch = []

                    # EODHD returns events in a nested structure sometimes, or list
                    if isinstance(data, dict):
                        # Flatten the nested structure if it comes as dict
                        # But with limit/offset it usually returns a list or specific structure
                        # Let's handle the standard list response or the dict response
                        if 'data' in data:
                             batch = data['data']
                        else:
                            # Sometimes it returns {date: [events]}
                            for date_str, events_list in data.items():
                                if isinstance(events_list, list):
                                    for event in events_list:
                                        event['date'] = date_str
                                        batch.append(event)
                    elif isinstance(data, list):
                        batch = data
                    else:
                        print(f"\n⚠️  Unexpected response format at offset {offset}")
                        break

                    # Add batch to results
                    batch_size = len(batch)
                    all_events.extend(batch)
                    total_fetched += batch_size

                    pbar.set_postfix({"fetched": total_fetched, "offset": offset})
                    pbar.update(1)

                    # Check if we got fewer than limit (means we're done)
                    if batch_size < limit:
                        print(f"\n✓ Reached end of results (got {batch_size} < {limit})")
                        break

                    # Move to next page
                    offset += limit

                    # Rate limiting
                    time.sleep(0.5)

                else:
                    print(f"\n❌ API returned status code {response.status_code}")
                    if response.text:
                        print(f"Response: {response.text[:200]}")
                    break

            except Exception as e:
                print(f"\n❌ Error fetching data at offset {offset}: {e}")
                break

    if not all_events:
        print("Warning: No events retrieved from API")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_events)

    # Remove duplicates
    if 'date' in df.columns and 'event' in df.columns:
        before_dedup = len(df)
        # Create a unique key for deduplication
        df['dedup_key'] = df['date'].astype(str) + df['event'].astype(str) + df['country'].astype(str)
        df = df.drop_duplicates(subset=['dedup_key'], keep='first')
        df = df.drop(columns=['dedup_key'])
        after_dedup = len(df)
        if before_dedup != after_dedup:
            print(f"🧹 Removed {before_dedup - after_dedup:,} duplicate events")

    print(f"\n✓ Retrieved {len(df):,} economic events")

    return df


def process_economic_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw economic events into standardized format.

    Required fields:
    - event_time (UTC)
    - currency (USD, EUR, GBP, JPY, etc.)
    - impact_level (0-2)
    - event_type (categorical)
    """
    if df.empty:
        return df

    # Standardize column names (EODHD API field names may vary)
    column_mapping = {
        'date': 'event_date',
        'time': 'event_time',
        'country': 'currency',
        'importance': 'impact_level',
        'event': 'event_type',
        'type': 'event_type',  # EODHD uses 'type' for event name
        'actual': 'actual_value',
        'forecast': 'forecast_value',
        'previous': 'previous_value'
    }

    # Rename columns if they exist
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Handle event_time from date column (EODHD format)
    if 'event_date' in df.columns:
        df['event_time'] = pd.to_datetime(df['event_date'], errors='coerce', utc=True)
        df.drop(columns=['event_date'], inplace=True)
    elif 'date' in df.columns:
        df['event_time'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

    # Map currency codes (country names to currency codes)
    country_to_currency = {
        'United States': 'USD',
        'US': 'USD',
        'USA': 'USD',
        'Eurozone': 'EUR',
        'Euro': 'EUR',
        'EUR': 'EUR',
        'Germany': 'EUR',
        'France': 'EUR',
        'Italy': 'EUR',
        'Spain': 'EUR',
        'United Kingdom': 'GBP',
        'UK': 'GBP',
        'Britain': 'GBP',
        'Japan': 'JPY',
        'Switzerland': 'CHF',
        'Canada': 'CAD',
        'Australia': 'AUD',
        'New Zealand': 'NZD',
        'China': 'CNY',
    }

    if 'currency' in df.columns:
        df['currency'] = df['currency'].replace(country_to_currency)
        # Keep only rows with valid 3-letter currency codes
        df = df[df['currency'].str.len() == 3]

    # Generate impact_level from actual/previous data
    # If we have actual and previous values, we can estimate impact from change magnitude
    if 'impact_level' not in df.columns:
        print("Note: impact_level not in API data, deriving from change magnitude")
        df['impact_level'] = 1  # Default to medium

        # High impact: large absolute percentage change or specific event types
        high_impact_events = [
            'GDP', 'Employment', 'Unemployment', 'Non-Farm', 'Payrolls', 'CPI',
            'Interest Rate', 'FOMC', 'Fed', 'ECB', 'NFP', 'Retail Sales',
            'Consumer Price', 'Producer Price', 'PPI', 'Industrial Production'
        ]

        if 'event_type' in df.columns:
            for term in high_impact_events:
                mask = df['event_type'].str.contains(term, case=False, na=False)
                df.loc[mask, 'impact_level'] = 2

        # If change_percentage exists, use it to refine impact
        if 'change_percentage' in df.columns:
            # Large changes (>1%) = high impact
            df.loc[df['change_percentage'].abs() > 1.0, 'impact_level'] = 2
            # Small changes (<0.1%) = low impact
            df.loc[df['change_percentage'].abs() < 0.1, 'impact_level'] = 0
    else:
        # Standardize impact level to 0-2
        impact_mapping = {
            'Low': 0,
            'low': 0,
            'Medium': 1,
            'medium': 1,
            'High': 2,
            'high': 2,
            '0': 0,
            '1': 1,
            '2': 2,
            '3': 2,  # Map highest to 2
            0: 0,
            1: 1,
            2: 2,
            3: 2
        }
        df['impact_level'] = df['impact_level'].replace(impact_mapping)
        df['impact_level'] = pd.to_numeric(df['impact_level'], errors='coerce')

    # Filter out rows with missing critical fields
    required_fields = ['event_time', 'currency', 'event_type']

    df = df.dropna(subset=required_fields)

    # Sort by event time
    df = df.sort_values('event_time').reset_index(drop=True)

    print(f"✓ Processed {len(df):,} valid events")
    if len(df) > 0:
        print(f"✓ Event time range: {df['event_time'].min()} to {df['event_time'].max()}")
        print(f"✓ Currencies: {sorted(df['currency'].unique().tolist())}")
        print(f"✓ Impact levels: {sorted(df['impact_level'].unique().tolist())}")

    return df


def generate_event_features(price_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate event proximity features for each 1h bar.

    For each bar compute:
    - hours_to_next_event
    - hours_since_last_event
    - is_event_now (within current hour)
    - one-hot encode event currency
    - severity score (impact level)
    """
    if events_df.empty:
        print("Warning: No events available for feature generation")
        # Return price_df with null event features
        return price_df

    print("\nGenerating event proximity features...")

    # Ensure timestamp is datetime
    price_df = price_df.copy()
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)

    # Initialize feature columns
    price_df['hours_to_next_event'] = None
    price_df['hours_since_last_event'] = None
    price_df['is_event_now'] = 0
    price_df['event_severity'] = 0

    # Get unique currencies
    currencies = sorted(events_df['currency'].unique())
    for currency in currencies:
        price_df[f'event_currency_{currency}'] = 0

    # Process each bar
    for idx in tqdm(range(len(price_df)), desc="Computing event features"):
        bar_time = price_df.loc[idx, 'timestamp']

        # Find events within ±3 hours window
        time_window_start = bar_time - pd.Timedelta(hours=3)
        time_window_end = bar_time + pd.Timedelta(hours=3)

        nearby_events = events_df[
            (events_df['event_time'] >= time_window_start) &
            (events_df['event_time'] <= time_window_end)
        ]

        # Is event happening now? (within current hour)
        current_hour_events = events_df[
            (events_df['event_time'] >= bar_time) &
            (events_df['event_time'] < bar_time + pd.Timedelta(hours=1))
        ]

        if len(current_hour_events) > 0:
            price_df.loc[idx, 'is_event_now'] = 1
            # Get max severity of events this hour
            price_df.loc[idx, 'event_severity'] = current_hour_events['impact_level'].max()
            # One-hot encode currencies
            for currency in current_hour_events['currency'].unique():
                if f'event_currency_{currency}' in price_df.columns:
                    price_df.loc[idx, f'event_currency_{currency}'] = 1

        # Hours to next event
        future_events = events_df[events_df['event_time'] > bar_time]
        if len(future_events) > 0:
            next_event_time = future_events.iloc[0]['event_time']
            hours_to_next = (next_event_time - bar_time).total_seconds() / 3600
            price_df.loc[idx, 'hours_to_next_event'] = hours_to_next

        # Hours since last event
        past_events = events_df[events_df['event_time'] <= bar_time]
        if len(past_events) > 0:
            last_event_time = past_events.iloc[-1]['event_time']
            hours_since_last = (bar_time - last_event_time).total_seconds() / 3600
            price_df.loc[idx, 'hours_since_last_event'] = hours_since_last

    # Forward-fill features within ±3 hours window
    # For hours_to_next_event and hours_since_last_event, we'll limit fill to 3 hours
    price_df['hours_to_next_event'] = price_df['hours_to_next_event'].fillna(method='ffill', limit=3)
    price_df['hours_since_last_event'] = price_df['hours_since_last_event'].fillna(method='bfill', limit=3)

    # Fill remaining NaNs with large values (no event nearby)
    price_df['hours_to_next_event'] = price_df['hours_to_next_event'].fillna(999)
    price_df['hours_since_last_event'] = price_df['hours_since_last_event'].fillna(999)

    print(f"\n✓ Generated event features")
    print(f"✓ Event features columns: {[col for col in price_df.columns if 'event' in col or 'hours' in col]}")

    return price_df


def main():
    """Main pipeline execution"""

    print("="*80)
    print("TASK 3: Economic Calendar Pipeline")
    print("="*80)

    # Get project root directory (cross-platform)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Scripts are in task_XX subdirectories

    # Load existing price data with sessions
    price_file = project_root / 'data' / 'EURUSD_4H_2020_2025_with_sessions.csv'
    print(f"\nLoading price data from: {price_file}")
    price_df = pd.read_csv(price_file)
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)

    print(f"✓ Loaded {len(price_df):,} price bars")
    print(f"✓ Date range: {price_df['timestamp'].min()} to {price_df['timestamp'].max()}")

    # Determine date range for events
    start_date = price_df['timestamp'].min().to_pydatetime()
    end_date = price_df['timestamp'].max().to_pydatetime()

    # Step 1: Fetch economic events
    print("\n" + "="*80)
    print("Step 1: Fetching economic events from EODHD API")
    print("="*80)
    events_df = fetch_economic_events(start_date, end_date, API_KEY)

    # Inspect raw data
    if not events_df.empty:
        print("\n📊 Raw API Response Inspection:")
        print(f"Columns: {events_df.columns.tolist()}")
        print(f"\nFirst 3 rows:")
        print(events_df.head(3))
        print(f"\nData types:")
        print(events_df.dtypes)
        print(f"\nNull counts:")
        print(events_df.isnull().sum())

    # Step 2: Process events
    print("\n" + "="*80)
    print("Step 2: Processing economic events")
    print("="*80)
    events_df = process_economic_events(events_df)

    # Save raw events
    if not events_df.empty:
        events_file = project_root / 'data' / 'economic_events_2020_2025.csv'
        events_df.to_csv(events_file, index=False)
        print(f"\n✓ Events saved to: {events_file}")

    # Step 3: Generate features
    print("\n" + "="*80)
    print("Step 3: Generating event proximity features")
    print("="*80)
    price_df = generate_event_features(price_df, events_df)

    # Step 4: Save enhanced data
    output_file = project_root / 'data' / 'EURUSD_4H_2020_2025_with_events.csv'
    price_df.to_csv(output_file, index=False)

    print("\n" + "="*80)
    print("✅ TASK 3 COMPLETE")
    print("="*80)
    print(f"✓ Output file: {output_file}")
    print(f"✓ Total rows: {len(price_df):,}")
    print(f"✓ Total features: {len(price_df.columns)}")
    print(f"\n✓ Feature columns:")
    for col in price_df.columns:
        print(f"  - {col}")

    print("\n📊 Sample of event features:")
    event_cols = [col for col in price_df.columns if 'event' in col or 'hours' in col]
    print(price_df[['timestamp'] + event_cols].head(10))


if __name__ == "__main__":
    main()

