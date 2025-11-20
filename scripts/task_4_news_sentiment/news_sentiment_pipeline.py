#!/usr/bin/env python3
"""
TASK 4: News Sentiment Pipeline
Fetch news from EODHD API and extract LLM-based sentiment for currencies.
"""

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# EODHD API Configuration
API_KEY = os.environ.get("EODHD_API_TOKEN")
BASE_URL = 'https://eodhd.com/api/news'

# OpenRouter API for LLM sentiment analysis
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


def fetch_news_data(start_date: datetime, end_date: datetime, api_key: str) -> pd.DataFrame:
    """
    Fetch ALL news articles from EODHD API using proper offset-based pagination.

    Args:
        start_date: Start date for news
        end_date: End date for news
        api_key: EODHD API key

    Returns:
        DataFrame with news articles
    """
    all_news = []
    url = BASE_URL

    print(f"\nFetching news from EODHD API...")
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

    with tqdm(desc="Fetching news", unit=" batches") as pbar:
        while True:
            params = {
                'api_token': api_key,
                's': 'EURUSD.FOREX',  # Symbol for forex news
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

                    # Handle different response formats
                    if isinstance(data, list):
                        batch = data
                    elif isinstance(data, dict) and 'data' in data:
                        batch = data['data']
                    else:
                        print(f"\n⚠️  Unexpected response format at offset {offset}")
                        break

                    # Add batch to results
                    batch_size = len(batch)
                    all_news.extend(batch)
                    total_fetched += batch_size

                    pbar.set_postfix({"fetched": total_fetched, "offset": offset})
                    pbar.update(1)

                    # Check if we got fewer than limit (means we're done)
                    if batch_size < limit:
                        print(f"\n✓ Reached end of results (got {batch_size} \u003c {limit})")
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
                print(f"\n❌ Error fetching news at offset {offset}: {e}")
                break

    print(f"\n✅ Retrieved {total_fetched:,} total news articles")

    if not all_news:
        print("\n⚠️  Warning: No news retrieved from API")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_news)

    # Remove duplicates (in case of overlapping data)
    if 'title' in df.columns:
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['title'], keep='first')
        after_dedup = len(df)
        if before_dedup != after_dedup:
            print(f"🧹 Removed {before_dedup - after_dedup:,} duplicate articles")

    return df



def process_news_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw news data into standardized format.
    """
    if df.empty:
        return df

    print("\nProcessing news articles...")

    # Standardize timestamp
    if 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
    elif 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    elif 'published' in df.columns:
        df['timestamp'] = pd.to_datetime(df['published'], utc=True, errors='coerce')

    # Extract title and content
    if 'title' not in df.columns and 'headline' in df.columns:
        df['title'] = df['headline']

    # Drop rows without timestamp or title
    df = df.dropna(subset=['timestamp', 'title'])

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"✓ Processed {len(df):,} valid articles")
    if len(df) > 0:
        print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def extract_sentiment_with_llm(headlines: list[str], hour_timestamp: str) -> dict:
    """
    Use LLM via OpenRouter to extract sentiment for each currency from news headlines.

    Args:
        headlines: List of news headlines for the hour
        hour_timestamp: Timestamp string for logging

    Returns:
        Dict with sentiment scores: {sent_USD, sent_EUR, sent_GBP, sent_JPY, sent_risk}
    """
    if not headlines:
        return {
            'sent_USD': 0.0,
            'sent_EUR': 0.0,
            'sent_GBP': 0.0,
            'sent_JPY': 0.0,
            'sent_risk': 0.0
        }

    # Prepare prompt
    headlines_text = "\n".join([f"- {h}" for h in headlines[:20]])  # Limit to 20 headlines

    prompt = f"""Analyze these financial news headlines from {hour_timestamp} and extract sentiment for major currencies.

Headlines:
{headlines_text}

For each currency (USD, EUR, GBP, JPY), determine the sentiment from -1.0 (very negative) to +1.0 (very positive).
Also assess global risk sentiment: -1.0 (risk-off) to +1.0 (risk-on).

Consider:
- Positive indicators: economic growth, rate hikes, stability, positive data
- Negative indicators: recession fears, rate cuts, instability, negative data
- Risk-on: optimism, equity gains, low volatility
- Risk-off: fear, flight to safety, high volatility

Respond ONLY with a JSON object in this exact format:
{{"sent_USD": 0.0, "sent_EUR": 0.0, "sent_GBP": 0.0, "sent_JPY": 0.0, "sent_risk": 0.0}}

Do not include any explanation, just the JSON."""

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/swingtrader",
            "X-Title": "SwingTrader Sentiment Analysis"
        }

        payload = {
            "model": "google/gemini-2.5-flash-lite",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "max_tokens": 500
        }

        response = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            error_msg = f"API returned status code {response.status_code}"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg += f": {error_data['error']}"
            except:
                error_msg += f": {response.text[:200]}"
            raise Exception(error_msg)

        # Extract JSON from response
        response_data = response.json()
        response_text = response_data['choices'][0]['message']['content'].strip()

        # Try to extract JSON even if there's extra text
        # Look for JSON object in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            sentiment = json.loads(json_str)
        else:
            # If no JSON found, try parsing the whole thing
            sentiment = json.loads(response_text)

        # Validate and clamp values
        for key in ['sent_USD', 'sent_EUR', 'sent_GBP', 'sent_JPY', 'sent_risk']:
            if key not in sentiment:
                sentiment[key] = 0.0
            else:
                sentiment[key] = max(-1.0, min(1.0, float(sentiment[key])))

        return sentiment

    except json.JSONDecodeError as e:
        print(f"\nWarning: Failed to parse JSON for {hour_timestamp}")
        print(f"Response was: {response_text[:200] if 'response_text' in locals() else 'N/A'}")
        print(f"Error: {e}")
        return {
            'sent_USD': 0.0,
            'sent_EUR': 0.0,
            'sent_GBP': 0.0,
            'sent_JPY': 0.0,
            'sent_risk': 0.0
        }
    except Exception as e:
        print(f"\nWarning: LLM sentiment extraction failed for {hour_timestamp}: {e}")
        return {
            'sent_USD': 0.0,
            'sent_EUR': 0.0,
            'sent_GBP': 0.0,
            'sent_JPY': 0.0,
            'sent_risk': 0.0
        }


def generate_sentiment_features(price_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 4H sentiment features from news data.

    For each 4H bar:
    - Get all news headlines from that 4H window
    - Extract sentiment using LLM (one call per 4H bar with news)
    - Assign sentiment scores
    """
    if news_df.empty:
        print("\nWarning: No news available for sentiment analysis")
        # Return price_df with zero sentiment features
        price_df['sent_USD'] = 0.0
        price_df['sent_EUR'] = 0.0
        price_df['sent_GBP'] = 0.0
        price_df['sent_JPY'] = 0.0
        price_df['sent_risk'] = 0.0
        return price_df

    print("\nGenerating sentiment features...")

    # Ensure timestamps are datetime
    price_df = price_df.copy()
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)
    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'], utc=True)

    # Initialize sentiment columns
    price_df['sent_USD'] = 0.0
    price_df['sent_EUR'] = 0.0
    price_df['sent_GBP'] = 0.0
    price_df['sent_JPY'] = 0.0
    price_df['sent_risk'] = 0.0

    # Group news by 4H windows (aligned to price bars)
    news_df['bar_4h'] = news_df['timestamp'].dt.floor('4H')
    news_by_4h = news_df.groupby('bar_4h')['title'].apply(list).to_dict()

    print(f"✓ Grouped news into {len(news_by_4h)} 4H buckets")
    print(f"✓ Processing {len(news_by_4h)} 4H windows with news using 5 concurrent workers")

    # Concurrent LLM sentiment extraction
    def process_bar(bar_key, headlines):
        """Process a single 4H bar with LLM sentiment extraction."""
        sentiment = extract_sentiment_with_llm(headlines, bar_key.strftime('%Y-%m-%d %H:00'))
        return bar_key, sentiment

    # Use ThreadPoolExecutor for concurrent API calls
    sentiment_results = {}
    max_workers = 5  # 5 concurrent API calls

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_bar = {
            executor.submit(process_bar, bar_key, headlines): bar_key
            for bar_key, headlines in news_by_4h.items()
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(future_to_bar), desc="Extracting sentiment (concurrent)") as pbar:
            for future in as_completed(future_to_bar):
                try:
                    bar_key, sentiment = future.result()
                    sentiment_results[bar_key] = sentiment
                    pbar.update(1)
                except Exception as e:
                    bar_key = future_to_bar[future]
                    print(f"\n⚠️  Error processing {bar_key}: {e}")
                    # Use neutral sentiment on error
                    sentiment_results[bar_key] = {
                        'sent_USD': 0.0,
                        'sent_EUR': 0.0,
                        'sent_GBP': 0.0,
                        'sent_JPY': 0.0,
                        'sent_risk': 0.0
                    }
                    pbar.update(1)

    # Apply sentiment results to price bars
    print("\n✓ Applying sentiment to price bars...")
    for bar_key, sentiment in sentiment_results.items():
        # Find all bars that match this 4H window
        matching_bars = price_df[price_df['timestamp'].dt.floor('4H') == bar_key].index

        for idx in matching_bars:
            price_df.loc[idx, 'sent_USD'] = sentiment['sent_USD']
            price_df.loc[idx, 'sent_EUR'] = sentiment['sent_EUR']
            price_df.loc[idx, 'sent_GBP'] = sentiment['sent_GBP']
            price_df.loc[idx, 'sent_JPY'] = sentiment['sent_JPY']
            price_df.loc[idx, 'sent_risk'] = sentiment['sent_risk']

    # Apply exponential decay: sentiment fades over 3 days (18 bars of 4H) after news
    print("\n✓ Applying sentiment decay (3-day half-life for 4H bars)...")
    decay_bars = 18  # 3 days = 18 * 4H bars

    for col in ['sent_USD', 'sent_EUR', 'sent_GBP', 'sent_JPY', 'sent_risk']:
        # Forward fill with decay
        last_value = 0.0
        bars_since_last = 999

        for idx in range(len(price_df)):
            current_value = price_df.loc[idx, col]

            if current_value != 0.0:
                # New sentiment event
                last_value = current_value
                bars_since_last = 0
            else:
                # Apply decay
                bars_since_last += 1
                if bars_since_last <= decay_bars and last_value != 0.0:
                    # Exponential decay: value * exp(-bars/half_life)
                    decay_factor = 0.5 ** (bars_since_last / decay_bars)
                    price_df.loc[idx, col] = last_value * decay_factor
                else:
                    # Reset if too far from last event
                    last_value = 0.0
                    bars_since_last = 999

    print(f"\n✓ Generated sentiment features")
    print(f"✓ Sentiment statistics:")
    for col in ['sent_USD', 'sent_EUR', 'sent_GBP', 'sent_JPY', 'sent_risk']:
        mean = price_df[col].mean()
        std = price_df[col].std()
        print(f"  {col}: mean={mean:.3f}, std={std:.3f}")

    return price_df


def main():
    """Main pipeline execution"""

    print("="*80)
    print("TASK 4: News Sentiment Pipeline")
    print("="*80)

    # Check for OpenRouter API key
    if not OPENROUTER_API_KEY:
        print("\n⚠️  WARNING: OPENROUTER_API_KEY not found in environment")
        print("Please set OPENROUTER_API_KEY to use LLM sentiment extraction")
        print("Proceeding with neutral sentiment fallback...")

    # Get project root directory (cross-platform)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # Scripts are in task_XX subdirectories

    # Load existing price data with events
    price_file = project_root / 'data' / 'EURUSD_4H_2020_2025_with_events.csv'
    print(f"\nLoading price data from: {price_file}")
    price_df = pd.read_csv(price_file)
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)

    print(f"✓ Loaded {len(price_df):,} price bars")
    print(f"✓ Date range: {price_df['timestamp'].min()} to {price_df['timestamp'].max()}")

    # Determine date range for news
    start_date = price_df['timestamp'].min().to_pydatetime()
    end_date = price_df['timestamp'].max().to_pydatetime()

    # Check if cached news file exists
    news_file = project_root / 'data' / 'news_articles_2020_2025.csv'

    if os.path.exists(news_file):
        print("\n" + "="*80)
        print("Step 1: Loading cached news data")
        print("="*80)
        print(f"\n✓ Found cached news file: {news_file}")
        news_df = pd.read_csv(news_file)
        news_df['timestamp'] = pd.to_datetime(news_df['timestamp'], utc=True)
        print(f"✓ Loaded {len(news_df):,} cached articles")
        print(f"✓ Date range: {news_df['timestamp'].min()} to {news_df['timestamp'].max()}")
        print("\n💡 To re-fetch news, delete the cache file and run again")
    else:
        # Step 1: Fetch news
        print("\n" + "="*80)
        print("Step 1: Fetching news from EODHD API")
        print("="*80)
        news_df = fetch_news_data(start_date, end_date, API_KEY)

        # Inspect raw data
        if not news_df.empty:
            print("\n📊 Raw API Response Inspection:")
            print(f"Columns: {news_df.columns.tolist()}")
            print(f"\nFirst 3 articles:")
            print(news_df.head(3))
            print(f"\nNull counts:")
            print(news_df.isnull().sum())

        # Step 2: Process news
        print("\n" + "="*80)
        print("Step 2: Processing news articles")
        print("="*80)
        news_df = process_news_data(news_df)

        # Save news cache
        if not news_df.empty:
            news_df.to_csv(news_file, index=False)
            print(f"\n✓ News cached to: {news_file}")

    # Check if sentiment-enhanced data already exists
    output_file = project_root / 'data' / 'EURUSD_4H_2020_2025_with_sentiment.csv'

    if os.path.exists(output_file):
        print("\n" + "="*80)
        print("Step 3: Loading cached sentiment data")
        print("="*80)
        print(f"\n✓ Found cached sentiment file: {output_file}")
        price_df = pd.read_csv(output_file)
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)
        print(f"✓ Loaded {len(price_df):,} bars with sentiment features")
        print("\n💡 To regenerate sentiment, delete the cache file and run again")
    else:
        # Step 3: Generate sentiment features
        print("\n" + "="*80)
        print("Step 3: Generating sentiment features with LLM (via OpenRouter)")
        print("="*80)

        if OPENROUTER_API_KEY:
            price_df = generate_sentiment_features(price_df, news_df)
        else:
            print("\nSkipping LLM sentiment extraction (no API key)")
            print("Setting all sentiment features to neutral (0.0)")
            price_df['sent_USD'] = 0.0
            price_df['sent_EUR'] = 0.0
            price_df['sent_GBP'] = 0.0
            price_df['sent_JPY'] = 0.0
            price_df['sent_risk'] = 0.0

        # Save enhanced data
        price_df.to_csv(output_file, index=False)
        print(f"\n✓ Sentiment data cached to: {output_file}")

    print("\n" + "="*80)
    print("✅ TASK 4 COMPLETE")
    print("="*80)
    print(f"✓ Output file: {output_file}")
    print(f"✓ Total rows: {len(price_df):,}")
    print(f"✓ Total features: {len(price_df.columns)}")

    # Check if sentiment columns exist
    sentiment_cols = [col for col in price_df.columns if 'sent_' in col]
    if sentiment_cols:
        print(f"\n✓ Sentiment feature columns:")
        for col in sentiment_cols:
            print(f"  - {col}")

        print("\n📊 Sample of sentiment features:")
        print(price_df[['timestamp'] + sentiment_cols].head(20))

        print("\n📊 Sentiment statistics:")
        print(price_df[sentiment_cols].describe())
    else:
        print("\n⚠️  No sentiment columns found in output")

    print("\n" + "="*80)
    print("Cache Management:")
    print("="*80)
    print(f"News cache: {news_file}")
    print(f"Sentiment cache: {output_file}")
    print("\nTo regenerate from scratch, delete these files and run again.")


if __name__ == "__main__":
    main()

